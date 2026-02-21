"""Corpus builder for programmatic protein dataset expansion.

Maps custom spectral archetypes to structural database classifications
(CATH superfamily/topology) and queries RCSB PDB for additional proteins.
Part of Phase A of the GNCA infrastructure buildout.

Archetype → CATH mapping
-------------------------
  barrel        → CATH 3.20.20 (TIM barrel), 2.40.128 (lipocalin)
  globin        → CATH 1.10.490 (globin), 1.20.120 (4-helix bundle)
  dumbbell      → CATH 3.40.190 (periplasmic BP) + keyword search
  enzyme_active → EC-classified enzymes + keyword search
  allosteric    → CATH 3.40.50.300 (P-loop NTPase) + keyword search

Usage
-----
>>> from ibp_enm.corpus_builder import CorpusBuilder
>>> builder = CorpusBuilder()
>>> report = builder.build(target_per_archetype=200)
>>> report.save("gnca_candidates.json")
>>> print(report.summary())

History
-------
D137a — Phase A corpus expansion for GNCA training.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from .benchmark import LARGE_CORPUS, ProteinEntry

__all__ = [
    "CandidateProtein",
    "CorpusBuildReport",
    "CorpusBuilder",
    "ARCHETYPE_SEARCH_CONFIG",
]


# ═══════════════════════════════════════════════════════════════════
# Search configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CATHQuery:
    """CATH classification query for RCSB search API.

    Uses ``annotation_lineage.name`` with ``contains_phrase``
    because RCSB does not index lineage IDs for search.
    """
    cath_name: str   # searched via annotation_lineage.name
    cath_id: str     # informational (e.g. "3.20.20")
    description: str


@dataclass(frozen=True)
class KeywordQuery:
    """Full-text keyword query for RCSB search API."""
    terms: str
    description: str


@dataclass
class ArchetypeSearchConfig:
    """Search configuration for one archetype.

    Combines CATH-based structural queries with keyword searches
    to cast a wide net, then applies size and quality filters.
    """
    archetype: str
    description: str = ""
    cath_queries: List[CATHQuery] = field(default_factory=list)
    keyword_queries: List[KeywordQuery] = field(default_factory=list)
    require_ec: bool = False
    min_residues: int = 50
    max_residues: int = 800


# ── Per-archetype search strategies ─────────────────────────────

ARCHETYPE_SEARCH_CONFIG: Dict[str, ArchetypeSearchConfig] = {

    "barrel": ArchetypeSearchConfig(
        archetype="barrel",
        description=(
            "Rigid cylindrical scaffolds: TIM barrels (α/β barrel, "
            "CATH 3.20.20), lipocalin β-barrels (CATH 2.40.128), "
            "jelly-roll barrels"
        ),
        cath_queries=[
            CATHQuery("TIM Barrel", "3.20.20", "TIM barrel topology"),
            CATHQuery("Lipocalin", "2.40.128", "Lipocalin β-barrel superfamily"),
        ],
        keyword_queries=[
            KeywordQuery("TIM barrel enzyme", "TIM barrel keyword"),
            KeywordQuery(
                "beta barrel lipocalin retinol",
                "β-barrel / lipocalin keyword",
            ),
        ],
        min_residues=80,
        max_residues=650,
    ),

    "globin": ArchetypeSearchConfig(
        archetype="globin",
        description=(
            "Helical bundles: globins (CATH 1.10.490), "
            "4-helix bundle proteins (CATH 1.20.120), "
            "cytochromes, ferritins"
        ),
        cath_queries=[
            CATHQuery("Globin", "1.10.490", "Globin fold topology"),
            CATHQuery("Four Helix Bundle", "1.20.120", "Four helix bundle topology"),
        ],
        keyword_queries=[
            KeywordQuery(
                "globin myoglobin hemoglobin",
                "Globin family keyword",
            ),
            KeywordQuery(
                "four helix bundle cytokine ferritin",
                "4-helix bundle keyword",
            ),
        ],
        min_residues=50,
        max_residues=300,
    ),

    "dumbbell": ArchetypeSearchConfig(
        archetype="dumbbell",
        description=(
            "Multi-domain with hinge: periplasmic binding proteins "
            "(CATH 3.40.190), kinases, transferrins, multi-domain "
            "enzymes with inter-domain linkers"
        ),
        cath_queries=[
            CATHQuery("Periplasmic", "3.40.190", "Periplasmic binding protein domain"),
        ],
        keyword_queries=[
            KeywordQuery(
                "periplasmic binding protein",
                "Periplasmic BP keyword",
            ),
            KeywordQuery(
                "two-domain hinge bending cleft closure",
                "Hinge / domain motion keyword",
            ),
            KeywordQuery(
                "transferrin lactoferrin ovotransferrin",
                "Transferrin family keyword",
            ),
        ],
        min_residues=200,
        max_residues=800,
    ),

    "enzyme_active": ArchetypeSearchConfig(
        archetype="enzyme_active",
        description=(
            "Single-domain enzymes with localised active site: "
            "proteases, nucleases, lipases, oxidoreductases. "
            "Filtered by EC number presence."
        ),
        cath_queries=[],  # rely on EC filter + keywords
        keyword_queries=[
            KeywordQuery(
                "hydrolase protease nuclease active site",
                "Hydrolase keyword",
            ),
            KeywordQuery(
                "oxidoreductase catalytic mechanism",
                "Oxidoreductase keyword",
            ),
            KeywordQuery(
                "lipase esterase phosphatase lysozyme",
                "Hydrolase-2 keyword",
            ),
        ],
        require_ec=True,
        min_residues=80,
        max_residues=400,
    ),

    "allosteric": ArchetypeSearchConfig(
        archetype="allosteric",
        description=(
            "Signal-coupled domains: GTPases / P-loop NTPases "
            "(CATH 3.40.50.300), response regulators, allosteric "
            "enzymes with conformational switching"
        ),
        cath_queries=[
            CATHQuery(
                "P-loop",
                "3.40.50.300",
                "P-loop NTPase / GTPase superfamily",
            ),
        ],
        keyword_queries=[
            KeywordQuery(
                "allosteric regulation conformational change",
                "Allosteric keyword",
            ),
            KeywordQuery(
                "GTPase signal transduction switch",
                "GTPase keyword",
            ),
            KeywordQuery(
                "response regulator receiver domain",
                "Response regulator keyword",
            ),
        ],
        min_residues=80,
        max_residues=600,
    ),
}


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CandidateProtein:
    """A protein candidate discovered by the corpus builder."""

    pdb_id: str
    chain: str
    name: str
    archetype: str
    resolution: Optional[float] = None
    n_residues: Optional[int] = None
    uniprot_id: Optional[str] = None
    source_query: str = ""
    confidence: str = "medium"  # high / medium / low

    def to_protein_entry(self) -> ProteinEntry:
        """Convert to ProteinEntry for direct benchmark use."""
        safe_name = (
            self.name.replace(" ", "_")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")[:40]
        )
        return ProteinEntry(safe_name, self.pdb_id, self.chain, self.archetype)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CorpusBuildReport:
    """Results of a full corpus-builder run."""

    candidates: Dict[str, List[CandidateProtein]]
    total_searched: int = 0
    excluded_existing: int = 0
    excluded_quality: int = 0
    excluded_duplicate: int = 0
    search_errors: List[str] = field(default_factory=list)
    timestamp: str = ""

    # ── Derived properties ───────────────────────────────────────

    @property
    def total_candidates(self) -> int:
        return sum(len(v) for v in self.candidates.values())

    @property
    def per_archetype_counts(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self.candidates.items()}

    # ── Display ──────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            "",
            "═══════════════════════════════════════════",
            "  Corpus Build Report",
            "═══════════════════════════════════════════",
            f"  Timestamp:              {self.timestamp}",
            f"  Total PDB IDs searched: {self.total_searched}",
            f"  Total candidates:       {self.total_candidates}",
            f"  Excluded (existing):    {self.excluded_existing}",
            f"  Excluded (quality):     {self.excluded_quality}",
            f"  Excluded (duplicates):  {self.excluded_duplicate}",
        ]
        if self.search_errors:
            lines.append(f"  Search errors:          {len(self.search_errors)}")
        lines.append("")
        lines.append("  Per-archetype breakdown:")
        for arch, cands in self.candidates.items():
            n_high = sum(1 for c in cands if c.confidence == "high")
            n_med = sum(1 for c in cands if c.confidence == "medium")
            lines.append(
                f"    {arch:20s}: {len(cands):4d}  "
                f"(high={n_high}, medium={n_med})"
            )
        lines.append("═══════════════════════════════════════════")
        return "\n".join(lines)

    # ── Serialisation ────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save report to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": self.timestamp,
            "total_searched": self.total_searched,
            "total_candidates": self.total_candidates,
            "excluded_existing": self.excluded_existing,
            "excluded_quality": self.excluded_quality,
            "excluded_duplicate": self.excluded_duplicate,
            "search_errors": self.search_errors,
            "per_archetype_counts": self.per_archetype_counts,
            "candidates": {
                arch: [c.to_dict() for c in cands]
                for arch, cands in self.candidates.items()
            },
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CorpusBuildReport":
        """Load report from JSON."""
        data = json.loads(Path(path).read_text())
        candidates = {
            arch: [CandidateProtein(**c) for c in cands]
            for arch, cands in data["candidates"].items()
        }
        return cls(
            candidates=candidates,
            total_searched=data.get("total_searched", 0),
            excluded_existing=data.get("excluded_existing", 0),
            excluded_quality=data.get("excluded_quality", 0),
            excluded_duplicate=data.get("excluded_duplicate", 0),
            search_errors=data.get("search_errors", []),
            timestamp=data.get("timestamp", ""),
        )

    # ── Code generation ──────────────────────────────────────────

    def generate_python(self, var_prefix: str = "_GNCA") -> str:
        """Generate Python source for ProteinEntry lists.

        Produces code that can be pasted directly into benchmark.py.
        """
        lines: List[str] = []
        for arch, cands in self.candidates.items():
            varname = f"{var_prefix}_{arch.upper()}"
            lines.append(f"{varname} = [")
            for c in cands:
                entry = c.to_protein_entry()
                res_str = f"{c.resolution:.1f}Å" if c.resolution else "?"
                n_str = f"N={c.n_residues}" if c.n_residues else "N=?"
                lines.append(
                    f'    ProteinEntry("{entry.name}", '
                    f'"{entry.pdb_id}", "{entry.chain}", '
                    f'"{entry.archetype}"),  '
                    f"# {res_str}, {n_str}"
                )
            lines.append("]")
            lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# CorpusBuilder
# ═══════════════════════════════════════════════════════════════════

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"


class CorpusBuilder:
    """Orchestrates RCSB search, filtering, deduplication, and output.

    Parameters
    ----------
    max_resolution : float
        Maximum X-ray resolution (Å).  Default 2.5.
    max_results_per_query : int
        Cap on each individual RCSB search query.
    request_delay : float
        Seconds between API requests (rate limiting).
    verbose : bool
        Print progress to stdout.
    """

    def __init__(
        self,
        max_resolution: float = 2.5,
        max_results_per_query: int = 500,
        request_delay: float = 0.5,
        verbose: bool = True,
    ):
        self.max_resolution = max_resolution
        self.max_results_per_query = max_results_per_query
        self.request_delay = request_delay
        self.verbose = verbose

        # Existing corpus for exclusion
        self._existing_pdb_ids: Set[str] = {
            e.pdb_id.upper() for e in LARGE_CORPUS
        }

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ── RCSB Search API helpers ──────────────────────────────────

    def _base_quality_nodes(self) -> list:
        """Quality-filter nodes shared by all searches."""
        return [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "exptl.method",
                    "operator": "exact_match",
                    "value": "X-RAY DIFFRACTION",
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal",
                    "value": self.max_resolution,
                },
            },
        ]

    def _search_by_cath(self, cath_id: str) -> List[str]:
        """Search RCSB for entries with a given CATH classification.

        Parameters
        ----------
        cath_id : str
            Actually the CATH *name* (e.g. ``"TIM Barrel"``).

        Uses ``annotation_lineage.name`` with ``contains_phrase``
        because RCSB indexes CATH lineage names but not raw IDs
        in their search API.
        """
        nodes = self._base_quality_nodes() + [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": (
                        "rcsb_polymer_entity_annotation"
                        ".annotation_lineage.name"
                    ),
                    "operator": "contains_phrase",
                    "value": cath_id,  # actually the CATH name
                },
            },
        ]
        return self._execute_search(nodes)

    def _search_by_keywords(
        self,
        terms: str,
        require_ec: bool = False,
    ) -> List[str]:
        """Full-text search across PDB title, abstract, keywords."""
        nodes = self._base_quality_nodes() + [
            {
                "type": "terminal",
                "service": "full_text",
                "parameters": {
                    "value": terms,
                },
            },
        ]
        if require_ec:
            nodes.append(
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": (
                            "rcsb_polymer_entity"
                            ".rcsb_ec_lineage.id"
                        ),
                        "operator": "exists",
                    },
                }
            )
        return self._execute_search(nodes)

    def _execute_search(self, nodes: list) -> List[str]:
        """POST a search to RCSB and return PDB IDs."""
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": nodes,
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "start": 0,
                    "rows": self.max_results_per_query,
                },
                "sort": [
                    {
                        "sort_by": (
                            "rcsb_entry_info.resolution_combined"
                        ),
                        "direction": "asc",
                    }
                ],
            },
        }

        time.sleep(self.request_delay)
        try:
            resp = requests.post(
                RCSB_SEARCH_URL, json=query, timeout=30,
            )
            if resp.status_code == 204:
                # No results
                return []
            resp.raise_for_status()
            data = resp.json()
            return [
                hit["identifier"]
                for hit in data.get("result_set", [])
            ]
        except requests.exceptions.HTTPError as exc:
            self._log(f"    ⚠ RCSB search HTTP error: {exc}")
            # Log the response body for debugging
            try:
                self._log(f"    Response: {exc.response.text[:200]}")
            except Exception:
                pass
            return []
        except Exception as exc:
            self._log(f"    ⚠ RCSB search error: {exc}")
            return []

    # ── Metadata fetching ────────────────────────────────────────

    def _fetch_metadata_batch(
        self, pdb_ids: List[str],
    ) -> Dict[str, dict]:
        """Batch-fetch title, resolution, chain, UniProt via GraphQL.

        Processes in batches of 40 to stay within API limits.
        Falls back to individual REST queries on GraphQL failure.
        """
        all_meta: Dict[str, dict] = {}
        batch_size = 40

        for i in range(0, len(pdb_ids), batch_size):
            batch = pdb_ids[i : i + batch_size]
            self._log(
                f"    Metadata batch {i // batch_size + 1}"
                f" ({len(batch)} entries)..."
            )
            meta = self._graphql_batch(batch)
            if meta:
                all_meta.update(meta)
            else:
                # Fallback to individual REST
                self._log("    Falling back to individual REST...")
                for pid in batch:
                    m = self._rest_single(pid)
                    if m:
                        all_meta[pid] = m

        return all_meta

    def _graphql_batch(
        self, pdb_ids: List[str],
    ) -> Optional[Dict[str, dict]]:
        """Fetch metadata for a batch via RCSB GraphQL."""
        gql = """
        query ($ids: [String!]!) {
          entries(entry_ids: $ids) {
            rcsb_id
            struct {
              title
            }
            rcsb_entry_info {
              resolution_combined
              deposited_polymer_monomer_count
            }
            polymer_entities {
              rcsb_polymer_entity_container_identifiers {
                uniprot_ids
                auth_asym_ids
                entity_id
              }
              entity_poly {
                rcsb_sample_sequence_length
              }
            }
          }
        }
        """

        time.sleep(self.request_delay)
        try:
            resp = requests.post(
                RCSB_GRAPHQL_URL,
                json={"query": gql, "variables": {"ids": pdb_ids}},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            self._log(f"    ⚠ GraphQL error: {exc}")
            return None

        result: Dict[str, dict] = {}
        entries = data.get("data", {}).get("entries") or []
        for entry in entries:
            if entry is None:
                continue
            pdb_id = entry.get("rcsb_id", "")
            info = entry.get("rcsb_entry_info") or {}
            struct = entry.get("struct") or {}

            # Resolve primary protein entity → chain + UniProt
            chain = "A"
            uniprot_id = None
            seq_length = None

            for ent in entry.get("polymer_entities") or []:
                if ent is None:
                    continue
                ids = (
                    ent.get(
                        "rcsb_polymer_entity_container_identifiers"
                    )
                    or {}
                )
                poly = ent.get("entity_poly") or {}

                auth_chains = ids.get("auth_asym_ids") or []
                if auth_chains:
                    chain = auth_chains[0]

                uniprots = ids.get("uniprot_ids") or []
                if uniprots:
                    uniprot_id = uniprots[0]

                sl = poly.get("rcsb_sample_sequence_length")
                if sl:
                    seq_length = sl

                break  # use first protein entity

            # resolution_combined is a list in GraphQL
            raw_res = info.get("resolution_combined")
            if isinstance(raw_res, list) and raw_res:
                resolution = raw_res[0]
            elif isinstance(raw_res, (int, float)):
                resolution = float(raw_res)
            else:
                resolution = None

            result[pdb_id] = {
                "title": (struct.get("title") or "").strip(),
                "resolution": resolution,
                "n_residues": (
                    seq_length
                    or info.get("deposited_polymer_monomer_count")
                ),
                "chain": chain,
                "uniprot_id": uniprot_id,
            }

        return result

    def _rest_single(self, pdb_id: str) -> Optional[dict]:
        """Fetch metadata for one entry via REST (fallback)."""
        url = (
            f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        )
        time.sleep(self.request_delay)
        try:
            resp = requests.get(url, timeout=15)
            if not resp.ok:
                return None
            data = resp.json()
            info = data.get("rcsb_entry_info") or {}
            struct = data.get("struct") or {}
            return {
                "title": (struct.get("title") or "").strip(),
                "resolution": info.get("resolution_combined"),
                "n_residues": info.get(
                    "deposited_polymer_monomer_count"
                ),
                "chain": "A",
                "uniprot_id": None,
            }
        except Exception:
            return None

    # ── Candidate building ───────────────────────────────────────

    @staticmethod
    def _clean_name(title: str, pdb_id: str) -> str:
        """Build a ProteinEntry-style name from PDB title."""
        if not title:
            return f"Protein_{pdb_id}"
        # Normalise whitespace, take first ~35 chars
        name = " ".join(title.split())[:35].strip()
        # Convert to identifier
        name = (
            name.replace(" ", "_")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .replace("'", "")
            .replace('"', "")
            .replace("/", "_")
            .replace("-", "_")
            .replace(".", "")
            .replace(":", "_")
        )
        # Collapse multiple underscores
        while "__" in name:
            name = name.replace("__", "_")
        name = name.strip("_")
        # Append PDB ID for uniqueness
        return f"{name}_{pdb_id}"

    def _deduplicate(
        self, candidates: List[CandidateProtein],
    ) -> tuple[List[CandidateProtein], int]:
        """Remove duplicates (same UniProt or same PDB ID).

        Keeps the highest-resolution representative.
        """
        # Sort best resolution first
        sorted_cands = sorted(
            candidates,
            key=lambda c: c.resolution if c.resolution else 99.0,
        )

        seen_uniprot: Set[str] = set()
        seen_pdb: Set[str] = set()
        kept: List[CandidateProtein] = []
        n_removed = 0

        for c in sorted_cands:
            pdb_upper = c.pdb_id.upper()
            if pdb_upper in seen_pdb:
                n_removed += 1
                continue
            seen_pdb.add(pdb_upper)

            if c.uniprot_id:
                if c.uniprot_id in seen_uniprot:
                    n_removed += 1
                    continue
                seen_uniprot.add(c.uniprot_id)

            kept.append(c)

        return kept, n_removed

    # ── Per-archetype search ─────────────────────────────────────

    def search_archetype(
        self, archetype: str,
    ) -> tuple[List[CandidateProtein], int, int, int, List[str]]:
        """Run all search strategies for one archetype.

        Returns
        -------
        candidates : list[CandidateProtein]
        n_total_ids : int
            Raw count before filtering.
        n_excluded_existing : int
        n_excluded_quality : int
        errors : list[str]
        """
        config = ARCHETYPE_SEARCH_CONFIG[archetype]

        self._log(f"\n{'─' * 60}")
        self._log(f"Archetype: {archetype}")
        self._log(f"  {config.description}")

        pdb_id_sources: Dict[str, str] = {}
        errors: List[str] = []

        # ── CATH queries ──
        for cq in config.cath_queries:
            self._log(
                f"  → CATH '{cq.cath_name}' [{cq.cath_id}] "
                f"({cq.description})..."
            )
            try:
                ids = self._search_by_cath(cq.cath_name)
                self._log(f"    {len(ids)} entries")
                for pid in ids:
                    if pid not in pdb_id_sources:
                        pdb_id_sources[pid] = (
                            f"CATH:{cq.cath_name}"
                        )
            except Exception as exc:
                msg = f"CATH '{cq.cath_name}': {exc}"
                errors.append(msg)
                self._log(f"    ⚠ {msg}")

        # ── Keyword queries ──
        for kq in config.keyword_queries:
            self._log(
                f"  → Keyword: '{kq.terms}' ({kq.description})..."
            )
            try:
                ids = self._search_by_keywords(
                    kq.terms, require_ec=config.require_ec,
                )
                self._log(f"    {len(ids)} entries")
                for pid in ids:
                    if pid not in pdb_id_sources:
                        pdb_id_sources[pid] = (
                            f"keyword:{kq.terms[:25]}"
                        )
            except Exception as exc:
                msg = f"Keyword '{kq.terms}': {exc}"
                errors.append(msg)
                self._log(f"    ⚠ {msg}")

        all_ids = list(pdb_id_sources.keys())
        n_total = len(all_ids)
        self._log(f"  Unique PDB IDs from search: {n_total}")

        if not all_ids:
            return [], n_total, 0, 0, errors

        # ── Fetch metadata ──
        self._log("  Fetching metadata...")
        metadata = self._fetch_metadata_batch(all_ids)

        # ── Filter and build candidates ──
        candidates: List[CandidateProtein] = []
        n_excluded_existing = 0
        n_excluded_quality = 0

        for pid in all_ids:
            meta = metadata.get(pid, {})
            source = pdb_id_sources[pid]

            # Exclude existing corpus
            if pid.upper() in self._existing_pdb_ids:
                n_excluded_existing += 1
                continue

            # Resolution
            res = meta.get("resolution")
            if res is not None and res > self.max_resolution:
                n_excluded_quality += 1
                continue

            # Size bounds
            n_res = meta.get("n_residues")
            if n_res is not None:
                if (n_res < config.min_residues
                        or n_res > config.max_residues):
                    n_excluded_quality += 1
                    continue

            title = meta.get("title", pid)
            name = self._clean_name(title, pid)
            confidence = (
                "high" if source.startswith("CATH") else "medium"
            )

            candidates.append(
                CandidateProtein(
                    pdb_id=pid,
                    chain=meta.get("chain", "A"),
                    name=name,
                    archetype=archetype,
                    resolution=res,
                    n_residues=n_res,
                    uniprot_id=meta.get("uniprot_id"),
                    source_query=source,
                    confidence=confidence,
                )
            )

        self._log(
            f"  After filters: {len(candidates)} candidates  "
            f"(excl existing={n_excluded_existing}, "
            f"quality={n_excluded_quality})"
        )

        # ── Deduplicate ──
        candidates, n_dedup = self._deduplicate(candidates)
        self._log(
            f"  After dedup:   {len(candidates)} candidates  "
            f"(removed {n_dedup} duplicates)"
        )

        return (
            candidates,
            n_total,
            n_excluded_existing,
            n_excluded_quality,
            errors,
        )

    # ── Main entry point ─────────────────────────────────────────

    def build(
        self,
        archetypes: Optional[List[str]] = None,
        target_per_archetype: int = 200,
    ) -> CorpusBuildReport:
        """Full pipeline: search → filter → dedup → cap → report.

        Parameters
        ----------
        archetypes : list[str] | None
            Archetypes to search.  Default: all five.
        target_per_archetype : int
            Maximum candidates kept per archetype (best resolution
            first).
        """
        import datetime

        if archetypes is None:
            archetypes = list(ARCHETYPE_SEARCH_CONFIG.keys())

        self._log(
            "\n═══════════════════════════════════════════════════"
        )
        self._log("  Corpus Builder — Phase A (D137a)")
        self._log(f"  Archetypes: {', '.join(archetypes)}")
        self._log(f"  Target per archetype: {target_per_archetype}")
        self._log(f"  Max resolution: {self.max_resolution} Å")
        self._log(
            f"  Max per query: {self.max_results_per_query}"
        )
        self._log(
            f"  Existing corpus: {len(self._existing_pdb_ids)} PDB IDs"
        )
        self._log(
            "═══════════════════════════════════════════════════"
        )

        all_candidates: Dict[str, List[CandidateProtein]] = {}
        all_errors: List[str] = []
        total_searched = 0
        total_excl_existing = 0
        total_excl_quality = 0
        total_excl_dedup = 0

        for arch in archetypes:
            (
                cands,
                n_searched,
                n_excl_exist,
                n_excl_qual,
                errors,
            ) = self.search_archetype(arch)

            total_searched += n_searched
            total_excl_existing += n_excl_exist
            total_excl_quality += n_excl_qual
            all_errors.extend(errors)

            # Cap to target (keep best resolution)
            if len(cands) > target_per_archetype:
                cands = sorted(
                    cands,
                    key=lambda c: (
                        c.resolution if c.resolution else 99.0
                    ),
                )[:target_per_archetype]
                self._log(
                    f"  → Capped to {target_per_archetype}"
                )

            all_candidates[arch] = cands

        report = CorpusBuildReport(
            candidates=all_candidates,
            total_searched=total_searched,
            excluded_existing=total_excl_existing,
            excluded_quality=total_excl_quality,
            excluded_duplicate=total_excl_dedup,
            search_errors=all_errors,
            timestamp=datetime.datetime.now().isoformat(),
        )

        self._log(report.summary())

        return report

    # ── Convenience ────────────────────────────────────────────

    def to_protein_entries(
        self, report: CorpusBuildReport,
    ) -> List[ProteinEntry]:
        """Convert all candidates across archetypes to
        ``ProteinEntry`` objects for direct benchmark use."""
        entries: List[ProteinEntry] = []
        for cands in report.candidates.values():
            for c in cands:
                entries.append(c.to_protein_entry())
        return entries
