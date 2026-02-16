"""Functional site resolution for protein structures.

Replaces the D96 hardcoded approach with automated annotation
fetching from UniProt + PDBe SIFTS, plus a curated benchmark
database for our 8 well-characterised proteins.

Three resolution strategies (tried in order):
  1. User-provided annotations (highest priority)
  2. UniProt REST API + PDBe SIFTS mapping (automated, comprehensive)
  3. Built-in benchmark database (fallback for known proteins)

The resolver maps UniProt feature annotations to 0-based Cα indices
matching the IBPProteinAnalyzer's coordinate arrays.

Categories:
  active_site  — Catalytic residues (UniProt: Active site)
  binding      — Substrate/cofactor contacts (UniProt: Binding site)
  metal        — Metal coordination (UniProt: Metal binding)
  mutagenesis  — Experimentally characterised mutations (UniProt: Mutagenesis)
  regulatory   — Allosteric/regulatory regions (UniProt: Region with keywords)
  domain_boundary — Inferred hinges: ±3 residues from domain transitions
  disulfide    — Structural disulfide bonds (UniProt: Disulfide bond)

Composite categories (computed):
  all_functional — Union of all above
  mechanical     — domain_boundary (hinge-like, controls motion)
  chemical       — active_site ∪ binding ∪ metal (controls chemistry)

Usage:
    from ibp_enm.functional_sites import FunctionalSiteResolver

    resolver = FunctionalSiteResolver()
    ann = resolver.resolve("2LAO", chain="A", n_residues=238)
    print(ann.coverage)          # 0.13
    print(ann.active_site)       # {10, 11, 12, ...}
    print(ann.as_d96_format())   # backwards-compatible dict
"""

import json
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List, Any


@dataclass
class FunctionalAnnotation:
    """Functional site annotation for a single protein structure.

    All residue indices are 0-based, matching IBPProteinAnalyzer's
    coordinate array ordering (i.e., the order Cα atoms appear in
    the mmCIF file for the specified chain).
    """
    pdb_id: str = ""
    chain: str = "A"
    n_residues: int = 0
    uniprot_id: str = ""
    source: str = ""       # "api", "benchmark", "user", "fallback"

    # Core categories: each is a set of 0-based residue indices
    active_site: Set[int] = field(default_factory=set)
    binding: Set[int] = field(default_factory=set)
    metal: Set[int] = field(default_factory=set)
    mutagenesis: Set[int] = field(default_factory=set)
    regulatory: Set[int] = field(default_factory=set)
    domain_boundary: Set[int] = field(default_factory=set)
    disulfide: Set[int] = field(default_factory=set)

    # Domain annotations (for domain interaction analysis)
    domain_ranges: List[Dict[str, Any]] = field(default_factory=list)

    # Composite categories
    @property
    def all_functional(self) -> Set[int]:
        """Union of all annotated categories."""
        return (self.active_site | self.binding | self.metal |
                self.mutagenesis | self.regulatory |
                self.domain_boundary | self.disulfide)

    @property
    def mechanical(self) -> Set[int]:
        """Mechanical sites: domain boundaries / hinges.

        D96 showed locks concentrate here at 3.99× enrichment.
        These residues control conformational MOTION.
        """
        return self.domain_boundary

    @property
    def chemical(self) -> Set[int]:
        """Chemical function sites: active + binding + metal.

        D96 showed 1.50× enrichment — present but weaker than
        mechanical. These residues control CHEMISTRY.
        """
        return self.active_site | self.binding | self.metal

    @property
    def signalling(self) -> Set[int]:
        """Signalling sites: regulatory + mutagenesis.

        Sites involved in allosteric communication and evolutionary
        pressure. D96 showed allosteric sites at 4.49× enrichment.
        """
        return self.regulatory | self.mutagenesis

    @property
    def coverage(self) -> float:
        """Fraction of residues with any functional annotation."""
        if self.n_residues == 0:
            return 0.0
        return len(self.all_functional) / self.n_residues

    @property
    def n_categories_populated(self) -> int:
        """How many categories have at least one residue."""
        cats = [self.active_site, self.binding, self.metal,
                self.mutagenesis, self.regulatory,
                self.domain_boundary, self.disulfide]
        return sum(1 for c in cats if len(c) > 0)

    def as_d96_format(self) -> Dict[str, List[int]]:
        """Return in D96-compatible format {category: sorted list}.

        Maps our richer categories back to the 4 original D96 categories
        for backwards compatibility.
        """
        return {
            "active_site": sorted(self.active_site | self.binding | self.metal),
            "hinge": sorted(self.domain_boundary),
            "allosteric": sorted(self.regulatory),
            "mutation": sorted(self.mutagenesis),
        }

    def as_dict(self) -> Dict[str, Any]:
        """Serialisable dictionary representation."""
        return {
            "pdb_id": self.pdb_id,
            "chain": self.chain,
            "n_residues": self.n_residues,
            "uniprot_id": self.uniprot_id,
            "source": self.source,
            "coverage": round(self.coverage, 4),
            "n_categories": self.n_categories_populated,
            "active_site": sorted(self.active_site),
            "binding": sorted(self.binding),
            "metal": sorted(self.metal),
            "mutagenesis": sorted(self.mutagenesis),
            "regulatory": sorted(self.regulatory),
            "domain_boundary": sorted(self.domain_boundary),
            "disulfide": sorted(self.disulfide),
            "all_functional": sorted(self.all_functional),
            "mechanical": sorted(self.mechanical),
            "chemical": sorted(self.chemical),
            "signalling": sorted(self.signalling),
            "domain_ranges": self.domain_ranges,
        }


class FunctionalSiteResolver:
    """Resolve functional annotations for protein structures.

    Tries three strategies in order:
    1. User-provided `fallback_sites` dict (if given)
    2. UniProt REST API + PDBe SIFTS (requires network)
    3. Built-in benchmark database (for our 8 well-known proteins)

    Parameters
    ----------
    cache_dir : Path, optional
        Directory to cache API responses (avoids repeated fetches).
    hinge_margin : int
        Number of residues on either side of a domain boundary to
        include as "domain_boundary" (default 3).
    """

    def __init__(self, cache_dir: Optional[Path] = None,
                 hinge_margin: int = 3):
        self._cache: Dict[str, FunctionalAnnotation] = {}
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._hinge_margin = hinge_margin
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def resolve(self, pdb_id: str, chain: str = 'A',
                n_residues: Optional[int] = None,
                fallback_sites: Optional[Dict[str, List[int]]] = None,
                ) -> FunctionalAnnotation:
        """Resolve functional annotations for a PDB chain.

        Parameters
        ----------
        pdb_id : str
            4-letter PDB identifier.
        chain : str
            Chain identifier (default 'A').
        n_residues : int, optional
            Number of Cα atoms — used for bounds checking.
        fallback_sites : dict, optional
            User-provided {category: [residue_indices]} to use if
            API resolution fails. Categories can be any of our
            standard names or D96-format names.

        Returns
        -------
        FunctionalAnnotation
        """
        key = f"{pdb_id.upper()}_{chain}"
        if key in self._cache:
            cached = self._cache[key]
            if n_residues and cached.n_residues != n_residues:
                cached.n_residues = n_residues
            return cached

        # Check disk cache
        if self._cache_dir:
            disk_path = self._cache_dir / f"{key}.json"
            if disk_path.exists():
                ann = self._load_from_disk(disk_path, n_residues)
                if ann is not None:
                    self._cache[key] = ann
                    return ann

        ann = FunctionalAnnotation(pdb_id=pdb_id.upper(), chain=chain)
        if n_residues:
            ann.n_residues = n_residues

        # Strategy 1: User-provided fallback
        if fallback_sites:
            self._apply_fallback(ann, fallback_sites)
            ann.source = "user"
            self._cache[key] = ann
            self._save_to_disk(ann)
            return ann

        # Strategy 2+3: API resolution MERGED with benchmark database.
        # UniProt API provides standardised features (Active site, Binding
        # site, Metal binding, Mutagenesis, Domain). The benchmark DB adds
        # curated domain knowledge (hinges, allosteric networks, mutation
        # hotspots) that UniProt doesn't capture in structured features.
        # Merging both gives the richest annotation.
        api_ok = False
        try:
            self._resolve_from_api(ann)
            api_ok = True
        except Exception as e:
            warnings.warn(f"API resolution failed for {pdb_id}/{chain}: {e}")

        # Always merge benchmark data if available (it includes curated
        # domain knowledge that the API doesn't capture)
        if pdb_id.upper() in _BENCHMARK_DB:
            self._apply_benchmark(ann, pdb_id.upper())
            ann.source = "api+benchmark" if api_ok else "benchmark"
        elif api_ok:
            ann.source = "api"
        else:
            ann.source = "none"

        self._cache[key] = ann
        self._save_to_disk(ann)
        return ann

    # ------------------------------------------------------------------
    # Strategy 2: UniProt + PDBe SIFTS API
    # ------------------------------------------------------------------
    def _resolve_from_api(self, ann: FunctionalAnnotation):
        """Fetch annotations from UniProt REST + PDBe SIFTS."""
        import requests

        # Step 1: Parse mmCIF to build label_seq_id → 0-based index
        res_mapping = self._get_residue_mapping(ann.pdb_id, ann.chain)
        if ann.n_residues == 0:
            ann.n_residues = len(res_mapping)

        # Step 2: Get UniProt ID + SIFTS residue mapping
        uniprot_id, sifts_ranges = self._get_sifts(ann.pdb_id, ann.chain)
        ann.uniprot_id = uniprot_id

        # Step 3: Build UniProt position → 0-based index
        unp_to_idx = self._build_unp_to_idx(sifts_ranges, res_mapping)

        # Step 4: Fetch UniProt features
        features = self._fetch_uniprot(uniprot_id)

        # Step 5: Categorise features into annotation
        self._categorise_features(ann, features, unp_to_idx)

    def _get_residue_mapping(self, pdb_id: str, chain: str) -> Dict[int, int]:
        """Parse mmCIF to build {label_seq_id: 0-based Cα index}.

        This matches the order in which fetch_ca() collects atoms,
        ensuring our annotations align with the coordinate array.
        """
        import requests
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        mapping = {}
        idx = 0
        seen = set()
        for line in resp.text.split('\n'):
            if not line.startswith('ATOM'):
                continue
            parts = line.split()
            if len(parts) < 15:
                continue
            atom_name = parts[3]
            chain_id = parts[6]
            res_seq = parts[8]
            alt_id = parts[4] if len(parts) > 4 else '.'

            if atom_name != 'CA' or chain_id != chain:
                continue
            if alt_id not in ('.', '?', 'A', ''):
                continue
            key = (chain_id, res_seq)
            if key in seen:
                continue
            seen.add(key)

            try:
                seq_id = int(res_seq)
                mapping[seq_id] = idx
                idx += 1
            except ValueError:
                continue

        return mapping

    def _get_sifts(self, pdb_id: str, chain: str):
        """Get UniProt ID and residue range mapping from PDBe SIFTS."""
        import requests
        url = (f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/"
               f"{pdb_id.lower()}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        pdb_data = data.get(pdb_id.lower(), {}).get("UniProt", {})

        # Try chain_id match first, then struct_asym_id
        for uniprot_id, info in pdb_data.items():
            for m in info.get("mappings", []):
                if m.get("chain_id") == chain:
                    return uniprot_id, info.get("mappings", [])

        for uniprot_id, info in pdb_data.items():
            for m in info.get("mappings", []):
                if m.get("struct_asym_id") == chain:
                    return uniprot_id, info.get("mappings", [])

        # Last resort: return first mapping
        for uniprot_id, info in pdb_data.items():
            mappings = info.get("mappings", [])
            if mappings:
                return uniprot_id, mappings

        raise ValueError(f"No UniProt mapping for {pdb_id}/{chain}")

    @staticmethod
    def _build_unp_to_idx(sifts_ranges, res_mapping):
        """Build {UniProt_position: 0-based_index} mapping."""
        mapping = {}
        for sifts in sifts_ranges:
            pdb_start = sifts["start"]["residue_number"]
            pdb_end = sifts["end"]["residue_number"]
            unp_start = sifts["unp_start"]
            unp_end = sifts["unp_end"]

            n_res = min(unp_end - unp_start, pdb_end - pdb_start) + 1
            for offset in range(n_res):
                unp_pos = unp_start + offset
                pdb_pos = pdb_start + offset
                if pdb_pos in res_mapping:
                    mapping[unp_pos] = res_mapping[pdb_pos]

        return mapping

    @staticmethod
    def _fetch_uniprot(uniprot_id: str) -> list:
        """Fetch feature annotations from UniProt REST API."""
        import requests
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        headers = {"Accept": "application/json"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("features", [])

    def _categorise_features(self, ann: FunctionalAnnotation,
                             features: list,
                             unp_to_idx: Dict[int, int]):
        """Map UniProt features to our annotation categories."""
        domain_ranges = []
        N = ann.n_residues

        # Keywords that suggest regulatory/allosteric function
        REGULATORY_KEYWORDS = {
            "allosteric", "regulatory", "effector", "activation",
            "inhibitor", "modulator", "signal", "switch",
        }

        for feat in features:
            feat_type = feat.get("type", "")
            loc = feat.get("location", {})
            start_val = loc.get("start", {}).get("value")
            end_val = loc.get("end", {}).get("value")
            desc = (feat.get("description", "") or "").lower()

            if start_val is None or end_val is None:
                continue

            # Convert UniProt positions to 0-based indices
            indices = set()
            for pos in range(int(start_val), int(end_val) + 1):
                if pos in unp_to_idx:
                    idx = unp_to_idx[pos]
                    if N == 0 or idx < N:
                        indices.add(idx)

            if not indices:
                continue

            # Categorise by UniProt feature type
            if feat_type == "Active site":
                ann.active_site.update(indices)
            elif feat_type == "Binding site":
                ann.binding.update(indices)
            elif feat_type == "Metal binding":
                ann.metal.update(indices)
            elif feat_type == "Mutagenesis":
                ann.mutagenesis.update(indices)
            elif feat_type == "Disulfide bond":
                ann.disulfide.update(indices)
            elif feat_type == "Domain":
                domain_ranges.append({
                    "name": feat.get("description", ""),
                    "start": min(indices),
                    "end": max(indices),
                    "indices": sorted(indices),
                })
            elif feat_type in ("Region", "Site"):
                if any(kw in desc for kw in REGULATORY_KEYWORDS):
                    ann.regulatory.update(indices)
                elif "active" in desc or "catalytic" in desc:
                    ann.active_site.update(indices)
                elif "binding" in desc:
                    ann.binding.update(indices)
            elif feat_type == "Nucleotide binding":
                ann.binding.update(indices)

        # Infer domain boundaries (hinges) from domain annotations
        ann.domain_ranges = domain_ranges
        if len(domain_ranges) >= 2:
            domain_ranges_sorted = sorted(domain_ranges, key=lambda d: d["start"])
            for i in range(1, len(domain_ranges_sorted)):
                prev_end = domain_ranges_sorted[i - 1]["end"]
                curr_start = domain_ranges_sorted[i]["start"]
                margin = self._hinge_margin
                for r in range(max(0, prev_end - margin),
                               min(N if N > 0 else 10000,
                                   curr_start + margin + 1)):
                    ann.domain_boundary.add(r)

    # ------------------------------------------------------------------
    # Strategy 1: User-provided fallback
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_fallback(ann: FunctionalAnnotation,
                        sites: Dict[str, List[int]]):
        """Apply user-provided site dict to annotation.

        Accepts both our standard category names and D96-format names.
        """
        N = ann.n_residues

        # Standard category mapping
        mapping = {
            "active_site": "active_site",
            "binding": "binding",
            "binding_site": "binding",
            "metal": "metal",
            "metal_binding": "metal",
            "mutagenesis": "mutagenesis",
            "mutation": "mutagenesis",    # D96 name
            "regulatory": "regulatory",
            "allosteric": "regulatory",   # D96 name
            "domain_boundary": "domain_boundary",
            "hinge": "domain_boundary",   # D96 name
            "disulfide": "disulfide",
        }

        for key, residues in sites.items():
            target_attr = mapping.get(key.lower())
            if target_attr is None:
                continue
            target_set = getattr(ann, target_attr)
            for r in residues:
                if N == 0 or r < N:
                    target_set.add(int(r))

    # ------------------------------------------------------------------
    # Strategy 3: Built-in benchmark database
    # ------------------------------------------------------------------
    def _apply_benchmark(self, ann: FunctionalAnnotation, pdb_id: str):
        """Apply built-in benchmark data for known proteins."""
        data = _BENCHMARK_DB.get(pdb_id, {})
        self._apply_fallback(ann, data.get("sites", {}))
        ann.uniprot_id = data.get("uniprot_id", "")

    # ------------------------------------------------------------------
    # Disk caching
    # ------------------------------------------------------------------
    def _save_to_disk(self, ann: FunctionalAnnotation):
        """Save annotation to disk cache."""
        if self._cache_dir is None:
            return
        key = f"{ann.pdb_id}_{ann.chain}"
        path = self._cache_dir / f"{key}.json"
        try:
            with open(path, 'w') as f:
                json.dump(ann.as_dict(), f, indent=2)
        except Exception:
            pass  # Non-critical; just skip caching

    @staticmethod
    def _load_from_disk(path: Path,
                        n_residues: Optional[int]) -> Optional[FunctionalAnnotation]:
        """Load annotation from disk cache."""
        try:
            with open(path) as f:
                data = json.load(f)
            ann = FunctionalAnnotation(
                pdb_id=data.get("pdb_id", ""),
                chain=data.get("chain", "A"),
                n_residues=n_residues or data.get("n_residues", 0),
                uniprot_id=data.get("uniprot_id", ""),
                source=data.get("source", "cache"),
            )
            for cat in ["active_site", "binding", "metal", "mutagenesis",
                        "regulatory", "domain_boundary", "disulfide"]:
                setattr(ann, cat, set(data.get(cat, [])))
            ann.domain_ranges = data.get("domain_ranges", [])
            return ann
        except Exception:
            return None


# ======================================================================
# Built-in benchmark database
# D96-validated functional sites for our 8 well-characterised proteins.
# All residue numbers are 0-based Cα indices.
# ======================================================================

_BENCHMARK_DB = {
    "2LAO": {
        "uniprot_id": "P02911",
        "sites": {
            "active_site": (list(range(10, 16)) + [69, 70, 76, 77,
                            114, 115, 116, 117]),
            "hinge": list(range(85, 93)) + list(range(186, 195)),
            "allosteric": [14, 15, 69, 70, 76, 77, 114, 115, 116, 117],
            "mutation": [11, 14, 69, 70, 77, 115, 116],
        },
    },
    "4AKE": {
        "uniprot_id": "P69441",
        "sites": {
            "active_site": (list(range(7, 14)) +
                            [130, 131, 132, 133, 134, 135, 136]),
            "hinge": [29, 30, 31, 66, 67, 68, 117, 118, 119,
                      155, 156, 157, 158, 159, 160],
            "allosteric": list(range(30, 40)) + list(range(118, 135)),
            "mutation": [7, 8, 9, 10, 11, 12, 13, 130, 131, 132, 133],
        },
    },
    "1CLL": {
        "uniprot_id": "P0DP23",
        "sites": {
            "active_site": (list(range(20, 32)) + list(range(56, 68)) +
                            list(range(93, 105)) + list(range(129, 141))),
            "hinge": list(range(73, 85)),
            "allosteric": [17, 18, 35, 36, 51, 52, 91, 92, 108, 109,
                           124, 125, 127, 128],
            "mutation": [20, 24, 28, 56, 60, 64, 93, 97, 101,
                         129, 133, 137],
        },
    },
    "5CSC": {
        "uniprot_id": "P00480",
        "sites": {
            "active_site": [274, 275, 314, 315, 316, 317, 318, 319, 320,
                            327, 328, 329, 338, 339, 375, 376, 377, 378,
                            379, 380, 381],
            "hinge": (list(range(46, 56)) + list(range(265, 275)) +
                      list(range(414, 424))),
            "allosteric": [46, 47, 48, 49, 50, 51, 414, 415, 416, 417,
                           418, 419, 420],
            "mutation": [274, 314, 316, 327, 338, 375, 376, 380],
        },
    },
    "2LZM": {
        "uniprot_id": "P00720",
        "sites": {
            "active_site": [10, 11, 19, 20, 25, 26, 30, 31, 32, 34,
                            107, 108, 113, 114, 115],
            "hinge": list(range(60, 82)),
            "allosteric": [60, 61, 62, 65, 66, 67, 68, 69, 70, 71,
                           72, 73, 99, 100, 101, 102, 103],
            "mutation": [3, 10, 19, 25, 44, 46, 59, 60, 67, 68, 86,
                         93, 96, 99, 118, 133, 152, 153, 157],
        },
    },
    "1RX2": {
        "uniprot_id": "P0ABQ4",
        "sites": {
            "active_site": [4, 5, 6, 7, 20, 21, 22, 26, 27, 28, 30, 31,
                            44, 45, 46, 47, 48, 62, 93, 94, 112, 113],
            "hinge": list(range(9, 24)) + list(range(116, 125)),
            "allosteric": [15, 16, 17, 18, 44, 45, 46, 47, 48, 77, 78,
                           93, 94, 101, 102, 122, 123, 124],
            "mutation": [5, 26, 27, 28, 30, 31, 78, 93, 94, 112, 113,
                         151, 152, 153],
        },
    },
    "1HHP": {
        "uniprot_id": "P03366",
        "sites": {
            "active_site": [23, 24, 25, 26, 27, 28, 29],
            "hinge": list(range(43, 59)),
            "allosteric": list(range(10, 24)) + list(range(78, 88)),
            "mutation": [10, 20, 23, 24, 30, 31, 32, 33, 36, 46, 47,
                         48, 49, 50, 53, 54, 71, 73, 76, 77, 81, 82,
                         83, 84, 88, 89, 90],
        },
    },
    "2HHB": {
        "uniprot_id": "P69905",
        "sites": {
            "active_site": [57, 58, 62, 63, 86, 87, 91, 92],
            "hinge": (list(range(37, 45)) + list(range(93, 100)) +
                      list(range(135, 141))),
            "allosteric": [34, 35, 36, 37, 38, 39, 40, 41, 92, 93, 94,
                           95, 96, 97, 98, 99, 126, 127, 130, 131, 136,
                           137, 138, 139, 140],
            "mutation": [6, 26, 47, 57, 58, 62, 63, 68, 86, 87, 91, 92,
                         126, 127],
        },
    },
}
