"""PDB data fetching utilities.

Fetch Cα coordinates, B-factors, and metadata from RCSB PDB via mmCIF.
Search RCSB for candidate structures.
"""

import numpy as np
import requests
from typing import Optional, Tuple, List, Dict


def fetch_pdb_ca_data(pdb_id: str, chain: Optional[str] = None,
                      min_res: int = 50, max_res: int = 300
                      ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                 Optional[int], Optional[float]]:
    """Fetch Cα coordinates and B-factors from RCSB.

    Parameters
    ----------
    pdb_id : str
        4-character PDB identifier.
    chain : str, optional
        Restrict to a specific chain.  If None, uses the first chain that
        passes quality filters.
    min_res, max_res : int
        Residue count bounds.

    Returns
    -------
    coords : (N, 3) ndarray or None
    b_factors : (N,) ndarray or None
    n_chains : int or None
    resolution : float or None
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None, None, None, None
    except Exception:
        return None, None, None, None

    lines = resp.text.split('\n')

    # ── Parse mmCIF ATOM records ──
    ca_atoms: List[Dict] = []
    in_atom_site = False
    col_names: List[str] = []

    for line in lines:
        if line.startswith('_atom_site.'):
            in_atom_site = True
            col_names.append(line.strip().split('.')[1])
            continue
        if in_atom_site and (line.startswith('_') or line.startswith('#')
                            or line.startswith('loop_')):
            if not line.startswith('_atom_site.'):
                in_atom_site = False
                if line.startswith('loop_'):
                    col_names = []
                continue
        if in_atom_site and line.startswith(('ATOM', 'HETATM')):
            parts = line.split()
            if len(parts) >= len(col_names):
                record = dict(zip(col_names, parts))
                if (record.get('label_atom_id') == 'CA'
                        and record.get('group_PDB') == 'ATOM'
                        and record.get('label_alt_id', '.') in ('.', 'A')):
                    try:
                        ca_atoms.append({
                            'chain': record.get('label_asym_id', 'A'),
                            'x': float(record['Cartn_x']),
                            'y': float(record['Cartn_y']),
                            'z': float(record['Cartn_z']),
                            'bfactor': float(record.get('B_iso_or_equiv', '0')),
                            'occupancy': float(record.get('occupancy', '1.0')),
                            'resnum': int(record.get('label_seq_id', '0')),
                        })
                    except (ValueError, KeyError):
                        pass

    if not ca_atoms:
        return None, None, None, None

    # ── Chain selection ──
    chains = sorted(set(a['chain'] for a in ca_atoms))
    if chain is not None:
        ca_atoms = [a for a in ca_atoms if a['chain'] == chain]
    elif len(chains) != 1:
        ca_atoms = [a for a in ca_atoms if a['chain'] == chains[0]]

    # ── Size filter ──
    if len(ca_atoms) < min_res or len(ca_atoms) > max_res:
        return None, None, None, None

    # ── Occupancy filter ──
    ca_atoms = [a for a in ca_atoms if a['occupancy'] > 0.99]
    if len(ca_atoms) < min_res:
        return None, None, None, None

    coords = np.array([[a['x'], a['y'], a['z']] for a in ca_atoms])
    bfactors = np.array([a['bfactor'] for a in ca_atoms])

    # ── B-factor quality ──
    if bfactors.std() < 0.5:
        return None, None, None, None
    if bfactors.min() < 0:
        return None, None, None, None

    # ── Parse resolution ──
    resolution = _parse_resolution(lines)

    return coords, bfactors, len(chains), resolution


def _parse_resolution(lines: List[str]) -> Optional[float]:
    """Extract resolution from mmCIF text."""
    for line in lines:
        if 'resolution' in line.lower() and 'high' in line.lower():
            for p in line.split():
                try:
                    v = float(p)
                    if 0.5 < v < 5.0:
                        return v
                except ValueError:
                    pass
    return None


def search_rcsb(max_results: int = 200,
                max_resolution: float = 1.8) -> List[str]:
    """Search RCSB for high-resolution X-ray structures.

    Returns a list of PDB IDs suitable for benchmarking.
    """
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text",
                 "parameters": {
                     "attribute": "exptl.method",
                     "operator": "exact_match",
                     "value": "X-RAY DIFFRACTION"}},
                {"type": "terminal", "service": "text",
                 "parameters": {
                     "attribute": "rcsb_entry_info.resolution_combined",
                     "operator": "less_or_equal",
                     "value": max_resolution}},
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_results},
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined",
                       "direction": "asc"}]
        }
    }
    try:
        resp = requests.post(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            json=query, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return [hit['identifier'] for hit in data.get('result_set', [])]
    except Exception as e:
        print(f"RCSB search failed: {e}")
        return []


def fetch_pdb_annotations(pdb_id: str, chain: str = "A") -> Dict:
    """Fetch structural annotations from RCSB REST APIs.

    Uses two endpoints:
      1. GraphQL for entity-level data (UniProt IDs)
      2. REST for instance-level features (CATH, SCOP, ECOD, secondary structure)

    Returns dict with keys:
      'uniprot_ids': list of UniProt accessions
      'instance_features': list of feature dicts (type, name, positions)
      'entity_annotations': list of annotation dicts
    """
    result: Dict = {
        'uniprot_ids': [],
        'instance_features': [],
        'entity_annotations': [],
    }

    # ── Entity-level: UniProt IDs via GraphQL ──
    gql_query = """
    query ($id: String!) {
      entry(entry_id: $id) {
        polymer_entities {
          rcsb_polymer_entity_container_identifiers {
            uniprot_ids
            auth_asym_ids
          }
        }
      }
    }
    """
    try:
        resp = requests.post(
            "https://data.rcsb.org/graphql",
            json={"query": gql_query, "variables": {"id": pdb_id}},
            timeout=15)
        resp.raise_for_status()
        entry = resp.json().get("data", {}).get("entry", {})
        for ent in (entry.get("polymer_entities") or []):
            ids = ent.get("rcsb_polymer_entity_container_identifiers", {})
            for uid in (ids.get("uniprot_ids") or []):
                result['uniprot_ids'].append(uid)
    except Exception:
        pass

    # ── Instance-level: CATH/SCOP/ECOD features via REST ──
    try:
        rest_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity_instance/{pdb_id}/{chain}"
        resp = requests.get(rest_url, timeout=15)
        if resp.ok:
            data = resp.json()
            for feat in (data.get("rcsb_polymer_instance_feature") or []):
                result['instance_features'].append({
                    'type': feat.get('type', ''),
                    'name': feat.get('name', ''),
                    'feature_positions': feat.get('feature_positions', []),
                })
    except Exception:
        pass

    return result


def fetch_uniprot_features(uniprot_id: str) -> List[Dict]:
    """Fetch feature annotations from UniProt REST API.

    Returns list of feature dicts with keys: type, location.start, location.end,
    description, etc.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("features", [])
    except Exception:
        return []
