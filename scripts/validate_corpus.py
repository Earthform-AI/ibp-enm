#!/usr/bin/env python3
"""Validate that every PDB entry in a corpus is fetchable and has
enough Cα atoms for ENM analysis (≥50 residues).

Usage:
    python scripts/validate_corpus.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import requests
import numpy as np

def fetch_ca_count(pdb_id: str, chain: str) -> int:
    """Return number of Cα atoms for pdb_id:chain, or -1 on error."""
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return -1
        lines = resp.text.split("\n")
    except Exception:
        return -1

    def _extract(lines, chain_id, record_types, chain_col):
        coords = []
        seen = set()
        for line in lines:
            if not any(line.startswith(rt) for rt in record_types):
                continue
            parts = line.split()
            if len(parts) < 15:
                continue
            atom_name = parts[3]
            cid = parts[chain_col] if len(parts) > chain_col else ""
            res_seq = parts[8]
            alt_id = parts[4] if len(parts) > 4 else "."
            if atom_name != "CA" or cid != chain_id:
                continue
            if alt_id not in (".", "?", "A", ""):
                continue
            dedup_key = (cid, res_seq) if res_seq not in (".", "?") else parts[1]
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            coords.append(1)
        return len(coords)

    # Try 4 strategies
    for rt, col in [
        (("ATOM",), 6), (("ATOM",), 18),
        (("ATOM", "HETATM"), 6), (("ATOM", "HETATM"), 18),
    ]:
        n = _extract(lines, chain, rt, col)
        if n >= 20:
            return n
    return n  # last attempt count


# ── Proposed corpus entries ──────────────────────────────────────
# Format: (name, pdb_id, chain, archetype)

PROPOSED_NEW = [
    # ── enzyme_active (27 new) ──
    ("Barnase",             "1BNI", "A", "enzyme_active"),
    ("Staph_nuclease",      "1STN", "A", "enzyme_active"),
    ("Phospholipase_A2",    "1BP2", "A", "enzyme_active"),
    ("Carboxypeptidase_A",  "5CPA", "A", "enzyme_active"),
    ("Pepsin",              "4PEP", "A", "enzyme_active"),
    ("Penicillopepsin",     "3APP", "A", "enzyme_active"),
    ("Actinidin",           "2ACT", "A", "enzyme_active"),
    ("Beta_lactamase_TEM",  "1BTL", "A", "enzyme_active"),
    ("Cyclophilin_A",       "2CPL", "A", "enzyme_active"),
    ("Thioredoxin",         "2TRX", "A", "enzyme_active"),
    ("Cutinase",            "1CEX", "A", "enzyme_active"),
    ("Cu_Zn_SOD",           "2SOD", "A", "enzyme_active"),
    ("Lysozyme_human",      "1LZ1", "A", "enzyme_active"),
    ("Xylanase_Bcirculans", "1XNB", "A", "enzyme_active"),
    ("Ribonuclease_Sa",     "1RGG", "A", "enzyme_active"),
    ("CytC_peroxidase",     "2CYP", "A", "enzyme_active"),
    ("Alpha_lytic_protease","2ALP", "A", "enzyme_active"),
    ("Proteinase_K",        "2PKC", "A", "enzyme_active"),
    ("Glutathione_Strans",  "2GST", "A", "enzyme_active"),
    ("Galactose_oxidase",   "1GOF", "A", "enzyme_active"),
    ("Carbonic_anhydrase_I","2CAB", "A", "enzyme_active"),
    ("Trypsinogen",         "1TGN", "A", "enzyme_active"),
    ("Phosphoglycerate_mut","3PGM", "A", "enzyme_active"),
    ("Endoglucanase",       "1EGZ", "A", "enzyme_active"),
    ("Savinase",            "1SVN", "A", "enzyme_active"),
    ("Lipase_Rhizomucor",   "4TGL", "A", "enzyme_active"),
    ("Dihydroorotase",      "1J79", "A", "enzyme_active"),

    # ── barrel (30 new) ──
    ("HisF",                "1THF", "A", "barrel"),
    ("NAL_lyase",           "1NAL", "A", "barrel"),
    ("Indole3GP_synthase",  "1IGS", "A", "barrel"),
    ("PRAI",                "1PII", "A", "barrel"),
    ("Old_yellow_enzyme",   "1OYA", "A", "barrel"),
    ("PI_PLC",              "1PTG", "A", "barrel"),
    ("Orotidine_decarb",    "1DQX", "A", "barrel"),
    ("Alpha_amylase",       "1SMD", "A", "barrel"),
    ("Deoxyribose_P_ald",   "1JCJ", "A", "barrel"),
    ("GFP",                 "1GFL", "A", "barrel"),
    ("Retinol_BP",          "1RBP", "A", "barrel"),
    ("Avidin",              "1AVD", "A", "barrel"),
    ("FABP_intestinal",     "1IFB", "A", "barrel"),
    ("Concanavalin_A",      "3CNA", "A", "barrel"),
    ("Beta_glucosidase",    "1BGA", "A", "barrel"),
    ("Alanine_racemase",    "1BD0", "A", "barrel"),
    ("Muconate_lactonize",  "1MUC", "A", "barrel"),
    ("Thiamin_P_synthase",  "2TPS", "A", "barrel"),
    ("Methylmalonyl_mutase","1REQ", "A", "barrel"),
    ("Glycerophosphodiesterase","1YMQ","A","barrel"),
    ("Cellulase_Cel5A",     "1CEN", "A", "barrel"),
    ("Dihydropteroate_syn", "1AJ0", "A", "barrel"),
    ("IGPS",                "1A53", "A", "barrel"),
    ("Phosphotriesterase",  "1HZY", "A", "barrel"),
    ("Transaldolase",       "1ONR", "A", "barrel"),
    ("Flavocytochrome_b2",  "1FCB", "A", "barrel"),
    ("Luciferase_bacterial","1LUC", "A", "barrel"),
    ("Chitinase_A1",        "1CTN", "A", "barrel"),
    ("Neopullulanase",      "1J0H", "A", "barrel"),
    ("Pyruvate_oxidase",    "1POW", "A", "barrel"),

    # ── globin (30 new) ──
    ("Hemoglobin_lamprey",  "2LHB", "A", "globin"),
    ("Cytochrome_b562",     "256B", "A", "globin"),
    ("Cytochrome_c2",       "1C2R", "A", "globin"),
    ("Cytochrome_c551",     "351C", "A", "globin"),
    ("Myohemerythrin",       "2MHR", "A", "globin"),
    ("Cytoglobin",           "1V5H", "A", "globin"),
    ("Hemoglobin_Ascaris",   "1ASH", "A", "globin"),
    ("Phycocyanin",          "1CPC", "A", "globin"),
    ("Ferritin",             "1FHA", "A", "globin"),
    ("Bacterioferritin",     "1BFR", "A", "globin"),
    ("Hemoglobin_Chironomus","1ECD", "A", "globin"),
    ("Hemoglobin_sickle",    "2HBS", "A", "globin"),
    ("Hemoglobin_Scapharca", "4SDH", "A", "globin"),
    ("Interleukin_4",        "1HIK", "A", "globin"),
    ("Growth_hormone",       "1HGU", "A", "globin"),
    ("GM_CSF",               "2GMF", "A", "globin"),
    ("EPO",                  "1BUY", "A", "globin"),
    ("Interferon_beta",      "1AU1", "A", "globin"),
    ("ROP_protein",          "1ROP", "A", "globin"),
    ("Cyt_c3",               "2CDV", "A", "globin"),
    ("Hemerythrin",           "1HMD", "A", "globin"),
    ("Myoglobin_horse",       "1WLA", "A", "globin"),
    ("Hemoglobin_fetal_G",    "1FDH", "A", "globin"),
    ("Cytochrome_c6",         "1CYJ", "A", "globin"),
    ("Leghemoglobin_lupin",   "1GDI", "A", "globin"),
    ("Hemoglobin_sea_lamprey","1HBG", "A", "globin"),
    ("Cyt_c_oxidase_sub2",    "2OCC", "A", "globin"),
    ("Flavodoxin",             "1FLV", "A", "globin"),
    ("Ferredoxin",             "1FDX", "A", "globin"),
    ("Apomyoglobin",           "1U7S", "A", "globin"),

    # ── dumbbell (30 new) ──
    ("Galactose_BP",           "2GBP", "A", "dumbbell"),
    ("Ribose_BP",              "2DRI", "A", "dumbbell"),
    ("Sulfate_BP",             "1SBP", "A", "dumbbell"),
    ("Maltodextrin_BP",        "1ANF", "A", "dumbbell"),
    ("Dipeptide_BP",           "1DPP", "A", "dumbbell"),
    ("Histidine_BP",           "1HSL", "A", "dumbbell"),
    ("Ferric_BP",              "1MRP", "A", "dumbbell"),
    ("Ovotransferrin",         "1OVT", "A", "dumbbell"),
    ("Hexokinase_yeast",       "2YHX", "A", "dumbbell"),
    ("Actin",                  "1J6Z", "A", "dumbbell"),
    ("EF_Tu",                  "1TTT", "A", "dumbbell"),
    ("Serum_albumin",          "1AO6", "A", "dumbbell"),
    ("Glutathione_reductase",  "3GRS", "A", "dumbbell"),
    ("GluR2_LBD",              "1FTJ", "A", "dumbbell"),
    ("Aspartate_aminotrans",   "7AAT", "A", "dumbbell"),
    ("DNA_pol_beta",           "1BPY", "A", "dumbbell"),
    ("Leucine_BP",             "1USG", "A", "dumbbell"),
    ("Phosphate_BP",           "1A54", "A", "dumbbell"),
    ("Hsp90_ATPase",           "1YES", "A", "dumbbell"),
    ("Aminoacyl_tRNA_synth",   "1EUY", "A", "dumbbell"),
    ("Protein_disulfide_iso",  "1MEK", "A", "dumbbell"),
    ("Catalase_HPII",          "1IPH", "A", "dumbbell"),
    ("Aconitase",              "7ACN", "A", "dumbbell"),
    ("Glutamine_synthetase",   "2GLS", "A", "dumbbell"),
    ("Phosphoenolpyr_carboxyk","1KHB", "A", "dumbbell"),
    ("Citrate_lyase",          "1K6W", "A", "dumbbell"),
    ("G6PD",                   "1QKI", "A", "dumbbell"),
    ("Aldehyde_dehydrogenase", "1BXS", "A", "dumbbell"),
    ("Alcohol_dehydrogenase",  "2OHX", "A", "dumbbell"),
    ("Dihydropteridine_red",   "1DHR", "A", "dumbbell"),

    # ── allosteric (31 new) ──
    ("Ras_p21",                "5P21", "A", "allosteric"),
    ("G_alpha_i",              "1GIA", "A", "allosteric"),
    ("Transducin_alpha",       "1TAG", "A", "allosteric"),
    ("Cdc42",                  "1AN0", "A", "allosteric"),
    ("Ran_GTPase",             "1BYU", "A", "allosteric"),
    ("NtrC_receiver",          "1NTR", "A", "allosteric"),
    ("FixJ_receiver",          "1D5W", "A", "allosteric"),
    ("FBPase",                 "1FBP", "A", "allosteric"),
    ("Isocitrate_DH",          "1AI2", "A", "allosteric"),
    ("ERK2",                   "2ERK", "A", "allosteric"),
    ("CDK2_cycA",              "1FIN", "A", "allosteric"),
    ("Src_kinase",             "2SRC", "A", "allosteric"),
    ("DnaK_ATPase",            "1DKG", "A", "allosteric"),
    ("CRP",                    "2CGP", "A", "allosteric"),
    ("Lac_repressor",          "1LBI", "A", "allosteric"),
    ("Spo0F",                  "1SRR", "A", "allosteric"),
    ("PhoB_receiver",          "1B00", "A", "allosteric"),
    ("Calcineurin_A",          "1AUI", "A", "allosteric"),
    ("Arf1",                   "1HUR", "A", "allosteric"),
    ("Hemoglobin_R",           "1HHO", "A", "allosteric"),
    ("Threonine_deaminase",    "1TDJ", "A", "allosteric"),
    ("PGDH",                   "1PSD", "A", "allosteric"),
    ("Aspartate_kinase",       "2J0W", "A", "allosteric"),
    ("GDH_bovine",             "1HWZ", "A", "allosteric"),
    ("IMPDH",                  "1JR1", "A", "allosteric"),
    ("Ribonuc_reductase_R1",   "1RLR", "A", "allosteric"),
    ("Pyruvate_dehydrog_E1",   "1W85", "A", "allosteric"),
    ("RNR_R2",                 "1RIB", "A", "allosteric"),
    ("Tryptophan_repressor",   "2OZ9", "A", "allosteric"),
    ("ArcA_receiver",          "1XHE", "A", "allosteric"),
    ("OmpR_receiver",          "1ODD", "A", "allosteric"),
]


def main():
    print(f"Validating {len(PROPOSED_NEW)} proposed PDB entries ...")
    print("=" * 70)

    ok, fail = [], []
    by_arch = {}
    for name, pdb_id, chain, arch in PROPOSED_NEW:
        if arch not in by_arch:
            by_arch[arch] = {"ok": 0, "fail": 0}

        n = fetch_ca_count(pdb_id, chain)
        status = "OK" if n >= 50 else "FAIL"
        marker = "✓" if n >= 50 else "✗"
        print(f"  {marker} {name:30s} {pdb_id}:{chain}  N={n:>4d}  [{arch}]")

        if n >= 50:
            ok.append((name, pdb_id, chain, arch, n))
            by_arch[arch]["ok"] += 1
        else:
            fail.append((name, pdb_id, chain, arch, n))
            by_arch[arch]["fail"] += 1

        time.sleep(0.15)  # rate limit

    print()
    print("=" * 70)
    print(f"PASSED: {len(ok)}/{len(PROPOSED_NEW)}")
    print(f"FAILED: {len(fail)}")
    print()

    for arch in sorted(by_arch):
        d = by_arch[arch]
        print(f"  {arch:16s}  ok={d['ok']:2d}  fail={d['fail']:2d}")

    if fail:
        print()
        print("FAILED entries (need replacement):")
        for name, pdb_id, chain, arch, n in fail:
            print(f"  {name} ({pdb_id}:{chain}) → N={n}")

    # Output valid entries as Python code
    print()
    print("=" * 70)
    print("VALID entries (paste into benchmark.py):")
    print("=" * 70)
    for arch in ["enzyme_active", "barrel", "globin", "dumbbell", "allosteric"]:
        entries = [(n, p, c, a, cnt) for n, p, c, a, cnt in ok if a == arch]
        if entries:
            print(f"\n# ── {arch} ({len(entries)} new) ──")
            for name, pdb_id, chain, arch_, cnt in entries:
                print(f'    ProteinEntry("{name}", "{pdb_id}", "{chain}", "{arch_}"),  # N={cnt}')


if __name__ == "__main__":
    main()
