# IBP-ENM as a Service — Packaging Plan

**Date:** 2026-04-14  
**Status:** Draft for review  
**Goal:** Package IBP-ENM into revenue-generating consulting and hosted analysis offerings

---

## Service Tiers

### Tier 1: Consulting & Analysis-on-Demand

**What:** You personally analyze client proteins using IBP-ENM and deliver a structured report.

**Deliverable per protein:**
- Structural archetype classification (enzyme/barrel/allosteric/dumbbell/globin)
- Domain boundary map with residue ranges
- Hinge site identification with confidence scores
- 7-instrument reaction profile (symmetry breaking, resonance sensitivity, etc.)
- B-factor prediction vs experimental (3 perspectives + consensus)
- Allosteric pathway candidates (propagative instrument)
- Comparison to known archetype signatures

**Pricing:**
| Service | Price | Turnaround |
|---------|-------|-----------|
| Single protein analysis | $200–$500 | 1–2 business days |
| Protein comparison (2 conformations) | $500–$1,000 | 2–3 business days |
| Batch analysis (10+ proteins) | $100–$250 per protein | 1 week |
| Custom instrument development | $2,000–$5,000 per instrument | 2–4 weeks |
| Retained advisory (ongoing) | $3,000–$5,000/month | — |

**Target clients:**
- Biotech startups doing structure-based drug design
- Protein engineering companies
- Academic labs without computational structural biology expertise
- CROs (Contract Research Organizations) offering computational services

**Value proposition over alternatives:**
| IBP-ENM | MD Simulation | ML Methods | DynDom/HingeProt |
|---------|--------------|------------|-------------------|
| Single structure needed | Needs trajectory | Needs training data | Needs two structures |
| Minutes to run | Hours to days | Fast inference, slow training | Fast |
| Zero parameters (core) | Many parameters | Many hyperparameters | Some thresholds |
| Physically interpretable | Physically meaningful | Black box | Limited interpretation |
| 100% on benchmark | N/A | Varies | Varies |

### Tier 2: Hosted Analysis API (Future)

**What:** REST API where clients submit PDB IDs and receive structured JSON analysis.

**Implementation outline:**
```
POST /api/v1/analyze
  Body: { "pdb_id": "2LZM", "chain": "A" }
  Response: {
    "archetype": "enzyme_active",
    "confidence": 0.94,
    "domains": [{"residues": [1, 45], "label": 0}, ...],
    "hinges": [{"residue": 46, "score": 0.87}],
    "instruments": { "algebraic": {...}, "musical": {...}, ... },
    "b_factor_correlation": 0.82
  }
```

**Tech stack:**
- FastAPI or Flask (Python, minimal new code)
- IBP-ENM as the backend engine
- Rate limiting + API keys for authentication
- Deployed on Vercel (serverless) or a small VPS

**Pricing model:**
- Free tier: 5 analyses/month (lead generation)
- Researcher: $49/month — 50 analyses, JSON + basic visualization
- Lab: $199/month — 500 analyses, batch API, CSV export, priority
- Enterprise: $999/month — unlimited, SLA, custom instruments, direct support

**Build effort:** ~3–5 days for a working MVP (FastAPI wrapper around `run_single_protein()`).

**Note:** Build this AFTER you have 2–3 consulting clients. Don't build the API hoping clients will come — validate demand through manual consulting first, then automate the repeating requests.

---

## Packaging the Consulting Offer

### One-Page Pitch Document

Create a clean, professional PDF:

**Structure:**
1. **Header:** Earthform AI logo + "Protein Structural Analysis Services"
2. **The Problem (2 sentences):** Understanding protein structural mechanisms traditionally requires MD simulations (expensive, slow) or ML models (opaque, need training data).
3. **Our Approach (3 sentences):** IBP-ENM uses spectral decomposition of the elastic network to probe 7 independent physical signals from a single static structure. No training data. Zero tunable parameters in the core method.
4. **Key Results:**
   - 100% accuracy on 12-protein benchmark (HingeLens)
   - 57.7% on 52-protein expanded corpus (zero-parameter, outperforms 5-parameter baseline)
   - Published on PyPI: `pip install ibp-enm`
5. **What We Deliver:** (bullet list of deliverables from Tier 1 above)
6. **Pricing:** (simple table)
7. **Contact:** email + earthform.ai

### Landing Page Addition

Add a "Services" page to the Earthform Research Lab site (warden-landing):
- Route: `/services` or `/protein-analysis`
- Content pulled from `site.config.ts` (config-driven, easy to update)
- Contact form pointed at Formspree (already integrated)

---

## Client Acquisition Strategy

### Week 1–2: Warm Leads

1. **Post on BioStars** — "We offer protein structural analysis using a novel spectral method. AMA."
2. **Post on ResearchGate** — Share the preprint (once available), offer free analyses of interesting proteins
3. **Twitter/X scientific community** — Thread showing IBP-ENM analyzing a well-known protein with visual outputs
4. **Hacker News** — "Show HN: Zero-parameter protein classification from a single structure" — IBP-ENM is on PyPI

### Week 2–4: Cold Outreach

**Target companies (examples of types to search for):**
- Early-stage protein engineering/design startups
- Computational drug discovery companies pre-Series B
- Enzyme engineering companies

**Email template:**
> Subject: Structural analysis of [their target protein family] — no MD needed
>
> Hi [Name],
>
> I noticed [Company] is working on [specific area]. I built a tool (IBP-ENM, MIT-licensed on PyPI) that classifies protein structural archetypes and identifies domain boundaries, hinge sites, and allosteric pathways from a single PDB structure — no simulations, no training data.
>
> It achieves 100% accuracy on our benchmark corpus using a 7-instrument spectral decomposition method with zero tunable parameters.
>
> Would a quick analysis of [one of their proteins] be useful? I'm happy to run one for free so you can evaluate the output.
>
> Best,
> Josh Byrom
> Earthform AI

**The free analysis is the key move.** It costs you 15 minutes and creates a concrete reason for them to respond.

### Ongoing: Academic Collaborations

- Contact 3–5 structural biology PIs at universities
- Offer: "I'll run IBP-ENM on your protein set for free, you co-author the validation study"
- This gets you:
  - University affiliation (for grant eligibility)
  - Co-PI for STTR applications
  - Published validation (builds credibility for more clients)
  - Network into the structural biology community

---

## Revenue Projections for Service Model

### Conservative (2–3 clients)

| Month | Consulting | API Revenue | Total |
|-------|-----------|-------------|-------|
| 1 | $0 | $0 | $0 |
| 2 | $500 (1 pilot) | $0 | $500 |
| 3 | $2,000 | $0 | $2,000 |
| 4 | $3,000 | $0 | $3,000 |
| 5 | $4,000 | $200 | $4,200 |
| 6 | $5,000 | $500 | $5,500 |

### Moderate (5–8 clients + API early adopters)

| Month | Consulting | API Revenue | Total |
|-------|-----------|-------------|-------|
| 1 | $0 | $0 | $0 |
| 2 | $1,000 | $0 | $1,000 |
| 3 | $4,000 | $0 | $4,000 |
| 4 | $6,000 | $500 | $6,500 |
| 5 | $8,000 | $1,000 | $9,000 |
| 6 | $8,000 | $2,000 | $10,000 |

---

## Prerequisites (in order)

1. **IBP-ENM integration tests passing** (REVIEW_PROPOSAL Phase 1) — backs up the 100% claim
2. **LLC formed** — professional invoicing, liability protection
3. **Business bank account** — separate finances
4. **BioRxiv preprint submitted** — credibility artifact
5. **One-page pitch PDF** — send with cold outreach
6. **Free pilot analysis** for 2–3 prospects — demonstrates value
7. Then build the API (only after validating demand)

---

## What to Build vs What to Outsource

| Task | Build | Outsource | Skip |
|------|-------|-----------|------|
| Integration tests | ✅ | | |
| Preprint | ✅ | | |
| Pitch PDF | ✅ (Canva / LaTeX) | | |
| Landing page services section | ✅ (already have Astro site) | | |
| LLC + EIN | | ✅ (self-service) | |
| Bookkeeping | | ✅ (Wave/QuickBooks) | |
| API MVP | ✅ (when ready) | | |
| Logo/branding refresh | | | ✅ (later) |
| Custom docs site | | | ✅ (README is enough for now) |
| Marketing website | | | ✅ (existing site works) |
