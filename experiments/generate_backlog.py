"""Generate the phase-ordered backlog of experiment specs.

Each line in backlog.jsonl is a spec consumable by run_experiment.py.
Phases run in order; orchestrator pulls from the front of the queue.
"""
import json
from pathlib import Path

EXP = Path(__file__).resolve().parent
BACKLOG = EXP / "backlog.jsonl"

CURRENT_PCA = "_L126_K10_M20"

# Current-best baseline parameters (after sign fix + EWMA + band)
BEST_BASE = dict(
    pca_suffix=CURRENT_PCA,
    weight_type="equal",
    invert=True,
    c=1.0,
    w_max=0.05,
    G=1.0,
    smooth_alpha=0.3,
    band=0.0025,
    holding_days=None,
)

specs = []
i = 0
def add(phase, family, **overrides):
    global i
    spec = {"id": f"e{i:03d}", "phase": phase, "family": family, **BEST_BASE, **overrides}
    spec["notes"] = overrides.pop("notes", "") if "notes" in overrides else ""
    specs.append(spec)
    i += 1


# ─────────────────── Phase A: Rebalance every N days ───────────
# Rebalance every N days reduces turnover by ~1/N. Combined with smoothing,
# this is the cleanest way to capture multi-day signal at low cost.
for N in [2, 3, 5]:
    add("A", "reb_n_days", reb_n_days=N,
        smooth_alpha=0.3, band=0.0025,
        notes=f"reb every {N} days, EWMA(0.3)+band")
# Also raw (no smoothing) reb every N days, to isolate the rebalance-cadence effect
for N in [2, 5]:
    add("A", "reb_n_days", reb_n_days=N,
        smooth_alpha=None, band=None,
        notes=f"reb every {N} days, no smoothing")

# ─────────────────── Phase B: Threshold sweep ───────────────────
# Tighter thresholds = fewer, higher-conviction trades = lower turnover and
# potentially higher per-trade alpha
for c in [0.5, 0.75, 1.25, 1.5, 2.0]:
    add("B", "threshold", c=c, notes=f"c={c}")

# ─────────────────── Phase C: Smoothing/band tuning ─────────────
# Trade off gross alpha (preserved by less smoothing) vs net alpha
# (preserved by more smoothing → less turnover cost)
for alpha in [0.15, 0.5]:
    for band in [0.0, 0.005]:
        add("C", "smoothing", smooth_alpha=alpha, band=band,
            notes=f"alpha={alpha}, band={band}")
# Also: no smoothing at all (just the raw inverted weights), for reference
add("C", "smoothing", smooth_alpha=None, band=None, notes="no smoothing baseline")
# And: aggressive smoothing without band
add("C", "smoothing", smooth_alpha=0.1, band=None, notes="alpha=0.1, no band")

# ─────────────────── Phase D: Timing fix ────────────────────────
# The orchestrator handles the source-code edit and PCA regeneration.
# The specs reference the new PCA suffix that will exist after that.
TIMING_PCA = "_L126_K10_M20_t0"
add("D", "timing", pca_suffix=TIMING_PCA, holding_days=None,
    notes="timing-fixed PCA + invert + smoothing")
add("D", "timing", pca_suffix=TIMING_PCA, smooth_alpha=None, band=None,
    notes="timing-fixed PCA + invert, raw weights")
add("D", "timing", pca_suffix=TIMING_PCA, holding_days=3,
    smooth_alpha=None, band=None,
    notes="timing-fixed PCA + invert + N=3 hold")

# ─────────────────── Phase E: PCA grid (Latin-hypercube) ────────
# Sample 10 (L, K, M) cells from the 3*4*2=24 grid
PCA_CELLS = [
    (60, 5, 20), (60, 10, 50), (60, 15, 20),
    (120, 3, 50), (120, 5, 20), (120, 10, 50), (120, 15, 20),
    (250, 3, 20), (250, 5, 50), (250, 10, 20),
]
for L, K, M in PCA_CELLS:
    pca_suffix = f"_L{L}_K{K}_M{M}"
    add("E", "pca_grid", pca_suffix=pca_suffix,
        smooth_alpha=0.3, band=0.0025,
        notes=f"L={L} K={K} M={M}")

# ─────────────────── Phase F: Quantile signal ──────────────────
add("F", "quantile", weight_type="quantile", q=0.10,
    smooth_alpha=0.3, band=0.0025,
    notes="top/bottom 10% inverted")
add("F", "quantile", weight_type="quantile", q=0.05,
    notes="top/bottom 5% (concentrated)")
add("F", "quantile", weight_type="quantile", q=0.20,
    notes="top/bottom 20% (diffuse)")

# ─────────────────── Phase G: Mid-cap universe ─────────────────
# Orchestrator must first re-run download with rank ranges and regen PCA.
MIDCAP_PCA = "_L126_K10_M20_midcap"
add("G", "universe", pca_suffix=MIDCAP_PCA, notes="mid-cap (ranks 200-400) + invert + smoothing")
add("G", "universe", pca_suffix=MIDCAP_PCA, holding_days=3,
    smooth_alpha=None, band=None, notes="mid-cap + N=3 hold")
add("G", "universe", pca_suffix=MIDCAP_PCA, c=1.5,
    notes="mid-cap + tighter threshold")


with open(BACKLOG, "w") as fh:
    for s in specs:
        fh.write(json.dumps(s) + "\n")

print(f"Wrote {len(specs)} specs to {BACKLOG}")
phase_counts = {}
for s in specs:
    phase_counts[s["phase"]] = phase_counts.get(s["phase"], 0) + 1
for ph in sorted(phase_counts):
    print(f"  Phase {ph}: {phase_counts[ph]}")
