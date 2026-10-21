"""
AI water footprint pipeline, real data version.
Inputs: Li et al. (2023), Table 1 (U.S. rows), and Elsworth et al. (2025) benchmark.
Run: python water_pipeline_real.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path(r"C:\Users\Nscoo\Documents\School\Bias & Ethical Implications in AI\Midterm Report")
if not OUT.exists():
    OUT = Path.home() / "CS1684_outputs"
    OUT.mkdir(exist_ok=True)
print(f"Saving outputs to: {OUT}\n")

# ------------------------------------------------------------ 1. INPUTS
# Li et al. (2023), Table 1. PUE, on-site WUE (L/kWh), off-site EWIF (L/kWh),
# and the paper's own reported total water per request (mL) for GPT-3.
# Georgia is marked projected in the paper (data center under construction).
rows = [
    # region,       pue,   onsite_wue, ewif,  li_reported_ml_per_request
    ("U.S. Average", 1.170, 0.550,      3.142, 16.904),
    ("Arizona",      1.180, 1.630,      4.959, 29.926),
    ("Georgia*",     1.120, 0.060,      2.309, 10.585),
    ("Illinois",     1.350, 0.740,      2.233, 15.020),
    ("Iowa",         1.160, 0.140,      3.104, 14.963),
    ("Texas",        1.280, 0.250,      1.287,  7.590),
    ("Virginia",     1.140, 0.140,      2.385, 11.435),
    ("Washington",   1.150, 0.950,      9.501, 47.506),
    ("Wyoming",      1.110, 0.130,      2.574, 11.949),
]
df = pd.DataFrame(rows, columns=["region", "pue", "onsite_wue", "ewif",
                                 "li_reported_ml"])

# Li et al. energy per GPT-3 request (kWh) and Elsworth et al. (2025) Gemini prompt
LI_KWH = 0.004
ELS_TOTAL_WH = 0.24      # total energy per prompt incl. overhead (Elsworth Table 1)
ELS_OVERHEAD_WH = 0.02   # data center overhead (PUE) share (Elsworth Table 1)
# Server energy only, so PUE is not counted twice (Li's formula applies PUE itself)
ELS_KWH = (ELS_TOTAL_WH - ELS_OVERHEAD_WH) / 1000
ELS_WUE = 1.15           # Google fleet on-site WUE, L/kWh (Elsworth Section 3.3)
ELS_ML = 0.26
PROMPTS = 10_000

# ---------------------------------------------- 2. VALIDATION AGAINST LI
# Reproduce Li's Table 1 "total water per request" with their own energy value.
df["li_check_ml"] = LI_KWH * (df["onsite_wue"] + df["pue"] * df["ewif"]) * 1000
df["li_diff_ml"] = df["li_check_ml"] - df["li_reported_ml"]

# --------------------------------- 3. NORMALIZE: WATER PER 10,000 PROMPTS
# Same regional factors, but Elsworth's server energy per prompt, so regions differ
# only by location.
df["onsite_l_per_10k"] = ELS_KWH * df["onsite_wue"] * PROMPTS
df["offsite_l_per_10k"] = ELS_KWH * df["pue"] * df["ewif"] * PROMPTS
df["total_l_per_10k"] = df["onsite_l_per_10k"] + df["offsite_l_per_10k"]
df["offsite_share"] = df["offsite_l_per_10k"] / df["total_l_per_10k"]

# ------------------------------------------------------ 4. BENCHMARK
# Elsworth: 0.26 mL per prompt = 2.6 L per 10,000 prompts.
bench = ELS_ML * PROMPTS / 1000
df["onsite_vs_elsworth"] = df["onsite_l_per_10k"] / bench
df["total_vs_elsworth"] = df["total_l_per_10k"] / bench
implied_wue = ELS_ML / 1000 / ELS_KWH  # L/kWh implied by Elsworth's numbers
# Elsworth's own formula: (E_total - E_overhead) * WUE. Reproduces their 0.26 mL.
els_check_ml = ELS_KWH * ELS_WUE * 1000

# ------------------------------------------------------- 5. RANKING
df["rank_total"] = df["total_l_per_10k"].rank(ascending=False).astype(int)
df["max_to_min"] = df["total_l_per_10k"].max() / df["total_l_per_10k"].min()

# ------------------------------------------------------- 6. OUTPUT
print(f"Elsworth benchmark: {bench:.2f} L per 10,000 prompts "
      f"(implied {implied_wue:.2f} L/kWh)")
print(f"Elsworth formula check: (0.24 - 0.02 Wh) x {ELS_WUE} L/kWh = "
      f"{els_check_ml:.3f} mL (paper reports {ELS_ML} mL)\n")

print("Validation vs Li et al. Table 1 (mL per request, 0.004 kWh):")
print(df[["region", "li_reported_ml", "li_check_ml", "li_diff_ml"]]
      .round(3).to_string(index=False), "\n")

cols = ["region", "onsite_l_per_10k", "offsite_l_per_10k", "total_l_per_10k",
        "offsite_share", "onsite_vs_elsworth", "total_vs_elsworth", "rank_total"]
print("Water per 10,000 prompts (0.22 Wh server energy per prompt):")
print(df[cols].sort_values("rank_total").round(2).to_string(index=False))
print(f"\nHighest / lowest region: {df['max_to_min'].iloc[0]:.1f}x")

df.to_csv(OUT / "results_by_region.csv", index=False)

# Figure: stacked on-site and off-site, Elsworth line
d = df.sort_values("total_l_per_10k", ascending=False)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(d["region"], d["onsite_l_per_10k"], label="On-site (cooling)")
ax.bar(d["region"], d["offsite_l_per_10k"], bottom=d["onsite_l_per_10k"],
       label="Off-site (electricity)")
ax.axhline(bench, color="black", linestyle="--",
           label="Elsworth et al. (2025): 2.6 L")
ax.set_ylabel("Liters per 10,000 prompts")
ax.set_title("Modeled water per 10,000 prompts by U.S. location")
plt.xticks(rotation=30, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "water_per_10k_prompts.png", dpi=200)

# ------------------------------------------- 7. TRAINING VS EVERYDAY USE
# Li et al. Table 1: GPT-3 training water (million L), training energy 1287 MWh.
TRAIN_MWH = 1287
train_total = {"U.S. Average": 5.439, "Arizona": 9.629, "Georgia*": 3.406,
               "Illinois": 4.833, "Iowa": 4.814, "Texas": 2.442,
               "Virginia": 3.679, "Washington": 15.285, "Wyoming": 3.845}
df["li_train_million_l"] = df["region"].map(train_total)
df["train_check_million_l"] = (TRAIN_MWH * 1000
    * (df["onsite_wue"] + df["pue"] * df["ewif"]) / 1e6)
# Number of GPT-3 requests (0.004 kWh each) whose total water equals training water
df["breakeven_million_requests"] = (df["li_train_million_l"] * 1e6
    / (df["li_check_ml"] / 1000)) / 1e6

print("\nTraining vs everyday use (GPT-3, Li et al. Table 1):")
print(df[["region", "li_train_million_l", "train_check_million_l",
          "breakeven_million_requests"]].round(2).to_string(index=False))

# ------------------------------------------- 8. ENERGY PER REQUEST AND VOLUME
# Two primary-source anchors for energy per request. Same location factors for both,
# so the difference comes only from the energy assumption.
#   Li et al. (2023), Section 3.3.2: 0.004 kWh per medium-sized GPT-3 request
#   Elsworth et al. (2025), Table 1: 0.22 Wh server energy, median Gemini prompt
SCENARIOS = {
    "Li et al. GPT-3 request (4 Wh)": LI_KWH,
    "Elsworth et al. Gemini median prompt (0.22 Wh)": ELS_KWH,
}
rows = []
for name, kwh in SCENARIOS.items():
    for _, r in df.iterrows():
        liters_per_req = kwh * (r["onsite_wue"] + r["pue"] * r["ewif"])
        rows.append((name, r["region"], kwh * 1000,
                     liters_per_req * PROMPTS, liters_per_req * 1_000_000))
scale = pd.DataFrame(rows, columns=["scenario", "region", "wh_per_request",
                                    "liters_per_10k", "liters_per_1M"])
pivot = scale.pivot(index="region", columns="scenario", values="liters_per_10k")
pivot = pivot[list(SCENARIOS)]
pivot["energy_ratio"] = pivot.iloc[:, 0] / pivot.iloc[:, 1]
print("\nLiters per 10,000 requests by energy-per-request assumption:")
print(pivot.round(1).to_string())
print("\nLiters per 1,000,000 requests, U.S. Average:")
print(scale[scale["region"] == "U.S. Average"][["scenario", "liters_per_1M"]]
      .round(0).to_string(index=False))

scale.to_csv(OUT / "energy_per_request_comparison.csv", index=False)
df.to_csv(OUT / "results_by_region.csv", index=False)

plt.close('all')