"""
AI Water Footprint: mid-term code

Estimates the water footprint of AI prompts at nine U.S. data center locations
and normalizes it to liters per 10,000 prompts.

Sources:
  Li et al. (2023), Table 1: PUE, on-site WUE, and EWIF for each location
  Elsworth et al. (2025): energy and water for one Gemini prompt (our benchmark)

Needs: pandas, matplotlib
Run:   python water_pipeline_real.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Results (CSV files and the figure) are saved in a folder called "outputs"
# next to this file.
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# STEP 1: Inputs
# ---------------------------------------------------------------
# From Li et al. (2023), Table 1.
#   pue         = how much extra energy the data center uses beyond the servers
#   onsite_wue  = water used on-site for cooling (liters per kWh)
#   ewif        = water used at power plants to make electricity (liters per kWh)
#   li_water_ml = water per request (mL) that Li et al. report, used to check our math
#   li_train_L  = water Li et al. report for training GPT-3 (million liters)
# Georgia* is a projected value (data center was under construction).
data = [
    # location,     pue,   onsite_wue, ewif,  li_water_ml, li_train_L
    ("U.S. Average", 1.170, 0.550,      3.142, 16.904,       5.439),
    ("Arizona",      1.180, 1.630,      4.959, 29.926,       9.629),
    ("Georgia*",     1.120, 0.060,      2.309, 10.585,       3.406),
    ("Illinois",     1.350, 0.740,      2.233, 15.020,       4.833),
    ("Iowa",         1.160, 0.140,      3.104, 14.963,       4.814),
    ("Texas",        1.280, 0.250,      1.287,  7.590,       2.442),
    ("Virginia",     1.140, 0.140,      2.385, 11.435,       3.679),
    ("Washington",   1.150, 0.950,      9.501, 47.506,      15.285),
    ("Wyoming",      1.110, 0.130,      2.574, 11.949,       3.845),
]
df = pd.DataFrame(data, columns=["location", "pue", "onsite_wue", "ewif",
                                 "li_water_ml", "li_train_L"])

# Energy for one prompt (in kWh)
LI_ENERGY = 0.004                 # Li et al.: one GPT-3 request = 0.004 kWh (4 Wh)
ELSWORTH_TOTAL_WH = 0.24          # Elsworth et al.: one Gemini prompt, including overhead
ELSWORTH_OVERHEAD_WH = 0.02       # the overhead part of that 0.24 Wh
# PUE already covers overhead, so we only use the server energy (0.22 Wh).
# Using 0.24 Wh would count the overhead twice.
ELSWORTH_ENERGY = (ELSWORTH_TOTAL_WH - ELSWORTH_OVERHEAD_WH) / 1000

ELSWORTH_WATER_ML = 0.26          # Elsworth et al.: water for one Gemini prompt (on-site)
ELSWORTH_WUE = 1.15               # Elsworth et al.: Google's on-site WUE (liters per kWh)
PROMPTS = 10000                   # we normalize everything to 10,000 prompts

# ---------------------------------------------------------------
# STEP 2: Check our math against Li et al.
# ---------------------------------------------------------------
# Water per request = energy x (on-site WUE + PUE x EWIF)
# We use Li's own energy (0.004 kWh) and see if we get their numbers.
df["our_water_ml"] = LI_ENERGY * (df["onsite_wue"] + df["pue"] * df["ewif"]) * 1000
df["difference_ml"] = df["our_water_ml"] - df["li_water_ml"]

print("STEP 2: Our water per request vs. Li et al. (mL)")
print(df[["location", "li_water_ml", "our_water_ml", "difference_ml"]]
      .round(3).to_string(index=False))

# Check our math against Elsworth et al. too.
# Their formula: (total energy - overhead) x WUE
elsworth_check_ml = ELSWORTH_ENERGY * ELSWORTH_WUE * 1000
print(f"\nElsworth check: we get {elsworth_check_ml:.3f} mL, "
      f"they report {ELSWORTH_WATER_ML} mL")

# ---------------------------------------------------------------
# STEP 3: Normalize to liters per 10,000 prompts
# ---------------------------------------------------------------
# Every location uses the same energy (Elsworth's 0.22 Wh), so the
# only thing that changes between locations is the location itself.
df["onsite_L"] = ELSWORTH_ENERGY * df["onsite_wue"] * PROMPTS
df["offsite_L"] = ELSWORTH_ENERGY * df["pue"] * df["ewif"] * PROMPTS
df["total_L"] = df["onsite_L"] + df["offsite_L"]
df["offsite_percent"] = df["offsite_L"] / df["total_L"] * 100

# ---------------------------------------------------------------
# STEP 4: Compare to the Elsworth benchmark
# ---------------------------------------------------------------
# 0.26 mL per prompt = 2.6 L per 10,000 prompts.
# Elsworth only counts on-site water, so compare it to our on-site column.
benchmark_L = ELSWORTH_WATER_ML * PROMPTS / 1000
df["onsite_vs_benchmark"] = df["onsite_L"] / benchmark_L
df["total_vs_benchmark"] = df["total_L"] / benchmark_L

df = df.sort_values("total_L", ascending=False)

print(f"\nSTEP 4: Water per 10,000 prompts (benchmark = {benchmark_L:.1f} L)")
print(df[["location", "onsite_L", "offsite_L", "total_L", "offsite_percent",
          "onsite_vs_benchmark", "total_vs_benchmark"]]
      .round(2).to_string(index=False))

highest = df["total_L"].max()
lowest = df["total_L"].min()
print(f"\nHighest location is {highest / lowest:.1f} times the lowest")

# ---------------------------------------------------------------
# STEP 5: Training vs. everyday use (GPT-3)
# ---------------------------------------------------------------
# How many requests until the water used for prompts equals the
# water used for training?
df["requests_to_match_training_millions"] = (
    df["li_train_L"] * 1_000_000 / (df["our_water_ml"] / 1000) / 1_000_000)

print("\nSTEP 5: Requests (in millions) until prompt water = training water")
print(df[["location", "li_train_L", "requests_to_match_training_millions"]]
      .round(1).to_string(index=False))

# ---------------------------------------------------------------
# STEP 6: Does the energy per prompt matter?
# ---------------------------------------------------------------
# Same locations, but two different energy values.
df["L_with_elsworth_energy"] = df["total_L"]
df["L_with_li_energy"] = LI_ENERGY * (df["onsite_wue"] + df["pue"] * df["ewif"]) * PROMPTS
df["energy_ratio"] = df["L_with_li_energy"] / df["L_with_elsworth_energy"]

print("\nSTEP 6: Liters per 10,000 prompts with each energy value")
print(df[["location", "L_with_elsworth_energy", "L_with_li_energy", "energy_ratio"]]
      .round(1).to_string(index=False))

# ---------------------------------------------------------------
# STEP 7: Save results and make the figure
# ---------------------------------------------------------------
df.to_csv(OUT / "results_by_region.csv", index=False)
df[["location", "L_with_elsworth_energy", "L_with_li_energy", "energy_ratio"]].to_csv(
    OUT / "energy_per_request_comparison.csv", index=False)

plt.figure(figsize=(8, 4.5))
plt.bar(df["location"], df["onsite_L"], label="On-site (cooling)")
plt.bar(df["location"], df["offsite_L"], bottom=df["onsite_L"],
        label="Off-site (electricity)")
plt.axhline(benchmark_L, color="black", linestyle="--",
            label="Elsworth et al. (2025): 2.6 L")
plt.ylabel("Liters per 10,000 prompts")
plt.title("Modeled water per 10,000 prompts by U.S. location")
plt.xticks(rotation=30, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "water_per_10k_prompts.png", dpi=200)
plt.close()

print(f"\nDone. Files saved in: {OUT}")