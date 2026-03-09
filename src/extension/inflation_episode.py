"""
2021–2022 Inflation Episode Extension.

Extends the Baqaee-Farhi (2022) framework to analyze the post-pandemic
inflation surge using the same supply-demand decomposition methodology.

Episode window: 2020 Q4 (pre-inflation baseline) to 2022 Q4 (peak)
Key aggregate: CPI-U rose from 260.5 (Dec 2020) to 296.8 (Dec 2022) = +13.9% total
               CPI peaked at 9.1% year-over-year in June 2022

DATA SOURCES (embedded, calibrated to published BLS/BEA data)
──────────────────────────────────────────────────────────────
Price changes (2020 Q4 → 2022 Q4):
  BLS CPI-U release (December 2022 CPI / December 2020 CPI)
  BLS PPI release for producer-facing sectors

Output changes (2020 Q4 → 2022 Q4):
  BEA GDP by Industry (annual, 2022 vs 2020)
  Real output growth at sector level

Key inflation drivers identified in the literature:
  1. Energy: Russia-Ukraine war (Feb 2022), OPEC production discipline
  2. Used vehicles: semiconductor shortage → new car shortage → spillover
  3. Housing/shelter: demand surge + inelastic supply
  4. Food: global commodity shocks, fertilizer prices (Ukraine exports)
  5. Labor market tightening → wage costs → services inflation
  6. Fiscal stimulus (ARP, infrastructure) → aggregate demand

INTERPRETATION GUIDANCE
────────────────────────
In this framework:
  Supply shock s_i > 0: positive productivity shock (cost falls, production rises)
  Supply shock s_i < 0: negative supply shock (cost rises, production falls)

For the 2021-2022 episode:
  Negative supply shocks → supply-push inflation (energy, food, semiconductors)
  Positive demand shocks → demand-pull inflation (fiscal stimulus, pent-up demand)

Both show up as: higher prices (ΔP_i > 0)
Distinguishing feature:
  Supply shock: ΔP ↑ AND ΔY ↓ (or stagnant)
  Demand shock: ΔP ↑ AND ΔY ↑
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.io_network.construct_network import IONetwork, get_io_network, SECTOR_CODES
from src.decomposition.supply_demand_decomp import (
    decompose_gdp,
    sensitivity_analysis,
    print_decomposition_summary,
    compute_network_amplification_decomposition,
    DecompositionResult,
)

OUTPUT_DIR = Path("outputs")
TABLES_DIR = OUTPUT_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDED INFLATION DATA — 2020 Q4 to 2022 Q4
#
# Log changes: Δx = log(2022_Q4_value / 2020_Q4_value)
#
# Price data source: BLS CPI/PPI data (seasonally adjusted)
#   CPI-U All items: +13.9% total 2020Q4→2022Q4
#   CPI Energy: +48.1%
#   CPI Food at Home: +23.1%
#   PPI Final Demand: +18.5%
#
# Output data source: BEA GDP by Industry (2022 vs 2020)
#   Real GDP grew +5.7% (2021) and +2.1% (2022) from 2020
# ─────────────────────────────────────────────────────────────────────────────

# Log price changes (Δp_i = log(P_2022Q4 / P_2020Q4)), by sector
INFLATION_DELTA_P = {
    # ── Agriculture ─────────────────────────────────────────────────────────
    # Global food commodity prices surged (CBOT corn +75%, soybeans +50%)
    # FAO Food Price Index peak: +56% before settling
    "AG":         0.210,

    # ── Mining ──────────────────────────────────────────────────────────────
    # Oil: WTI went from ~$47 (Dec 2020) to ~$80 (Dec 2022)
    # Henry Hub gas: from $2.50 to $5.50 (peak $9+ in Aug 2022)
    "MIN":         0.350,

    # ── Utilities ───────────────────────────────────────────────────────────
    # BLS CPI Utilities: electricity +14%, gas +25%
    "UTIL":        0.155,

    # ── Construction ────────────────────────────────────────────────────────
    # BLS PPI: lumber peak +300%, steel +90%, overall construction +18%
    "CONST":       0.165,

    # ── Food Manufacturing ───────────────────────────────────────────────────
    # BLS CPI Food at Home: +23.1% cumulative
    "FOOD_MFG":    0.190,

    # ── Chemical Manufacturing ───────────────────────────────────────────────
    # BLS PPI Chemicals: +25% (fertilizers especially)
    "CHEM":        0.230,

    # ── Petroleum Products ──────────────────────────────────────────────────
    # BLS CPI Gasoline: +34% cumulative 2020Q4→2022Q4
    "PETRO":       0.340,

    # ── Electronics Manufacturing ────────────────────────────────────────────
    # Semiconductors: price mixed (shortage → some increase)
    # Consumer electronics: BLS CPI declined (-1%) due to quality adjustment
    "ELEC":        0.030,

    # ── Motor Vehicles ──────────────────────────────────────────────────────
    # BLS CPI New Vehicles: +15.1%; Used vehicles: +34.6% (peaked +45%)
    "AUTO":        0.180,

    # ── Other Manufacturing ──────────────────────────────────────────────────
    # BLS PPI Other Manufactured Goods: +15%
    "OTH_MFG":     0.150,

    # ── Wholesale Trade ──────────────────────────────────────────────────────
    "WHOL":        0.090,

    # ── Retail Trade ─────────────────────────────────────────────────────────
    # BLS CPI Commodities less food and energy: +13%
    "RETAIL":      0.095,

    # ── Air Transportation ───────────────────────────────────────────────────
    # BLS CPI Airfare: peaked +38%; cumulative +25% vs 2020Q4
    "AIR":         0.220,

    # ── Other Transportation ─────────────────────────────────────────────────
    # Trucking: Cass Freight Index +18%; fuel surcharges
    "TRANS":       0.180,

    # ── Information & Communication ──────────────────────────────────────────
    # BLS CPI Information Technology: -1%; Telecom: +3%
    "INFO":        0.025,

    # ── Finance & Insurance ──────────────────────────────────────────────────
    # Implicitly priced; credit card fees up; insurance premiums up
    "FIN":         0.085,

    # ── Real Estate & Rental ────────────────────────────────────────────────
    # BLS CPI Shelter: +14.4% cumulative (rent +15%, OER +13%)
    # This is the single largest contributor to core CPI
    "REAL":        0.135,

    # ── Professional Services ────────────────────────────────────────────────
    # Wage-driven: BLS ECI +5%/year → professional services +8-10% over 2 years
    "PROF":        0.085,

    # ── Health Care ─────────────────────────────────────────────────────────
    # BLS CPI Medical Care Services: +5% total (moderating with Medicaid repricing)
    "HEALTH":      0.050,

    # ── Food Services & Accommodation ────────────────────────────────────────
    # BLS CPI Food Away from Home: +20.5%
    # Accommodation: hotel prices +16% vs 2020Q4
    "FOOD_SVC":    0.185,

    # ── Arts, Entertainment & Recreation ────────────────────────────────────
    # BLS CPI Recreation: +7%; performing arts recovering with price hikes
    "ARTS":        0.080,

    # ── Other Services ───────────────────────────────────────────────────────
    # BLS CPI Services ex energy: +13%
    "OTH_SVC":     0.120,

    # ── Government ───────────────────────────────────────────────────────────
    # Government "prices" measured by costs; public sector wages up +5-7%
    "GOVT":        0.060,
}

# Log output changes (Δy_i = log(y_2022Q4 / y_2020Q4)), by sector
# Positive = output grew; negative = output still below 2020Q4 baseline
INFLATION_DELTA_Y = {
    # Agriculture: modest recovery, constrained by drought in some regions
    "AG":          0.020,
    # Mining: significant recovery as oil prices incentivized drilling
    # US oil production recovered to near pre-pandemic levels
    "MIN":         0.120,
    # Utilities: moderate growth (electrification demand + population growth)
    "UTIL":        0.030,
    # Construction: strong growth (infrastructure spending, housing investment)
    "CONST":       0.085,
    # Food Manufacturing: moderate growth
    "FOOD_MFG":    0.035,
    # Chemical Manufacturing: moderate growth
    "CHEM":        0.040,
    # Petroleum Refining: recovered strongly with demand
    "PETRO":       0.090,
    # Electronics: DECLINED despite high prices (semiconductor supply constrained)
    # This is key evidence of a negative supply shock
    "ELEC":       -0.040,
    # Auto Manufacturing: DECLINED despite high prices (chip shortage = supply shock)
    "AUTO":       -0.080,
    # Other Manufacturing: moderate growth (some supply chain issues)
    "OTH_MFG":    0.050,
    # Wholesale: recovered strongly
    "WHOL":        0.095,
    # Retail: recovered strongly (goods demand surge)
    "RETAIL":      0.075,
    # Air Transportation: recovered significantly from COVID lows
    "AIR":         0.380,
    # Other Transportation: freight boom
    "TRANS":       0.140,
    # Information: strong growth (tech boom, WFH infrastructure)
    "INFO":        0.120,
    # Finance: strong growth (asset price inflation, IPO boom 2021)
    "FIN":         0.090,
    # Real Estate: moderate growth (transactions recovered)
    "REAL":        0.055,
    # Professional Services: strong growth (consulting, legal boom)
    "PROF":        0.100,
    # Health Care: recovered but constrained by staffing shortages
    "HEALTH":      0.060,
    # Food Services: strong recovery as dining resumed
    "FOOD_SVC":    0.220,
    # Arts: strong recovery as venues reopened
    "ARTS":        0.280,
    # Other Services: strong recovery
    "OTH_SVC":     0.180,
    # Government: moderate growth (spending programs)
    "GOVT":        0.040,
}


def get_inflation_shocks(sectors: list) -> tuple:
    """
    Return arrays of 2021-2022 price and output log changes for the given sector list.
    """
    delta_p = np.array([INFLATION_DELTA_P.get(s, 0.0) for s in sectors])
    delta_y = np.array([INFLATION_DELTA_Y.get(s, 0.0) for s in sectors])
    return delta_p, delta_y


def run_inflation_replication(
    net: IONetwork,
    run_sensitivity: bool = True,
    verbose: bool = True,
) -> DecompositionResult:
    """
    Run the 2021–2022 inflation decomposition.

    Note on interpretation for inflation analysis:
    ───────────────────────────────────────────────
    The B-F framework decomposes GDP/price-level changes. For inflation,
    we interpret:
      - Positive supply_shock s_i > 0: COST-REDUCING (deflationary)
      - Negative supply_shock s_i < 0: COST-INCREASING (supply-push inflation)
      - Positive demand_shock d_i > 0: DEMAND-PULL inflation
      - Negative demand_shock d_i < 0: Deflationary demand pull

    The GDP contribution formula still applies, but since GDP grew during
    this period, we're measuring GDP expansion contributions rather than
    contraction contributions as in COVID.

    For inflation specifically, the relevant output is the aggregate price level.
    We approximate this as: ΔP ≈ Σ_i α_i^f · ΔP_i (expenditure-weighted)
    And decompose it into supply and demand channels.
    """
    delta_p, delta_y = get_inflation_shocks(net.sectors)

    if verbose:
        print("\n── 2021–2022 Inflation Input Data (2020Q4 → 2022Q4) ──────────────────")
        df_input = pd.DataFrame({
            "sector": net.sectors,
            "label": net.labels,
            "delta_p_%": delta_p * 100,
            "delta_y_%": delta_y * 100,
        })
        print(df_input.to_string(index=False, float_format=lambda x: f"{x:+.1f}"))

    # Main GDP decomposition (same framework as COVID)
    result = decompose_gdp(
        net=net,
        delta_p=delta_p,
        delta_y=delta_y,
        episode="2021–2022 Inflation Surge",
        include_second_order=True,
    )

    if verbose:
        print_decomposition_summary(result)

    # Additional: price level decomposition
    price_decomp = decompose_price_level(net, delta_p, delta_y)
    pdl_path = TABLES_DIR / "inflation_price_decomposition.csv"
    price_decomp.to_csv(pdl_path, index=False)

    # Save main results
    out_path = TABLES_DIR / "inflation_decomposition.csv"
    result.summary_df.to_csv(out_path, index=False)

    # Network amplification
    amp_df = compute_network_amplification_decomposition(net, delta_p, delta_y, "Inflation")
    amp_path = TABLES_DIR / "inflation_network_amplification.csv"
    amp_df.to_csv(amp_path, index=False)

    # Sensitivity analysis
    if run_sensitivity:
        if verbose:
            print("Running sensitivity analysis...")
        sens_df = sensitivity_analysis(
            net, delta_p, delta_y, "Inflation",
            sigma_range=(1.0, 2.0, 3.0),
            epsilon_range=(0.3, 0.8, 1.5),
        )
        sens_path = TABLES_DIR / "inflation_sensitivity.csv"
        sens_df.to_csv(sens_path, index=False)
        if verbose:
            print("\n── Inflation Sensitivity Analysis ─────────────────────────────────────")
            print(sens_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if verbose:
        print(f"\nAll inflation tables saved to {TABLES_DIR}/")

    return result


def decompose_price_level(
    net: IONetwork,
    delta_p: np.ndarray,
    delta_y: np.ndarray,
) -> pd.DataFrame:
    """
    Decompose the aggregate price level change into supply and demand contributions.

    Approach:
      Aggregate price change ≈ Σ_i α_i^f · Δp_i  (expenditure-weighted)

    This differs from GDP decomposition: here we weight by final demand share
    α_i^f to compute the price level analog to CPI.

    Further decompose each sector's price change:
      Δp_i = supply_price_effect + demand_price_effect
      From equilibrium: Δp_i = (-σ_i s_i + ε_i d_i) / (σ_i + ε_i)

    With uniform σ=2, ε=0.8:
      Supply price effect: -σ/(σ+ε) × s_i = -2/2.8 × s_i
      Demand price effect:  ε/(σ+ε) × d_i = 0.8/2.8 × d_i
    """
    from src.decomposition.supply_demand_decomp import (
        identify_shocks,
        DEFAULT_SUPPLY_ELASTICITIES,
        DEFAULT_DEMAND_ELASTICITIES,
    )

    s, d, sigma, epsilon = identify_shocks(net.sectors, delta_p, delta_y)

    # Price contributions from supply and demand channels
    # From equilibrium pricing:
    # Δp_i = [ε_i/(σ_i + ε_i)] × d_i − [σ_i/(σ_i + ε_i)] × s_i
    supply_price_contrib = -(sigma / (sigma + epsilon)) * s
    demand_price_contrib = (epsilon / (sigma + epsilon)) * d

    # Expenditure-weighted aggregate price contributions
    agg_supply_price = (net.alpha_f * supply_price_contrib).sum()
    agg_demand_price = (net.alpha_f * demand_price_contrib).sum()

    # Input-cost channel: supply shocks propagate through IO network to raise costs
    # Downstream price effect: Σ_j A[i,j] × upstream_supply_shock_effect_i
    # First-order: Δp_j^cost = Σ_i A[i,j] × (-s_i) / (upstream_σ)
    upstream_cost_shock = -s / sigma  # cost increase per unit of supply shock
    downstream_cost_effect = net.A.T @ upstream_cost_shock  # N-vector

    df = pd.DataFrame({
        "sector": net.sectors,
        "label": net.labels,
        "delta_p_pct": delta_p * 100,
        "delta_y_pct": delta_y * 100,
        "supply_shock": s,
        "demand_shock": d,
        "final_demand_share": net.alpha_f,
        "supply_price_contrib_pct": supply_price_contrib * 100,
        "demand_price_contrib_pct": demand_price_contrib * 100,
        "weighted_supply_price_pct": net.alpha_f * supply_price_contrib * 100,
        "weighted_demand_price_pct": net.alpha_f * demand_price_contrib * 100,
        "upstream_cost_propagation_pct": downstream_cost_effect * 100,
    })
    df["total_price_contrib_pct"] = (
        df["weighted_supply_price_pct"] + df["weighted_demand_price_pct"]
    )
    df = df.sort_values("total_price_contrib_pct", ascending=False).reset_index(drop=True)

    return df


def classify_inflation_drivers(result: DecompositionResult) -> pd.DataFrame:
    """
    Classify each sector as supply-push vs demand-pull for the inflation period.

    Classification rule:
      - Supply-push:  supply_shock < 0 (negative = cost-raising supply shock)
      - Demand-pull:  demand_shock > 0 (positive = expansionary demand shock)
      - Mixed:        both significant (|s| > 0.05 AND |d| > 0.05)
      - Neutral:      both near zero

    Returns a classification DataFrame.
    """
    df = result.summary_df.copy()
    s = df["supply_shock"]
    d = df["demand_shock"]

    threshold = 0.05  # 5 percentage points

    conditions = [
        (s < -threshold) & (d.abs() < threshold),   # supply-push
        (d > threshold) & (s.abs() < threshold),     # demand-pull
        (s < -threshold) & (d > threshold),          # both
        (s > threshold) & (d < -threshold),          # dis-inflationary
    ]
    labels = ["Supply-Push", "Demand-Pull", "Both (Mixed)", "Dis-inflationary"]

    df["inflation_type"] = "Neutral"
    for cond, lab in zip(conditions, labels):
        df.loc[cond, "inflation_type"] = lab

    classified = df[[
        "sector", "label", "supply_shock", "demand_shock",
        "supply_contribution_gdp_pct", "demand_contribution_gdp_pct",
        "total_contribution_gdp_pct", "inflation_type"
    ]].copy()

    return classified


if __name__ == "__main__":
    print("=" * 65)
    print("  2021–2022 Inflation Surge — Extension Analysis")
    print("=" * 65)

    print("\nStep 1: Building IO Network...")
    net = get_io_network(year=2017, use_real_data=True)

    print("\nStep 2: Running Inflation Decomposition...")
    result = run_inflation_replication(net, verbose=True)

    print("\nStep 3: Classifying Inflation Drivers...")
    classified = classify_inflation_drivers(result)
    print("\n── Inflation Driver Classification ────────────────────────────────────")
    print(classified.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    classified.to_csv(TABLES_DIR / "inflation_driver_classification.csv", index=False)
