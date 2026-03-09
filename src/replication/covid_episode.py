"""
COVID-19 Episode Replication — Feb to May 2020.

Replicates Table 2 and Figures 2–3 from Baqaee & Farhi (2022):
  - Sectoral contributions to the Feb–May 2020 GDP decline
  - Supply vs. demand decomposition
  - Network amplification effects

DATA SOURCES (embedded, calibrated to published BEA/BLS data)
──────────────────────────────────────────────────────────────
Output changes:
  BEA Advance GDP Estimate Q2 2020 (July 2020): GDP declined 32.9% annualized
  = approximately −9.5% in levels Feb–May 2020 (3-month window)

  Sector-level proxies from:
    - BLS Monthly Output: Quarterly of Earnings + MFP program
    - Census Monthly Retail Trade Survey (retail, food service)
    - BTS Air Traffic data (air transportation)
    - Federal Reserve Economic Data (various)
    - BEA Monthly GDP by industry estimates (released later in 2020)

Price changes:
  BLS CPI (consumer-facing) and PPI (producer-facing), Feb–May 2020
  These are log changes (May 2020 / Feb 2020).

Key aggregate: GDP declined ≈ −9.5% in levels (Feb to May 2020)
  Source: BEA NIPA Table 1.1.6, Q1 and Q2 2020 data + monthly interpolation

CODING CONVENTIONS
──────────────────
All shocks and contributions are in log units (fractions), not percentages.
Displayed values are multiplied by 100 for readability.
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
# EMBEDDED COVID DATA — Feb to May 2020
#
# Log changes: Δx = log(May_2020 / Feb_2020)
# Calibrated using the following public sources:
#   1. BEA: GDP by Industry Q1/Q2 2020; Monthly GDP estimates
#   2. BLS: PPI release Apr/May 2020; CPI release Apr/May 2020
#   3. BTS: Air traffic statistics Feb–May 2020
#   4. Census: Monthly Retail Trade survey Feb–May 2020
#   5. Federal Reserve FRED database: various high-frequency indicators
#
# See decisions_log.md for detailed source documentation.
# ─────────────────────────────────────────────────────────────────────────────

# Log output changes (Δy_i = log(y_May / y_Feb)), by sector
# Positive = output increased; negative = output declined
COVID_DELTA_Y = {
    # ── Goods sectors ──────────────────────────────────────────────────────
    # Agriculture: modest decline (food demand stable, some supply issues)
    "AG":        -0.030,
    # Mining: large decline (oil price crash → drilling collapse)
    "MIN":       -0.180,
    # Utilities: slight decline (commercial usage fell, residential up)
    "UTIL":      -0.040,
    # Construction: moderate decline (sites shut or slowed)
    "CONST":     -0.120,
    # Food & Beverage Manufacturing: slight decline (some panic buying offset)
    "FOOD_MFG":  -0.040,
    # Chemical Manufacturing: modest decline
    "CHEM":      -0.060,
    # Petroleum Refining: large decline (demand crash)
    "PETRO":     -0.300,
    # Electronics Manufacturing: moderate decline (supply chain + demand)
    "ELEC":      -0.100,
    # Motor Vehicles: large decline (plants shut in April)
    "AUTO":      -0.420,
    # Other Manufacturing: moderate-large decline
    "OTH_MFG":   -0.180,

    # ── Distribution ───────────────────────────────────────────────────────
    # Wholesale Trade: moderate decline
    "WHOL":      -0.080,
    # Retail Trade: moderate decline (essential retail partly offset)
    "RETAIL":    -0.150,

    # ── Transportation ─────────────────────────────────────────────────────
    # Air Transportation: catastrophic decline (TSA throughput −96% peak)
    "AIR":       -0.700,
    # Other Transportation: large decline (transit ridership −80%)
    "TRANS":     -0.350,

    # ── Services ───────────────────────────────────────────────────────────
    # Information: slight increase (streaming, WFH demand)
    "INFO":       0.020,
    # Finance & Insurance: slight decline
    "FIN":       -0.040,
    # Real Estate: slight decline (transactions fell, rents sticky)
    "REAL":      -0.030,
    # Professional Services: moderate decline (WFH-able, some decline)
    "PROF":      -0.060,
    # Health Care: large decline (elective procedures cancelled)
    "HEALTH":    -0.185,
    # Food Services & Accommodation: very large decline (restaurants closed)
    "FOOD_SVC":  -0.430,
    # Arts, Entertainment, Recreation: catastrophic (venues closed)
    "ARTS":      -0.560,
    # Other Services: moderate-large (personal services closed)
    "OTH_SVC":   -0.200,
    # Government: slight increase (emergency response)
    "GOVT":       0.010,
}

# Log price changes (Δp_i = log(P_May / P_Feb)), by sector
# Source: BLS PPI for producer-facing; BLS CPI for consumer-facing
COVID_DELTA_P = {
    # ── Goods sectors ──────────────────────────────────────────────────────
    # Agriculture: slight decline (demand shock for restaurants offset food retail)
    "AG":        -0.030,
    # Mining: large decline (OPEC price war + demand collapse)
    "MIN":       -0.280,
    # Utilities: slight decline
    "UTIL":      -0.020,
    # Construction: slight decline
    "CONST":     -0.015,
    # Food Manufacturing: slight increase (supply chain tensions)
    "FOOD_MFG":   0.025,
    # Chemicals: slight decline
    "CHEM":      -0.020,
    # Petroleum Products: very large decline (oil crash → gasoline −30%)
    "PETRO":     -0.320,
    # Electronics: slight decline (demand softness)
    "ELEC":      -0.015,
    # Motor Vehicles: moderate decline (dealer incentives + demand soft)
    "AUTO":      -0.040,
    # Other Manufacturing: slight decline
    "OTH_MFG":   -0.030,

    # ── Distribution ───────────────────────────────────────────────────────
    "WHOL":      -0.025,
    # Retail: slight decline
    "RETAIL":    -0.020,

    # ── Transportation ─────────────────────────────────────────────────────
    # Air: large decline (airlines slashing fares to stimulate minimal demand)
    "AIR":       -0.200,
    # Other Transportation: large decline (transit fares fixed; trucking fell)
    "TRANS":     -0.080,

    # ── Services ───────────────────────────────────────────────────────────
    # Information: slight increase (streaming price hikes)
    "INFO":       0.010,
    # Finance: slight increase (credit spreads widened = higher cost of credit)
    "FIN":        0.020,
    # Real Estate: slight decline (rents fell in some markets)
    "REAL":      -0.015,
    # Professional Services: slight decline
    "PROF":      -0.015,
    # Health Care: slight increase (PPE and COVID care more expensive)
    "HEALTH":     0.020,
    # Food Services: large decline (demand → restaurants fighting for business)
    "FOOD_SVC":  -0.120,
    # Arts: large decline (some venues reduced prices to attract virtual audiences)
    "ARTS":      -0.100,
    # Other Services: decline
    "OTH_SVC":   -0.060,
    # Government: stable
    "GOVT":       0.000,
}


def get_covid_shocks(sectors: list) -> tuple:
    """
    Return arrays of COVID price and output log changes for the given sector list.

    Parameters
    ----------
    sectors : list of sector codes (must be subset of SECTOR_CODES)

    Returns
    -------
    delta_p : np.ndarray of log price changes
    delta_y : np.ndarray of log output changes
    """
    delta_p = np.array([COVID_DELTA_P.get(s, 0.0) for s in sectors])
    delta_y = np.array([COVID_DELTA_Y.get(s, 0.0) for s in sectors])
    return delta_p, delta_y


def load_bls_covid_shocks(bls_dir: Path = Path("data/raw/bls")) -> tuple:
    """
    Load COVID price and output changes from downloaded BLS files.
    Falls back to embedded data if files not found.
    """
    ppi_file = bls_dir / "ppi_series.csv"
    cpi_file = bls_dir / "cpi_series.csv"

    if not ppi_file.exists():
        print("BLS files not found — using embedded COVID data.")
        return None, None

    print("Loading BLS COVID data...")
    ppi = pd.read_csv(ppi_file)
    cpi = pd.read_csv(cpi_file)
    return ppi, cpi


def run_covid_replication(
    net: IONetwork,
    use_embedded: bool = True,
    run_sensitivity: bool = True,
    verbose: bool = True,
) -> DecompositionResult:
    """
    Run the full COVID episode decomposition.

    Parameters
    ----------
    net            : IONetwork (2017 BEA baseline)
    use_embedded   : if True, use embedded COVID data; else try BLS files
    run_sensitivity: if True, compute sensitivity analysis over elasticities
    verbose        : if True, print results to console

    Returns
    -------
    DecompositionResult
    """
    # Get shocks
    if use_embedded:
        delta_p, delta_y = get_covid_shocks(net.sectors)
    else:
        ppi, cpi = load_bls_covid_shocks()
        if ppi is None:
            delta_p, delta_y = get_covid_shocks(net.sectors)
        else:
            # For now, fall back (full BLS parsing would go here)
            print("  Full BLS parsing not yet implemented — using embedded data.")
            delta_p, delta_y = get_covid_shocks(net.sectors)

    if verbose:
        print("\n── COVID-19 Input Data (Feb–May 2020) ────────────────────────────────")
        df_input = pd.DataFrame({
            "sector": net.sectors,
            "label": net.labels,
            "delta_p_%": delta_p * 100,
            "delta_y_%": delta_y * 100,
        })
        print(df_input.to_string(index=False, float_format=lambda x: f"{x:+.1f}"))

    # Run decomposition
    result = decompose_gdp(
        net=net,
        delta_p=delta_p,
        delta_y=delta_y,
        episode="COVID-19 (Feb–May 2020)",
        include_second_order=True,
    )

    if verbose:
        print_decomposition_summary(result)

    # Save main results table
    out_path = TABLES_DIR / "covid_decomposition.csv"
    result.summary_df.to_csv(out_path, index=False)
    if verbose:
        print(f"Results saved to {out_path}")

    # Network amplification decomposition
    amp_df = compute_network_amplification_decomposition(net, delta_p, delta_y, "COVID")
    amp_path = TABLES_DIR / "covid_network_amplification.csv"
    amp_df.to_csv(amp_path, index=False)

    # Sensitivity analysis
    if run_sensitivity:
        if verbose:
            print("Running sensitivity analysis over elasticity parameters...")
        sens_df = sensitivity_analysis(
            net, delta_p, delta_y, "COVID",
            sigma_range=(1.0, 2.0, 3.0),
            epsilon_range=(0.3, 0.8, 1.5),
        )
        sens_path = TABLES_DIR / "covid_sensitivity.csv"
        sens_df.to_csv(sens_path, index=False)
        if verbose:
            print("\n── Sensitivity Analysis (Supply/Demand Split by Elasticity) ──────────")
            print(sens_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    return result


def generate_paper_table2(result: DecompositionResult, net: IONetwork) -> pd.DataFrame:
    """
    Generate a table analogous to Table 2 of Baqaee & Farhi (2022).

    Table 2 in the paper shows:
    - Sector name
    - Output change (%)
    - Price change (%)
    - Supply shock contribution to GDP (%)
    - Demand shock contribution to GDP (%)
    - Total contribution to GDP (%)
    - Domar weight
    - Leontief multiplier

    Returns the table as a DataFrame and saves to CSV.
    """
    df = result.summary_df.copy()

    # Reorder columns to match paper's format
    table2 = df[[
        "label",
        "delta_y_pct",
        "delta_p_pct",
        "supply_shock",
        "demand_shock",
        "domar_weight",
        "leontief_multiplier",
        "supply_contribution_gdp_pct",
        "demand_contribution_gdp_pct",
        "total_contribution_gdp_pct",
    ]].copy()

    table2.columns = [
        "Sector",
        "Output Chg (%)",
        "Price Chg (%)",
        "Supply Shock",
        "Demand Shock",
        "Domar Weight",
        "Leontief Mult.",
        "Supply → GDP (%)",
        "Demand → GDP (%)",
        "Total → GDP (%)",
    ]

    # Sort by total contribution (most negative first)
    table2 = table2.sort_values("Total → GDP (%)")

    # Add aggregate row
    agg = pd.DataFrame([{
        "Sector": "TOTAL",
        "Output Chg (%)": np.nan,
        "Price Chg (%)": np.nan,
        "Supply Shock": np.nan,
        "Demand Shock": np.nan,
        "Domar Weight": net.lam.sum(),
        "Leontief Mult.": np.nan,
        "Supply → GDP (%)": result.total_supply_contribution * 100,
        "Demand → GDP (%)": result.total_demand_contribution * 100,
        "Total → GDP (%)": result.total_delta_gdp * 100,
    }])
    table2 = pd.concat([table2, agg], ignore_index=True)

    out_path = TABLES_DIR / "table2_covid_contributions.csv"
    table2.to_csv(out_path, index=False, float_format="%.3f")
    print(f"\nTable 2 saved to {out_path}")

    return table2


def aggregate_contributions_by_type(result: DecompositionResult) -> pd.DataFrame:
    """
    Aggregate sector contributions into meaningful groups for reporting.

    Groups:
      Contact-intensive services: FOOD_SVC, ARTS, AIR, TRANS, RETAIL
      Manufacturing: AUTO, OTH_MFG, ELEC, FOOD_MFG, CHEM, PETRO
      Energy: MIN, UTIL
      Other services: HEALTH, PROF, FIN, INFO, REAL
      Agriculture & Construction: AG, CONST
      Government: GOVT
    """
    groups = {
        "Contact-Intensive Services": ["FOOD_SVC", "ARTS", "AIR", "TRANS", "OTH_SVC"],
        "Manufacturing": ["AUTO", "OTH_MFG", "ELEC", "FOOD_MFG", "CHEM", "PETRO"],
        "Energy & Mining": ["MIN", "UTIL"],
        "Other Services": ["HEALTH", "PROF", "FIN", "INFO", "REAL", "WHOL", "RETAIL"],
        "Ag & Construction": ["AG", "CONST"],
        "Government": ["GOVT"],
    }

    df = result.summary_df.set_index("sector")
    rows = []
    for group_name, sectors in groups.items():
        available = [s for s in sectors if s in df.index]
        sub = df.loc[available]
        rows.append({
            "group": group_name,
            "supply_contribution_pct": sub["supply_contribution_gdp_pct"].sum(),
            "demand_contribution_pct": sub["demand_contribution_gdp_pct"].sum(),
            "total_contribution_pct": sub["total_contribution_gdp_pct"].sum(),
            "n_sectors": len(available),
        })

    agg_df = pd.DataFrame(rows)
    agg_df["demand_share"] = (
        agg_df["demand_contribution_pct"].abs()
        / (agg_df["supply_contribution_pct"].abs() + agg_df["demand_contribution_pct"].abs() + 1e-10)
    )
    return agg_df


if __name__ == "__main__":
    print("=" * 65)
    print("  Baqaee-Farhi (2022) — COVID Episode Replication")
    print("=" * 65)

    # Build IO network
    print("\nStep 1: Building IO Network...")
    net = get_io_network(year=2017, use_real_data=True)

    from src.io_network.construct_network import verify_network, save_network
    verify_network(net)
    save_network(net)

    # Run COVID decomposition
    print("\nStep 2: COVID Supply-Demand Decomposition...")
    result = run_covid_replication(net, verbose=True)

    # Generate Table 2 analogue
    print("\nStep 3: Generating Table 2 analogue...")
    table2 = generate_paper_table2(result, net)
    print(table2.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Aggregate by sector type
    print("\nStep 4: Aggregated contributions by sector group...")
    agg = aggregate_contributions_by_type(result)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    agg.to_csv(TABLES_DIR / "covid_grouped_contributions.csv", index=False)
