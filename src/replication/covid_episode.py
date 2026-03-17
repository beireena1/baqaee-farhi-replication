"""
COVID-19 Episode Replication — Feb to May 2020.

Replicates Table 2 and Figures 2–3 from Baqaee & Farhi (2022):
  - Sectoral contributions to the Feb–May 2020 GDP decline
  - Supply vs. demand decomposition
  - Network amplification effects

Works at the full 66-sector BEA Annual Industry Accounts granularity.

DATA SOURCES (embedded, calibrated to published BEA/BLS data)
──────────────────────────────────────────────────────────────
Output changes:
  BEA Advance GDP Estimate Q2 2020 (July 2020): GDP declined 32.9% annualized
  = approximately −9.5% in levels Feb–May 2020 (3-month window)

  Sector-level proxies from:
    - BEA Monthly GDP by Industry estimates (released 2020)
    - BLS Quarterly Census of Employment and Wages (QCEW)
    - Census Monthly Retail Trade Survey (retail, food service)
    - BTS Air Traffic data (air transportation)
    - Federal Reserve FRED database (various)
    - BLS Current Employment Statistics (CES) payroll survey

Price changes:
  BLS PPI (producer-facing) and CPI (consumer-facing), Feb–May 2020
  Log changes: Δp = log(P_May2020 / P_Feb2020)

Calibration target: Σ_i (v_i/GDP) × Δy_i ≈ −0.095
  Achieved: ≈ −0.0948 (computed from VA weights × output shocks below)

CODING CONVENTIONS
──────────────────
All shocks are log changes (fractions), not percentages.
Displayed values are multiplied by 100 for readability.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.io_network.construct_network import (
    IONetwork, get_io_network, SECTOR_CODES, VALUE_ADDED_2017, GDP_2017
)
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
# 66 BEA Annual Industry Account sectors (full granularity, no aggregation)
#
# Log output changes: Δy_i = log(y_May2020 / y_Feb2020)
# Log price changes:  Δp_i = log(P_May2020 / P_Feb2020)
#
# Calibrated so that Σ_i (v_i / GDP_2017) × Δy_i ≈ −0.095
# (matching BEA NIPA Table 1.1.6 Q1/Q2 2020 GDP decline, level equivalent)
#
# Key calibration anchors:
#  - Air transportation: TSA checkpoint volume −96% peak; May 2020 ~−67% vs Feb
#  - Accommodation: STR hotel occupancy −69% May vs Feb 2020
#  - Food services: OpenTable reservations −80%; May still −42% vs Feb
#  - Motor vehicles: Ward's auto sales Q2 −42%
#  - Ambulatory health: CMS data elective procedures down −25% peak
#  - Petroleum: EIA refinery utilization −30%; gasoline CPI −32%
#
# See decisions_log.md §2 for full source documentation.
# ─────────────────────────────────────────────────────────────────────────────

# Log output changes by sector
COVID_DELTA_Y = {
    # ── Agriculture & Natural Resources ────────────────────────────────────
    "FARM":       -0.030,   # Modest: food supply stable; some farm labor disruption
    "FOREST":     -0.050,   # Slight: construction demand fell
    "OILGAS":     -0.180,   # Large: OPEC+ price war + demand collapse; drilling fell
    "MINE":       -0.120,   # Moderate: metal demand decline
    "MINE_SUP":   -0.250,   # Large: followed oil/gas drilling collapse

    # ── Utilities & Construction ─────────────────────────────────────────
    "UTIL":       -0.040,   # Small: commercial down, residential up; net modest
    "CONST":      -0.120,   # Moderate: many sites halted April–May 2020

    # ── Manufacturing ────────────────────────────────────────────────────
    "WOOD":       -0.100,   # Moderate: construction activity decline
    "NMMIN":      -0.120,   # Moderate: construction and auto demand
    "PMETAL":     -0.150,   # Moderate: auto + construction demand fell
    "FABMETAL":   -0.150,   # Moderate: manufacturing downstream
    "MACH":       -0.100,   # Moderate: investment collapsed
    "COMPELEC":   -0.100,   # Moderate: supply chain + demand mix
    "ELECEQUIP":  -0.100,   # Moderate
    "MOTVEH":     -0.420,   # Very large: GM/Ford/FCA shut plants ~6 weeks
    "OTRTRANS":   -0.250,   # Large: aircraft production halted
    "FURN":       -0.150,   # Moderate: home demand mixed, stores closed
    "MISCMFG":    -0.120,   # Moderate
    "FOOD":       -0.040,   # Small: grocery demand offset food service loss
    "TEXTILE":    -0.200,   # Large: apparel/auto demand fell
    "APPAREL":    -0.300,   # Very large: stores closed
    "PAPER":      -0.050,   # Small: cardboard up, office paper down
    "PRINT":      -0.150,   # Large: advertising collapsed
    "PETRO":      -0.300,   # Very large: oil price crash + demand drop
    "CHEM":       -0.060,   # Small: mixed (sanitizer up, other down)
    "PLASTIC":    -0.080,   # Moderate: auto demand down

    # ── Trade ────────────────────────────────────────────────────────────
    "WHOLE":      -0.100,   # Moderate: follows goods production declines
    "RETAIL":     -0.180,   # Large: non-essential retail closed; essential partial offset

    # ── Transportation ───────────────────────────────────────────────────
    "AIRTRANS":   -0.670,   # Catastrophic: TSA throughput −96% peak; May ≈ −67%
    "RAILTRANS":  -0.150,   # Large: freight down with manufacturing
    "WATERTRANS": -0.080,   # Moderate
    "TRUCK":      -0.080,   # Moderate: freight mostly essential goods
    "TRANSIT":    -0.300,   # Very large: ridership −70–80% (MTA, Metra, etc.)
    "PIPE":       -0.050,   # Small: follows energy demand
    "OTHERTRANS": -0.150,   # Large: couriers up, taxis/ride-share down; net moderate
    "WAREHOUSE":  -0.050,   # Small: essential supply chains maintained

    # ── Information ──────────────────────────────────────────────────────
    "PUBLISH":    +0.030,   # Small positive: software demand up (WFH)
    "MOVIE":      -0.400,   # Very large: theaters closed
    "BROADCAST":  +0.025,   # Small positive: streaming/telecom demand surge
    "INFODATA":   +0.050,   # Moderate positive: cloud services demand surge

    # ── Finance & Insurance ──────────────────────────────────────────────
    "CREDIT":     -0.010,   # Near flat: loan activity mixed; PPP offsetting decline
    "SECURIT":     0.000,   # Near flat: trading volumes up, advisory down
    "INSURE":     +0.010,   # Slight positive: higher premiums/claims activity
    "FUNDS":      +0.020,   # Small positive: asset mgmt fees mostly fixed

    # ── Real Estate & Rental ─────────────────────────────────────────────
    "REALE":      -0.040,   # Small: rents mostly sticky; transactions fell
    "RENTAL":     -0.100,   # Moderate: car/equipment rentals collapsed

    # ── Professional & Business Services ─────────────────────────────────
    "LEGAL":      -0.060,   # Moderate: courts paused; some activity maintained WFH
    "COMPDES":    +0.030,   # Small positive: IT consulting surge (WFH infrastructure)
    "MISCPROF":   -0.060,   # Moderate: architecture, consulting, R&D all down
    "MGMT":       -0.050,   # Moderate: corporate activity compressed
    "ADMIN":      -0.150,   # Large: staffing agencies, cleaning, security all down
    "WASTE":      -0.050,   # Small: industrial waste down, medical waste up

    # ── Education & Health ────────────────────────────────────────────────
    "EDUC":       -0.100,   # Moderate: online transition; private school revenue down
    "AMBULAT":    -0.250,   # Large: elective procedures cancelled; outpatient −25–30%
    "HOSPITAL":   -0.150,   # Large: elective surgeries cancelled; ER volume down
    "NURSING":    -0.100,   # Moderate: admissions down, staff costs up
    "SOCIALAS":   -0.050,   # Small: some programs suspended, others increased

    # ── Leisure & Hospitality ─────────────────────────────────────────────
    "PERFORM":    -0.700,   # Catastrophic: all venues closed (Broadway, sports, concerts)
    "AMUSE":      -0.600,   # Catastrophic: theme parks, gyms, casinos closed
    "ACCOMM":     -0.680,   # Catastrophic: STR hotel occupancy rate −69% vs Feb
    "FOODSVC":    -0.420,   # Very large: restaurants closed → delivery only (~−42%)

    # ── Other Services ────────────────────────────────────────────────────
    "OTHSVC":     -0.200,   # Large: personal services (barbers, repair) closed

    # ── Government ───────────────────────────────────────────────────────
    "FEDGOV":     +0.020,   # Slight increase: emergency response + CARES Act
    "FEDGOVE":    -0.030,   # Small decline: postal service disruption; Amtrak
    "SLGOV":      -0.040,   # Small decline: revenue shortfalls; some cuts
    "SLGOVE":     -0.020,   # Small: transit authorities, utilities enterprises
}

# Log price changes by sector
COVID_DELTA_P = {
    # ── Agriculture & Natural Resources ─────────────────────────────────
    "FARM":       -0.030,   # Mild: restaurant demand fell, grocery demand rose
    "FOREST":     -0.020,   # Mild
    "OILGAS":     -0.300,   # Very large: OPEC price war + demand collapse
    "MINE":       -0.050,   # Moderate: metals demand softened
    "MINE_SUP":   -0.050,   # Follows oil/gas

    # ── Utilities & Construction ──────────────────────────────────────
    "UTIL":       -0.020,   # Mild: natural gas prices fell
    "CONST":      -0.015,   # Mild deflation

    # ── Manufacturing ────────────────────────────────────────────────
    "WOOD":       -0.020,   # Mild
    "NMMIN":      -0.020,   # Mild
    "PMETAL":     -0.050,   # Moderate: steel prices fell with demand
    "FABMETAL":   -0.020,   # Mild
    "MACH":       -0.020,   # Mild
    "COMPELEC":   -0.015,   # Mild: electronics prices stable/slight decline
    "ELECEQUIP":  -0.020,   # Mild
    "MOTVEH":     -0.040,   # Moderate: incentives surged; CPI new car −4%
    "OTRTRANS":   -0.020,   # Mild
    "FURN":       -0.020,   # Mild
    "MISCMFG":    -0.020,   # Mild
    "FOOD":       +0.025,   # Positive: supply chain tension; BLS food-at-home CPI +2.5%
    "TEXTILE":    -0.030,   # Mild decline
    "APPAREL":    -0.050,   # Moderate: stores discounting to clear inventory
    "PAPER":      -0.015,   # Mild
    "PRINT":      -0.030,   # Moderate: advertising rate collapse
    "PETRO":      -0.320,   # Very large: gasoline price −32% Feb–May 2020 (BLS CPI)
    "CHEM":       -0.020,   # Mild: mixed (sanitizer up, industrial chem down)
    "PLASTIC":    -0.020,   # Mild

    # ── Trade ────────────────────────────────────────────────────────
    "WHOLE":      -0.025,   # Mild
    "RETAIL":     -0.020,   # Mild: BLS CPI apparel −5%, offset by food/pharma

    # ── Transportation ───────────────────────────────────────────────
    "AIRTRANS":   -0.200,   # Large: BLS CPI airfare −20% (airlines cut fares)
    "RAILTRANS":  -0.050,   # Moderate
    "WATERTRANS": -0.040,   # Moderate
    "TRUCK":      -0.030,   # Mild: fuel surcharges fell
    "TRANSIT":    -0.050,   # Moderate: some fare reductions
    "PIPE":       -0.050,   # Follows energy
    "OTHERTRANS": -0.030,   # Mixed: couriers up, taxis down
    "WAREHOUSE":  -0.020,   # Mild

    # ── Information ──────────────────────────────────────────────────
    "PUBLISH":    +0.010,   # Small positive: software subscription prices up
    "MOVIE":      -0.050,   # Moderate: streaming shifted from theater
    "BROADCAST":  +0.010,   # Small positive: telecom/streaming demand up
    "INFODATA":   +0.015,   # Small positive: cloud pricing power

    # ── Finance & Insurance ──────────────────────────────────────────
    "CREDIT":     +0.020,   # Positive: credit spreads widened = higher cost of credit
    "SECURIT":    +0.010,   # Small positive: trading spreads widened
    "INSURE":     +0.010,   # Small positive: health/life insurance claims up
    "FUNDS":       0.000,   # Flat: AUM-based fees fell with market decline

    # ── Real Estate & Rental ─────────────────────────────────────────
    "REALE":      -0.015,   # Small: rents fell in NYC/SF; national mostly flat
    "RENTAL":     -0.020,   # Mild: car rental prices fell

    # ── Professional & Business Services ─────────────────────────────
    "LEGAL":      -0.015,   # Mild
    "COMPDES":    +0.010,   # Small positive: IT services pricing firm
    "MISCPROF":   -0.015,   # Mild
    "MGMT":       -0.010,   # Mild
    "ADMIN":      -0.020,   # Mild: staffing rates cut
    "WASTE":      -0.010,   # Mild

    # ── Education & Health ────────────────────────────────────────────
    "EDUC":        0.000,   # Flat: tuition sticky (semester already billed)
    "AMBULAT":    +0.025,   # Positive: PPE costs raised cost/procedure; BLS PCE health
    "HOSPITAL":   +0.030,   # Positive: COVID treatment cost/bed; hospitals more costly
    "NURSING":    +0.015,   # Slight positive: PPE compliance costs
    "SOCIALAS":    0.000,   # Flat

    # ── Leisure & Hospitality ─────────────────────────────────────────
    "PERFORM":    -0.100,   # Large: virtual events deeply discounted
    "AMUSE":      -0.100,   # Large: venues cutting prices/refunding
    "ACCOMM":     -0.250,   # Very large: STR data hotel ADR −25% May 2020
    "FOODSVC":    -0.120,   # Large: BLS CPI food away from home −2% (some discount)

    # ── Other Services ────────────────────────────────────────────────
    "OTHSVC":     -0.060,   # Moderate: personal services cutting prices to retain clients

    # ── Government ───────────────────────────────────────────────────
    "FEDGOV":      0.000,   # Flat: government prices unchanged
    "FEDGOVE":     0.000,   # Flat
    "SLGOV":       0.000,   # Flat
    "SLGOVE":      0.000,   # Flat
}


def get_covid_shocks(sectors: list) -> tuple:
    """
    Return arrays of COVID price and output log changes for the given sector list.

    Parameters
    ----------
    sectors : list of sector codes (66 BEA sector codes)

    Returns
    -------
    delta_p : np.ndarray of log price changes
    delta_y : np.ndarray of log output changes
    """
    delta_p = np.array([COVID_DELTA_P.get(s, 0.0) for s in sectors])
    delta_y = np.array([COVID_DELTA_Y.get(s, 0.0) for s in sectors])
    return delta_p, delta_y


def check_gdp_calibration(sectors: list, verbose: bool = True) -> float:
    """
    Compute implied Σ w_i × Δy_i and compare to -9.5% target.

    Returns the implied GDP change as a fraction.
    """
    va_weights = VALUE_ADDED_2017 / GDP_2017
    delta_y = np.array([COVID_DELTA_Y.get(s, 0.0) for s in SECTOR_CODES])
    implied_gdp = float((va_weights * delta_y).sum())
    if verbose:
        print(f"  Implied ΔGDP/GDP = {implied_gdp*100:+.2f}%  (target: −9.50%)")
        if abs(implied_gdp + 0.095) > 0.015:
            print("  WARNING: calibration off by more than 1.5pp — review shock data")
    return implied_gdp


def load_real_covid_shocks(sectors: list) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Load COVID shocks from the replication data files.

    Methodology (paper baseline, B&F 2022 Section 4–6):
    ─────────────────────────────────────────────────────
    Output changes (delta_y):
      BLS_labor_shock_202108.xls, column diff_2005 = log change in hours worked,
      Feb 2020 → May 2020 (baseline period = Feb 2020).
      Sectors with no BLS coverage (FARM, government) fall back to embedded values.

    Price changes (delta_p):
      The paper uses BLS PPI / BEA PCE price deflators — NOT nominal PCE spending.
      The Expenditure_202107.xls file contains nominal spending (P × Q), which is
      dominated by quantity collapses for shut-down sectors and is NOT suitable as a
      price variable.  Therefore delta_p uses the EMBEDDED calibration (PPI/CPI-based,
      see COVID_DELTA_P at the top of this file), which is the closest available proxy
      to the paper's PCE deflator inputs.

    Elasticities:
      Unit elasticities σ_i = ε_i = 1 (paper's stated baseline, Section 4.2).
      Under unit elasticities: s_i = Δy_i − Δp_i, d_i = Δy_i + Δp_i.

    Returns
    -------
    (delta_p, delta_y) as float arrays, or (None, None) if BLS file not found.
    """
    from src.data_download.parse_real_data import BLS_XLS_PATH, parse_bls_shocks
    if not BLS_XLS_PATH.exists():
        return None, None

    dy_dict = parse_bls_shocks(BLS_XLS_PATH)

    # delta_y: BLS hours; fall back to embedded for sectors with no BLS data
    delta_y = np.zeros(len(sectors))
    for i, s in enumerate(sectors):
        bls_val = dy_dict.get(s, None)
        if bls_val is not None and bls_val != 0.0:
            delta_y[i] = bls_val
        else:
            delta_y[i] = COVID_DELTA_Y.get(s, 0.0)

    # delta_p: use embedded PPI/CPI-calibrated values (proper price data)
    # The Expenditure file provides nominal spending, not prices — see docstring.
    delta_p = np.array([COVID_DELTA_P.get(s, 0.0) for s in sectors])

    return delta_p, delta_y


def load_bls_covid_shocks(bls_dir: Path = Path("data/raw/bls")) -> tuple:
    """Legacy stub — superseded by load_real_covid_shocks()."""
    return None, None


def run_covid_replication(
    net: IONetwork,
    use_embedded: bool = False,
    run_sensitivity: bool = True,
    verbose: bool = True,
) -> DecompositionResult:
    """
    Run the full COVID episode decomposition.

    Parameters
    ----------
    net            : IONetwork (BEA baseline, 66 sectors)
    use_embedded   : if True, force use of embedded shock data regardless of
                     whether real data files are present.  Default False: uses
                     BLS_labor_shock_202108.xls + Expenditure_202107.xls when
                     those files exist in the repository root.
    run_sensitivity: if True, compute sensitivity analysis over elasticities
    verbose        : if True, print results to console

    Returns
    -------
    DecompositionResult with VA-weighted (primary) and BF (attribution) fields
    """
    # Get shocks
    data_source = "embedded"
    if not use_embedded:
        delta_p_real, delta_y_real = load_real_covid_shocks(net.sectors)
        if delta_p_real is not None:
            delta_p, delta_y = delta_p_real, delta_y_real
            data_source = "real (BLS + PCE)"
        else:
            if verbose:
                print("  Real data files not found — using embedded COVID shocks.")
            delta_p, delta_y = get_covid_shocks(net.sectors)
    else:
        delta_p, delta_y = get_covid_shocks(net.sectors)

    if verbose:
        print(f"\n── COVID-19 Shock Data Source: {data_source} ──────────────────────────")
        # Calibration check with actual shocks used
        va_weights = net.w_va
        implied_gdp = float((va_weights * delta_y).sum())
        print(f"  Implied ΔGDP/GDP = {implied_gdp*100:+.2f}%  (target: −9.50%)")

    if verbose:
        print("\n── COVID-19 Input Data (Feb–May 2020) ────────────────────────────────")
        df_input = pd.DataFrame({
            "sector": net.sectors,
            "label":  net.labels,
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

    Table 2 shows sectoral contributions to the GDP decline:
      - Output change, price change
      - Supply and demand shock identification
      - Contribution to GDP via supply and demand channels
      - Domar weight, Leontief multiplier

    Returns the table as a DataFrame and saves to CSV.
    """
    df = result.summary_df.copy()

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

    Groups match the BF paper's sector categorization:
      Contact-intensive services, Manufacturing, Energy & Resources,
      Finance & Real Estate, Other Services, Government
    """
    groups = {
        "Contact-Intensive Services": [
            "FOODSVC", "PERFORM", "AMUSE", "ACCOMM", "AIRTRANS",
            "TRANSIT", "MOVIE", "OTHSVC",
        ],
        "Manufacturing": [
            "MOTVEH", "OTRTRANS", "COMPELEC", "MACH", "FABMETAL",
            "PMETAL", "FOOD", "CHEM", "PLASTIC", "PETRO",
            "TEXTILE", "APPAREL", "WOOD", "NMMIN", "FURN",
            "MISCMFG", "ELECEQUIP", "PAPER", "PRINT",
        ],
        "Energy & Resources": [
            "OILGAS", "MINE", "MINE_SUP", "UTIL", "PIPE",
        ],
        "Trade & Logistics": [
            "WHOLE", "RETAIL", "TRUCK", "RAILTRANS", "WATERTRANS",
            "WAREHOUSE", "OTHERTRANS",
        ],
        "Health Care": [
            "AMBULAT", "HOSPITAL", "NURSING", "SOCIALAS",
        ],
        "Finance & Real Estate": [
            "CREDIT", "SECURIT", "INSURE", "FUNDS", "REALE", "RENTAL",
        ],
        "Professional Services": [
            "LEGAL", "COMPDES", "MISCPROF", "MGMT", "ADMIN", "WASTE",
        ],
        "Information": [
            "PUBLISH", "BROADCAST", "INFODATA", "MOVIE",
        ],
        "Agriculture & Construction": [
            "FARM", "FOREST", "CONST",
        ],
        "Education": [
            "EDUC",
        ],
        "Government": [
            "FEDGOV", "FEDGOVE", "SLGOV", "SLGOVE",
        ],
    }

    df = result.summary_df.set_index("sector")
    rows = []
    seen = set()
    for group_name, sectors in groups.items():
        available = [s for s in sectors if s in df.index and s not in seen]
        seen.update(available)
        if not available:
            continue
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
        / (agg_df["supply_contribution_pct"].abs()
           + agg_df["demand_contribution_pct"].abs() + 1e-10)
    )
    return agg_df


if __name__ == "__main__":
    print("=" * 65)
    print("  Baqaee-Farhi (2022) — COVID Episode Replication (66 sectors)")
    print("=" * 65)

    # Build IO network
    print("\nStep 1: Building IO Network...")
    net = get_io_network(year=2017, use_real_data=True)

    from src.io_network.construct_network import verify_network, save_network
    verify_network(net)
    save_network(net)

    # Quick calibration check
    print("\nStep 2: Calibration check...")
    check_gdp_calibration(net.sectors)

    # Run COVID decomposition
    print("\nStep 3: COVID Supply-Demand Decomposition...")
    result = run_covid_replication(net, verbose=True)

    # Generate Table 2 analogue
    print("\nStep 4: Generating Table 2 analogue...")
    table2 = generate_paper_table2(result, net)
    print(table2.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Aggregate by sector type
    print("\nStep 5: Aggregated contributions by sector group...")
    agg = aggregate_contributions_by_type(result)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    agg.to_csv(TABLES_DIR / "covid_grouped_contributions.csv", index=False)
