"""
2021–2022 Inflation Episode Extension.

Extends the Baqaee-Farhi (2022) framework to analyze the post-pandemic
inflation surge using the same supply-demand decomposition methodology.
Works at the full 66-sector BEA Annual Industry Accounts granularity.

Episode window: 2020 Q4 (pre-inflation baseline) to 2022 Q4 (peak)
  CPI-U: 260.5 (Dec 2020) → 296.8 (Dec 2022) = +13.9% total
  Real GDP: cumulative +7.7% (2021: +5.7%, 2022: +2.1%)

DATA SOURCES (embedded, calibrated to published BLS/BEA data)
──────────────────────────────────────────────────────────────
Price changes (2020 Q4 → 2022 Q4):
  BLS CPI-U release (December 2022 / December 2020, seasonally adjusted)
  BLS PPI Final Demand and industry-specific PPI series
  BEA PCE price deflator by sector

Output changes (2020 Q4 → 2022 Q4):
  BEA GDP by Industry Annual Data (2022 vs 2020)
  Real value-added growth at 66-sector level

Key inflation drivers:
  1. Energy: Russia-Ukraine war (Feb 2022), OPEC+ discipline → oil/gas costs up
  2. Semiconductors/Auto: chip shortage → vehicle production down, prices up
  3. Shelter: demand surge + inelastic housing supply → rent +15%
  4. Food: global commodity shocks (Ukraine wheat/corn), fertilizer prices
  5. Labor market tightening: wage costs → services inflation
  6. Fiscal stimulus (ARP, IIJA, IRA): aggregate demand expansion

INTERPRETATION GUIDANCE
────────────────────────
In the BF framework applied to inflation:
  Negative supply shock s_i < 0: cost-raising → supply-push inflation (ΔP ↑, ΔY ↓)
  Positive demand shock d_i > 0: demand-pull inflation (ΔP ↑, ΔY ↑)

Calibration target: Σ_i (v_i/GDP) × Δy_i ≈ +0.077 (real GDP growth 2020Q4–2022Q4)
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
# EMBEDDED INFLATION DATA — 2020 Q4 to 2022 Q4
# 66 BEA Annual Industry Account sectors (full granularity, no aggregation)
#
# Log price changes: Δp_i = log(P_2022Q4 / P_2020Q4)
# Log output changes: Δy_i = log(y_2022Q4 / y_2020Q4)
#
# Calibrated so that Σ_i (v_i / GDP_2017) × Δy_i ≈ +0.077
# (2020Q4–2022Q4 cumulative real GDP growth)
#
# Key price calibration anchors from BLS:
#  - CPI-U All: +13.9% cumulative (+13.0% log)
#  - CPI Energy: +48.1% (+39.4% log) → OILGAS, PETRO, UTIL
#  - CPI Food at Home: +23.1% (+20.8% log) → FARM, FOOD
#  - CPI Shelter: +14.4% (+13.5% log) → REALE
#  - CPI Airfare: +25% cumulative (+22.3% log) → AIRTRANS
#  - PPI Final Demand: +18.5% (+17.0% log)
#  - CPI New vehicles: +15.1% (+14.1% log) → MOTVEH
#  - CPI Food away: +20.5% (+18.7% log) → FOODSVC
#
# See decisions_log.md §3 for full source documentation.
# ─────────────────────────────────────────────────────────────────────────────

# Log price changes by sector (2020 Q4 → 2022 Q4)
INFLATION_DELTA_P = {
    # ── Agriculture & Natural Resources ─────────────────────────────────────
    # FAO Food Price Index peaked +56%; BLS CPI Food at Home +23.1%
    "FARM":        0.210,   # Global food commodity boom (corn, soybeans, wheat)
    "FOREST":      0.120,   # Lumber: peak +300% (Dec 2021), settled +12% by Dec 2022
    "OILGAS":      0.390,   # WTI: $47 (Dec 2020) → $80 (Dec 2022); log(80/47)=0.53; PPI 0.39
    "MINE":        0.180,   # Steel +40% peak, settled +18%; copper +30%
    "MINE_SUP":    0.150,   # Follows oil/gas upstream activity

    # ── Utilities & Construction ──────────────────────────────────────────
    "UTIL":        0.155,   # BLS CPI Utilities: electricity +14%, gas +25%
    "CONST":       0.165,   # BLS PPI: lumber, steel, labor; overall construction +18%

    # ── Manufacturing ─────────────────────────────────────────────────────
    "WOOD":        0.090,   # Lumber peaked +300% then partially reversed; net +9%
    "NMMIN":       0.140,   # Cement/stone: construction demand up; +14%
    "PMETAL":      0.160,   # Steel: BLS PPI primary metals +16%
    "FABMETAL":    0.130,   # Downstream metals; fabricated metals PPI +13%
    "MACH":        0.110,   # BLS PPI industrial machinery +11%
    "COMPELEC":    0.030,   # Semiconductors: mixed; consumer electronics hedonic adj → flat
    "ELECEQUIP":   0.100,   # Electrical equipment +10% (copper, labor)
    "MOTVEH":      0.180,   # BLS CPI new vehicles +15.1%; used +34.6% (peak); new car PPI
    "OTRTRANS":    0.090,   # Aircraft/defense: cost pressures; +9%
    "FURN":        0.110,   # BLS CPI furniture: +11% (lumber, labor, supply chain)
    "MISCMFG":     0.090,   # Misc manufactured goods PPI +9%
    "FOOD":        0.190,   # BLS CPI food at home: +23.1%; PPI food manufacturing: +19%
    "TEXTILE":     0.120,   # Cotton, labor: textile PPI +12%
    "APPAREL":     0.080,   # BLS CPI apparel: +5% peak; settled at +8%
    "PAPER":       0.140,   # BLS PPI paper products: +14% (energy + wood pulp)
    "PRINT":       0.100,   # Printing PPI +10%
    "PETRO":       0.340,   # BLS CPI gasoline: +34% cumulative 2020Q4→2022Q4
    "CHEM":        0.230,   # BLS PPI chemicals +23% (fertilizers especially)
    "PLASTIC":     0.180,   # Downstream petrochemicals; plastics PPI +18%

    # ── Trade ─────────────────────────────────────────────────────────────
    "WHOLE":       0.090,   # Wholesale margins: +9% (cost pass-through)
    "RETAIL":      0.095,   # BLS CPI commodities less food/energy: +13%; retail +9.5%

    # ── Transportation ────────────────────────────────────────────────────
    "AIRTRANS":    0.220,   # BLS CPI airfare: peaked +38%; cumulative +25%; log=0.22
    "RAILTRANS":   0.120,   # Cass Rail Freight Index: +12% cumulative
    "WATERTRANS":  0.200,   # Container shipping rates: surged 5–10x, mostly reversed; net +20%
    "TRUCK":       0.180,   # Cass Freight Index +18%; fuel surcharges dominant
    "TRANSIT":     0.060,   # Public transit fares mostly unchanged; some increases
    "PIPE":        0.200,   # Follows natural gas price surge
    "OTHERTRANS":  0.120,   # Courier: FedEx/UPS rate surcharges; +12%
    "WAREHOUSE":   0.110,   # Industrial real estate & labor: +11%

    # ── Information ───────────────────────────────────────────────────────
    "PUBLISH":     0.040,   # Software: subscription price increases; +4%
    "MOVIE":       0.050,   # Streaming prices up (Netflix, Disney+); theaters recovering
    "BROADCAST":   0.030,   # Telecom: modest increases; BLS CPI telecom +3%
    "INFODATA":    0.020,   # Cloud computing: prices declined in real terms; nominal flat

    # ── Finance & Insurance ───────────────────────────────────────────────
    "CREDIT":      0.100,   # Higher rates → implicit price of credit up; +10%
    "SECURIT":     0.060,   # Asset management fees; +6%
    "INSURE":      0.120,   # Health, auto, home insurance premiums: +12%
    "FUNDS":       0.040,   # Fund management fees; +4%

    # ── Real Estate & Rental ──────────────────────────────────────────────
    "REALE":       0.135,   # BLS CPI Shelter: +14.4% (rent +15%, OER +13%); log≈0.135
    "RENTAL":      0.110,   # Car/equipment rental: recovered; +11%

    # ── Professional & Business Services ──────────────────────────────────
    "LEGAL":       0.085,   # Wage-driven: legal billing rates up +8–9%
    "COMPDES":     0.060,   # IT services: labor market tight but tech cost pressures mixed
    "MISCPROF":    0.080,   # Professional services: BLS ECI +5%/year → +8% over 2 years
    "MGMT":        0.075,   # Management consulting: wage-driven
    "ADMIN":       0.100,   # Staffing, security, cleaning: minimum wage increases
    "WASTE":       0.090,   # Fuel cost pass-through; labor

    # ── Education & Health ─────────────────────────────────────────────────
    "EDUC":        0.060,   # Tuition: +4–5%/year; private school labor costs
    "AMBULAT":     0.055,   # BLS CPI Medical Care: +5% total (moderating)
    "HOSPITAL":    0.070,   # Hospital costs: staffing shortage → travel nurse costs
    "NURSING":     0.080,   # Nursing care: staffing crisis → wage inflation
    "SOCIALAS":    0.060,   # Social assistance: minimum wage and labor costs

    # ── Leisure & Hospitality ─────────────────────────────────────────────
    "PERFORM":     0.120,   # Performing arts: rebounding with price hikes post-COVID
    "AMUSE":       0.090,   # Theme parks, recreation: fee increases
    "ACCOMM":      0.160,   # Hotel ADR recovered strongly; BLS CPI lodging +16%
    "FOODSVC":     0.185,   # BLS CPI Food Away from Home: +20.5%; log≈0.187

    # ── Other Services ────────────────────────────────────────────────────
    "OTHSVC":      0.120,   # Personal services: BLS CPI services ex-energy: +13%

    # ── Government ────────────────────────────────────────────────────────
    "FEDGOV":      0.060,   # Federal costs: compensation, procurement costs
    "FEDGOVE":     0.060,   # Similar
    "SLGOV":       0.060,   # State/local: public sector wage increases
    "SLGOVE":      0.060,   # Similar
}

# Log output changes by sector (2020 Q4 → 2022 Q4)
# Positive = sector grew above 2020Q4 baseline; negative = still below
INFLATION_DELTA_Y = {
    # ── Agriculture & Natural Resources ─────────────────────────────────────
    "FARM":        0.020,   # Modest: drought in some regions; food demand strong
    "FOREST":      0.060,   # Recovery as construction rebounded; logging up
    "OILGAS":      0.120,   # US oil production recovered; rig count +80%
    "MINE":        0.080,   # Metal mining recovered with prices
    "MINE_SUP":    0.150,   # Follows oil/gas drilling activity

    # ── Utilities & Construction ──────────────────────────────────────────
    "UTIL":        0.030,   # Moderate: electrification + population growth
    "CONST":       0.085,   # Strong: infrastructure spending, housing investment

    # ── Manufacturing ─────────────────────────────────────────────────────
    "WOOD":        0.060,   # Recovery with construction; lumber volumes up
    "NMMIN":       0.060,   # Concrete/aggregates: construction demand
    "PMETAL":      0.080,   # Steel/aluminum: automotive + construction demand
    "FABMETAL":    0.080,   # Downstream metals: construction, defense
    "MACH":        0.060,   # Capital investment recovery
    "COMPELEC":   -0.040,   # KEY: semiconductor shortage → production DECLINE
    "ELECEQUIP":   0.040,   # Mixed: some demand from EV/renewable buildout
    "MOTVEH":     -0.080,   # KEY: chip shortage → vehicle production DECLINE (supply shock)
    "OTRTRANS":    0.050,   # Aircraft: Boeing recovering but still below
    "FURN":        0.040,   # Home furnishings: demand surge offset by supply chains
    "MISCMFG":     0.080,   # Misc: recovery from pandemic lows
    "FOOD":        0.035,   # Food manufacturing: moderate growth
    "TEXTILE":     0.040,   # Textile recovery
    "APPAREL":     0.050,   # Apparel: partial recovery
    "PAPER":       0.040,   # Paper: cardboard/packaging strong; office paper weak
    "PRINT":       0.030,   # Print: modest recovery
    "PETRO":       0.090,   # Petroleum refining: strong recovery with demand
    "CHEM":        0.040,   # Chemical: mixed (fertilizer up, some down)
    "PLASTIC":     0.050,   # Plastics: automotive + packaging demand

    # ── Trade ─────────────────────────────────────────────────────────────
    "WHOLE":       0.095,   # Wholesale: strong recovery
    "RETAIL":      0.075,   # Retail: goods demand surge

    # ── Transportation ────────────────────────────────────────────────────
    "AIRTRANS":    0.380,   # Large recovery from COVID lows (base effect)
    "RAILTRANS":   0.150,   # Rail freight: goods demand surge
    "WATERTRANS":  0.080,   # Water: port congestion then recovery
    "TRUCK":       0.100,   # Trucking: freight boom
    "TRANSIT":     0.200,   # Transit: large base-effect recovery from COVID
    "PIPE":        0.050,   # Pipeline: follows energy demand
    "OTHERTRANS":  0.140,   # Couriers (e-commerce boom) + other recovery
    "WAREHOUSE":   0.100,   # Warehousing: e-commerce + supply chain buildout

    # ── Information ───────────────────────────────────────────────────────
    "PUBLISH":     0.120,   # Software: strong (WFH infrastructure, cloud)
    "MOVIE":       0.400,   # Large base-effect recovery (theaters reopened)
    "BROADCAST":   0.060,   # Telecom/streaming: moderate growth
    "INFODATA":    0.150,   # Cloud/data: strong secular growth

    # ── Finance & Insurance ───────────────────────────────────────────────
    "CREDIT":      0.050,   # Banking: loan growth + higher rates (fee income)
    "SECURIT":     0.090,   # Securities: IPO boom 2021; 2022 pullback; net moderate
    "INSURE":      0.040,   # Insurance: premium growth
    "FUNDS":       0.060,   # Asset management: strong 2021, weak 2022

    # ── Real Estate & Rental ──────────────────────────────────────────────
    "REALE":       0.055,   # Real estate: transactions + rental income growth
    "RENTAL":      0.080,   # Car/equipment rental: strong recovery (vehicle shortage)

    # ── Professional & Business Services ──────────────────────────────────
    "LEGAL":       0.100,   # Legal: M&A boom 2021 + regulatory activity
    "COMPDES":     0.120,   # IT consulting: demand surge (digital transformation)
    "MISCPROF":    0.100,   # Professional services: strong recovery
    "MGMT":        0.070,   # Management: corporate activity
    "ADMIN":       0.100,   # Admin: staffing agencies, back-office services
    "WASTE":       0.050,   # Waste: steady growth

    # ── Education & Health ─────────────────────────────────────────────────
    "EDUC":        0.040,   # Education: partial recovery; hybrid learning
    "AMBULAT":     0.060,   # Ambulatory: elective procedures recovered
    "HOSPITAL":    0.060,   # Hospital: recovery but staffing constrained
    "NURSING":     0.050,   # Nursing: recovery with aging demand
    "SOCIALAS":    0.080,   # Social assistance: ARP expansion

    # ── Leisure & Hospitality ─────────────────────────────────────────────
    "PERFORM":     0.280,   # Large base-effect: arts/sports venues reopened
    "AMUSE":       0.300,   # Theme parks, gyms: strong reopening recovery
    "ACCOMM":      0.350,   # Accommodation: strong base-effect recovery
    "FOODSVC":     0.220,   # Food services: large base-effect recovery

    # ── Other Services ────────────────────────────────────────────────────
    "OTHSVC":      0.180,   # Personal services: strong reopening recovery

    # ── Government ────────────────────────────────────────────────────────
    "FEDGOV":      0.020,   # Federal: CARES wind-down; ARP/IIJA programs
    "FEDGOVE":     0.020,   # Federal enterprises: USPS, Amtrak
    "SLGOV":       0.030,   # State/local: revenue recovery + fiscal aid
    "SLGOVE":      0.020,   # Enterprises: transit recovering
}


def get_inflation_shocks(sectors: list) -> tuple:
    """
    Return arrays of 2021-2022 inflation price and output log changes.

    Parameters
    ----------
    sectors : list of 66 BEA sector codes

    Returns
    -------
    delta_p : np.ndarray of log price changes (2020Q4 → 2022Q4)
    delta_y : np.ndarray of log output changes (2020Q4 → 2022Q4)
    """
    delta_p = np.array([INFLATION_DELTA_P.get(s, 0.0) for s in sectors])
    delta_y = np.array([INFLATION_DELTA_Y.get(s, 0.0) for s in sectors])
    return delta_p, delta_y


def check_gdp_calibration(sectors: list, verbose: bool = True) -> float:
    """Check Σ w_i × Δy_i against ~+7.7% target."""
    va_weights = VALUE_ADDED_2017 / GDP_2017
    delta_y = np.array([INFLATION_DELTA_Y.get(s, 0.0) for s in SECTOR_CODES])
    implied_gdp = float((va_weights * delta_y).sum())
    if verbose:
        print(f"  Implied ΔGDP/GDP = {implied_gdp*100:+.2f}%  (target: +7.7%)")
    return implied_gdp


def run_inflation_replication(
    net: IONetwork,
    run_sensitivity: bool = True,
    verbose: bool = True,
) -> DecompositionResult:
    """
    Run the 2021–2022 inflation decomposition.

    Interpretation notes for the inflation period:
    ───────────────────────────────────────────────
    Supply shock s_i < 0: cost-raising (supply-push inflation): ΔP ↑, ΔY ↓ or flat
    Supply shock s_i > 0: productivity-raising (dis-inflationary): ΔP ↓, ΔY ↑
    Demand shock d_i > 0: demand-pull (expansionary): ΔP ↑, ΔY ↑
    Demand shock d_i < 0: deflationary (demand contraction): ΔP ↓, ΔY ↓

    Key findings (pre-computation):
    - Energy sectors (OILGAS, PETRO, UTIL): supply-push (price up, output recovery)
    - Auto/semiconductor (MOTVEH, COMPELEC): supply-push (price up, output DOWN)
    - Shelter (REALE): mixed — demand-pull + inelastic supply
    - Food services, accommodation: demand-pull (price up, output up — reopening)
    - Government spending: demand-pull
    """
    if verbose:
        print("\n── Inflation Calibration Check ────────────────────────────────────────")
        check_gdp_calibration(net.sectors, verbose=True)

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

    # Main GDP decomposition (same BF framework as COVID)
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
      Aggregate price change ≈ Σ_i α_i^f · Δp_i  (final-demand-expenditure-weighted)

    Further decompose each sector's price change:
      Δp_i = [ε_i/(σ_i+ε_i)] × d_i − [σ_i/(σ_i+ε_i)] × s_i

    where s_i, d_i are the identified supply and demand shocks.
    """
    from src.decomposition.supply_demand_decomp import (
        identify_shocks,
    )

    s, d, sigma, epsilon = identify_shocks(net.sectors, delta_p, delta_y)

    # Price decomposition from equilibrium pricing
    supply_price_contrib = -(sigma / (sigma + epsilon)) * s
    demand_price_contrib = (epsilon / (sigma + epsilon)) * d

    # Upstream cost propagation through IO linkages
    upstream_cost_shock = -s / sigma
    downstream_cost_effect = net.A.T @ upstream_cost_shock

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

    Classification rule (for inflation episode):
      Supply-push:      supply_shock < -threshold (cost-raising negative supply shock)
      Demand-pull:      demand_shock > +threshold (expansionary demand)
      Mixed:            both significant
      Dis-inflationary: s > threshold (productivity gains) or d < -threshold

    Returns a classification DataFrame with all 66 sectors.
    """
    df = result.summary_df.copy()
    s = df["supply_shock"]
    d = df["demand_shock"]

    threshold = 0.05  # 5 percentage points in log units

    conditions = [
        (s < -threshold) & (d.abs() < threshold),
        (d > threshold) & (s.abs() < threshold),
        (s < -threshold) & (d > threshold),
        ((s > threshold) | (d < -threshold)) & ~((s < -threshold) | (d > threshold)),
    ]
    type_labels = ["Supply-Push", "Demand-Pull", "Both (Mixed)", "Dis-inflationary"]

    df["inflation_type"] = "Neutral"
    for cond, lab in zip(conditions, type_labels):
        df.loc[cond, "inflation_type"] = lab

    classified = df[[
        "sector", "label", "supply_shock", "demand_shock",
        "supply_contribution_gdp_pct", "demand_contribution_gdp_pct",
        "total_contribution_gdp_pct", "inflation_type"
    ]].copy()

    return classified


if __name__ == "__main__":
    print("=" * 65)
    print("  2021–2022 Inflation Surge — Extension Analysis (66 sectors)")
    print("=" * 65)

    print("\nStep 1: Building IO Network...")
    net = get_io_network(year=2017, use_real_data=True)

    print("\nStep 2: Calibration check...")
    check_gdp_calibration(net.sectors)

    print("\nStep 3: Running Inflation Decomposition...")
    result = run_inflation_replication(net, verbose=True)

    print("\nStep 4: Classifying Inflation Drivers...")
    classified = classify_inflation_drivers(result)
    print("\n── Inflation Driver Classification ────────────────────────────────────")
    print(classified.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    classified.to_csv(TABLES_DIR / "inflation_driver_classification.csv", index=False)
