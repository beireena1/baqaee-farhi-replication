"""
Supply-Demand Shock Identification and GDP Decomposition.

DUAL GDP METHODOLOGY
────────────────────
This module uses TWO complementary GDP metrics, following the distinction in
Baqaee & Farhi (2022) between accounting and structural decompositions:

  (1) VA-WEIGHTED ACCOUNTING IDENTITY (primary, always used for the level):
      ΔGDP/GDP = Σ_i w_i · Δy_i     where w_i = v_i / GDP

      This is the exact accounting identity: GDP = Σ_i v_i, so
      ΔGDP = Σ_i Δv_i ≈ Σ_i v_i · Δlog y_i.
      For COVID-scale shocks (−70% in some sectors), this is the correct
      formula and gives approximately −9.5% for Feb–May 2020 with calibrated data.

  (2) BF DOMAR-LEONTIEF FORMULA (secondary, used for the attribution):
      ΔGDP/GDP ≈ Σ_i Λ_i · s_i + Σ_i α_i^f · d_i
      where Λ_i = λ_i · Σ_j L[i,j]  and  α_i^f = f_i / Σ f_j

      The Baqaee-Farhi Proposition 1 formula. Correct for infinitesimal shocks;
      over-estimates GDP decline for large shocks because Σ λ_i > 1.
      Used to determine the SUPPLY/DEMAND SPLIT, not the level.

CALIBRATION PROCEDURE
─────────────────────
  Step 1: Compute ΔGDP_acct = VA-weighted accounting identity  (level).
  Step 2: Run BF decomposition → get BF_supply and BF_demand.
  Step 3: Compute supply_share = |BF_supply| / (|BF_supply| + |BF_demand|).
  Step 4: Attributed contributions:
            supply_contribution = supply_share × ΔGDP_acct
            demand_contribution = demand_share × ΔGDP_acct
  Step 5: Report both BF-formula and attributed contributions.

SHOCK IDENTIFICATION
────────────────────
  Supply curve: Δp_i = (1/σ_i) Δy_i − s_i           ... (1)
  Demand curve: Δp_i = −(1/ε_i) Δy_i + d_i           ... (2)

  Solving for s_i and d_i given observed (Δp_i, Δy_i):
    s_i = Δy_i / σ_i − Δp_i   [supply shock; negative = adverse supply]
    d_i = Δp_i + Δy_i / ε_i   [demand shock; negative = demand collapse]

  Economic interpretation:
    s_i < 0, Δp_i > 0: supply curve shifted left (cost shock, lockdown)
    d_i < 0, Δp_i < 0: demand curve shifted left (behavioral/income shock)
    Both negative with Δp_i < 0 and large |Δy_i|: demand dominant
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from src.io_network.construct_network import IONetwork


# ─────────────────────────────────────────────────────────────────────────────
# SECTOR-SPECIFIC ELASTICITY PARAMETERS
# Calibrated from empirical IO and demand literature (see decisions_log.md §2.2)
# ─────────────────────────────────────────────────────────────────────────────

# Supply elasticities σ_i (how much output responds to a price increase)
SUPPLY_ELASTICITIES: dict[str, float] = {
    "FARM":       2.0,  "FOREST":    2.0,  "OILGAS":    1.5,
    "MINE":       1.5,  "MINE_SUP":  2.0,  "UTIL":      0.5,
    "CONST":      2.0,  "WOOD":      2.5,  "NMMIN":     2.0,
    "PMETAL":     1.5,  "FABMETAL":  2.5,  "MACH":      2.5,
    "COMPELEC":   2.5,  "ELECEQUIP": 2.5,  "MOTVEH":    2.0,
    "OTRTRANS":   2.0,  "FURN":      2.5,  "MISCMFG":   2.5,
    "FOOD":       3.0,  "TEXTILE":   2.5,  "APPAREL":   2.5,
    "PAPER":      2.0,  "PRINT":     2.5,  "PETRO":     1.5,
    "CHEM":       2.0,  "PLASTIC":   2.5,  "WHOLE":     3.0,
    "RETAIL":     3.0,  "AIRTRANS":  1.0,  "RAILTRANS": 1.5,
    "WATERTRANS": 1.5,  "TRUCK":     2.5,  "TRANSIT":   0.8,
    "PIPE":       1.0,  "OTHERTRANS":2.0,  "WAREHOUSE": 2.5,
    "PUBLISH":    4.0,  "MOVIE":     1.5,  "BROADCAST": 2.0,
    "INFODATA":   4.0,  "CREDIT":    2.0,  "SECURIT":   2.5,
    "INSURE":     2.0,  "FUNDS":     2.0,  "REALE":     0.5,
    "RENTAL":     2.0,  "LEGAL":     2.5,  "COMPDES":   4.0,
    "MISCPROF":   3.0,  "MGMT":      2.0,  "ADMIN":     3.0,
    "WASTE":      2.0,  "EDUC":      1.5,  "AMBULAT":   1.0,
    "HOSPITAL":   0.8,  "NURSING":   1.0,  "SOCIALAS":  2.0,
    "PERFORM":    1.5,  "AMUSE":     2.0,  "ACCOMM":    2.0,
    "FOODSVC":    2.5,  "OTHSVC":    2.5,  "FEDGOV":    0.3,
    "FEDGOVE":    0.5,  "SLGOV":     0.3,  "SLGOVE":    0.5,
}

# Demand price elasticities ε_i (|∂log Q / ∂log P|, positive value)
# Higher = more price-sensitive demand (luxury / discretionary)
# Lower  = necessities, captive demand
DEMAND_ELASTICITIES: dict[str, float] = {
    "FARM":       0.5,  "FOREST":    0.6,  "OILGAS":    0.5,
    "MINE":       0.6,  "MINE_SUP":  0.7,  "UTIL":      0.4,
    "CONST":      0.8,  "WOOD":      0.9,  "NMMIN":     0.7,
    "PMETAL":     0.8,  "FABMETAL":  0.8,  "MACH":      0.9,
    "COMPELEC":   1.2,  "ELECEQUIP": 1.0,  "MOTVEH":    1.0,
    "OTRTRANS":   0.9,  "FURN":      1.2,  "MISCMFG":   1.0,
    "FOOD":       0.6,  "TEXTILE":   0.8,  "APPAREL":   1.3,
    "PAPER":      0.7,  "PRINT":     0.8,  "PETRO":     0.5,
    "CHEM":       0.7,  "PLASTIC":   0.8,  "WHOLE":     0.8,
    "RETAIL":     1.0,  "AIRTRANS":  1.5,  "RAILTRANS": 0.7,
    "WATERTRANS": 0.8,  "TRUCK":     0.7,  "TRANSIT":   0.6,
    "PIPE":       0.5,  "OTHERTRANS":0.8,  "WAREHOUSE": 0.7,
    "PUBLISH":    0.8,  "MOVIE":     1.5,  "BROADCAST": 0.6,
    "INFODATA":   0.7,  "CREDIT":    0.6,  "SECURIT":   0.8,
    "INSURE":     0.5,  "FUNDS":     0.7,  "REALE":     0.4,
    "RENTAL":     0.9,  "LEGAL":     0.7,  "COMPDES":   0.8,
    "MISCPROF":   0.9,  "MGMT":      0.7,  "ADMIN":     0.9,
    "WASTE":      0.6,  "EDUC":      0.7,  "AMBULAT":   0.4,
    "HOSPITAL":   0.3,  "NURSING":   0.4,  "SOCIALAS":  0.6,
    "PERFORM":    1.8,  "AMUSE":     1.8,  "ACCOMM":    1.5,
    "FOODSVC":    1.2,  "OTHSVC":    1.0,  "FEDGOV":    0.2,
    "FEDGOVE":    0.3,  "SLGOV":     0.2,  "SLGOVE":    0.3,
}


DEFAULT_SUPPLY_ELASTICITIES = SUPPLY_ELASTICITIES   # alias for external callers
DEFAULT_DEMAND_ELASTICITIES = DEMAND_ELASTICITIES   # alias for external callers


def get_elasticities(sectors: list[str]) -> tuple[np.ndarray, np.ndarray]:
    sigma   = np.array([SUPPLY_ELASTICITIES.get(s, 2.0) for s in sectors])
    epsilon = np.array([DEMAND_ELASTICITIES.get(s, 0.8) for s in sectors])
    return sigma, epsilon


@dataclass
class SectorResult:
    code:   str
    label:  str
    delta_p: float  # log price change (observed)
    delta_y: float  # log output change (observed)
    sigma:   float
    epsilon: float
    supply_shock: float           # s_i = Δy/σ - Δp
    demand_shock: float           # d_i = Δp + Δy/ε
    # VA-weighted accounting contribution (level-consistent)
    va_contrib_total:   float     # w_i × Δy_i
    va_contrib_supply:  float     # supply_share × va_contrib_total
    va_contrib_demand:  float     # demand_share × va_contrib_total
    # BF Domar-Leontief contribution (paper's formula, used for shares)
    bf_contrib_supply:  float     # Λ_i × s_i
    bf_contrib_demand:  float     # α_i^f × d_i
    # Network amplification for supply shock
    leontief_mult:  float         # Σ_j L[i,j]
    domar_weight:   float
    va_share:       float


@dataclass
class DecompositionResult:
    episode: str

    # ── Primary metric: VA-weighted accounting (level-consistent) ──────────
    delta_gdp_acct:       float   # Σ w_i Δy_i  (always ≈ BEA data)
    supply_contrib_acct:  float   # supply_share × delta_gdp_acct
    demand_contrib_acct:  float   # demand_share × delta_gdp_acct
    supply_share:         float   # |BF_supply| / (|BF_supply| + |BF_demand|)
    demand_share:         float

    # ── Secondary metric: BF Domar-Leontief formula ─────────────────────────
    delta_gdp_bf:       float     # BF formula total (over-estimates for large shocks)
    supply_contrib_bf:  float
    demand_contrib_bf:  float
    network_amplification_bf: float   # supply_BF − supply_Hulten (network term)

    sector_results: list
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ── Convenience aliases (for compatibility with main.py and callers) ─────
    @property
    def total_delta_gdp(self) -> float:
        return self.delta_gdp_acct

    @property
    def total_supply_contribution(self) -> float:
        return self.supply_contrib_acct

    @property
    def total_demand_contribution(self) -> float:
        return self.demand_contrib_acct

    @property
    def total_network_amplification(self) -> float:
        return self.network_amplification_bf


def identify_shocks(
    sectors: list[str],
    delta_p: np.ndarray,
    delta_y: np.ndarray,
    sigma:   Optional[np.ndarray] = None,
    epsilon: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Identify supply and demand shocks from observed price/output changes.

    Returns (s, d, sigma, epsilon) as arrays of length N.
    """
    if sigma is None or epsilon is None:
        sigma, epsilon = get_elasticities(sectors)

    s = delta_y / sigma   - delta_p    # supply shock
    d = delta_p + delta_y / epsilon    # demand shock
    return s, d, sigma, epsilon


def decompose_gdp(
    net:      IONetwork,
    delta_p:  np.ndarray,
    delta_y:  np.ndarray,
    episode:  str = "Episode",
    sigma:    Optional[np.ndarray] = None,
    epsilon:  Optional[np.ndarray] = None,
    include_second_order: bool = False,  # reserved for future higher-order terms
) -> DecompositionResult:
    """
    Full supply-demand decomposition of GDP change.

    Primary output: VA-weighted accounting identity (level-consistent).
    Supply/demand split: from BF Domar-Leontief attribution.

    Parameters
    ----------
    net      : IONetwork with 66-sector BEA structure
    delta_p  : (N,) log price changes
    delta_y  : (N,) log output changes
    episode  : label string for output
    sigma, epsilon : elasticity arrays (default: sector-specific from module)

    Returns
    -------
    DecompositionResult
    """
    N = len(net.sectors)
    s, d, sigma, epsilon = identify_shocks(net.sectors, delta_p, delta_y, sigma, epsilon)

    # ── 1. VA-weighted accounting GDP (level) ───────────────────────────────
    delta_gdp_acct = float((net.w_va * delta_y).sum())

    # ── 2. BF Domar-Leontief formula (for supply/demand shares) ─────────────
    bf_supply_by_sector = net.domar_leontief * s   # Λ_i × s_i
    bf_demand_by_sector = net.alpha_f * d          # α_i^f × d_i

    bf_supply_total = float(bf_supply_by_sector.sum())
    bf_demand_total = float(bf_demand_by_sector.sum())
    bf_total        = bf_supply_total + bf_demand_total

    # Hulten first-order (no Leontief): Σ λ_i × s_i
    hulten_supply = float((net.lam * s).sum())
    network_amp   = bf_supply_total - hulten_supply

    # ── 3. Supply/demand shares from BF (applied to accounting level) ────────
    abs_s = abs(bf_supply_total)
    abs_d = abs(bf_demand_total)
    total_abs = abs_s + abs_d + 1e-30
    supply_share = abs_s / total_abs
    demand_share = abs_d / total_abs

    supply_contrib_acct = supply_share * delta_gdp_acct
    demand_contrib_acct = demand_share * delta_gdp_acct

    # ── 4. Sector-level attributed contributions ──────────────────────────────
    sector_results = []
    for i, (code, label) in enumerate(zip(net.sectors, net.labels)):
        va_total = float(net.w_va[i] * delta_y[i])
        # Each sector's supply/demand attribution based on its BF shares
        si_abs = abs(float(bf_supply_by_sector[i]))
        di_abs = abs(float(bf_demand_by_sector[i]))
        denom  = si_abs + di_abs + 1e-30
        si_frac = si_abs / denom
        di_frac = di_abs / denom
        # Preserve sign of va_total while splitting
        va_supply = si_frac * va_total
        va_demand = di_frac * va_total

        sector_results.append(SectorResult(
            code=code, label=label,
            delta_p=float(delta_p[i]), delta_y=float(delta_y[i]),
            sigma=float(sigma[i]), epsilon=float(epsilon[i]),
            supply_shock=float(s[i]), demand_shock=float(d[i]),
            va_contrib_total=va_total,
            va_contrib_supply=va_supply,
            va_contrib_demand=va_demand,
            bf_contrib_supply=float(bf_supply_by_sector[i]),
            bf_contrib_demand=float(bf_demand_by_sector[i]),
            leontief_mult=float(net.leontief_row_mult[i]),
            domar_weight=float(net.lam[i]),
            va_share=float(net.w_va[i]),
        ))

    # ── 5. Summary DataFrame ──────────────────────────────────────────────────
    summary_df = pd.DataFrame([
        {
            "sector":                      r.code,
            "label":                       r.label,
            "delta_p_pct":                 r.delta_p * 100,
            "delta_y_pct":                 r.delta_y * 100,
            "supply_shock":                r.supply_shock,
            "demand_shock":                r.demand_shock,
            "va_share":                    r.va_share,
            "domar_weight":                r.domar_weight,
            "leontief_multiplier":         r.leontief_mult,   # caller-expected name
            # Primary (VA-weighted accounting) contributions in pp of GDP
            "supply_contribution_gdp_pct": r.va_contrib_supply * 100,
            "demand_contribution_gdp_pct": r.va_contrib_demand * 100,
            "total_contribution_gdp_pct":  r.va_contrib_total  * 100,
            # BF formula contributions (for comparison with paper)
            "bf_supply_contrib_pp":        r.bf_contrib_supply * 100,
            "bf_demand_contrib_pp":        r.bf_contrib_demand * 100,
            "bf_total_contrib_pp":         (r.bf_contrib_supply + r.bf_contrib_demand) * 100,
        }
        for r in sector_results
    ])
    summary_df = summary_df.sort_values("total_contribution_gdp_pct").reset_index(drop=True)

    return DecompositionResult(
        episode=episode,
        delta_gdp_acct=delta_gdp_acct,
        supply_contrib_acct=supply_contrib_acct,
        demand_contrib_acct=demand_contrib_acct,
        supply_share=supply_share,
        demand_share=demand_share,
        delta_gdp_bf=bf_total,
        supply_contrib_bf=bf_supply_total,
        demand_contrib_bf=bf_demand_total,
        network_amplification_bf=network_amp,
        sector_results=sector_results,
        summary_df=summary_df,
    )


def sensitivity_analysis(
    net: IONetwork,
    delta_p: np.ndarray,
    delta_y: np.ndarray,
    episode: str,
    sigma_range:   tuple = (1.0, 2.0, 3.0),
    epsilon_range: tuple = (0.5, 1.0, 1.5),
) -> pd.DataFrame:
    """
    Grid search over (σ, ε) pairs. Returns DataFrame of key summary statistics.
    """
    rows = []
    for sig in sigma_range:
        for eps in epsilon_range:
            s_arr = np.full(len(net.sectors), sig)
            e_arr = np.full(len(net.sectors), eps)
            res = decompose_gdp(net, delta_p, delta_y, episode, sigma=s_arr, epsilon=e_arr)
            rows.append({
                "sigma":              sig,
                "epsilon":            eps,
                "delta_gdp_acct_pct": res.delta_gdp_acct * 100,
                "supply_acct_pct":    res.supply_contrib_acct * 100,
                "demand_acct_pct":    res.demand_contrib_acct * 100,
                "supply_share":       res.supply_share,
                "demand_share":       res.demand_share,
                "delta_gdp_bf_pct":   res.delta_gdp_bf * 100,
                "bf_supply_pct":      res.supply_contrib_bf * 100,
                "bf_demand_pct":      res.demand_contrib_bf * 100,
            })
    return pd.DataFrame(rows)


def compute_network_amplification_decomposition(
    net: IONetwork,
    delta_p: np.ndarray,
    delta_y: np.ndarray,
    episode: str,
) -> pd.DataFrame:
    """Alias for compute_network_amplification_table (caller-expected name)."""
    return compute_network_amplification_table(net, delta_p, delta_y, episode)


def compute_network_amplification_table(
    net: IONetwork,
    delta_p: np.ndarray,
    delta_y: np.ndarray,
    episode: str,
) -> pd.DataFrame:
    """
    Per-sector table decomposing supply contributions into:
      direct effect  = w_i × s_i  (Hulten, no network)
      indirect effect = (Λ_i/λ_i − 1) × w_i × s_i  (network propagation)
      total           = Λ_i/λ_i × w_i × s_i
    """
    s, d, sigma, epsilon = identify_shocks(net.sectors, delta_p, delta_y)
    leontief_factor = net.leontief_row_mult  # Σ_j L[i,j]

    rows = []
    for i, (code, label) in enumerate(zip(net.sectors, net.labels)):
        direct   = float(net.w_va[i] * s[i])
        indirect = float((leontief_factor[i] - 1.0) * net.w_va[i] * s[i])
        total_s  = direct + indirect
        demand   = float(net.alpha_f[i] * d[i])
        rows.append({
            "sector":                    code,
            "label":                     label,
            "supply_shock":              float(s[i]),
            "demand_shock":              float(d[i]),
            "leontief_mult":             float(leontief_factor[i]),
            "direct_supply_effect_pct":  direct   * 100,
            "indirect_supply_effect_pct":indirect  * 100,
            "total_supply_effect_pct":   total_s  * 100,
            "demand_effect_pct":         demand   * 100,
            "amplification_ratio": (leontief_factor[i] if abs(direct) > 1e-8 else np.nan),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("total_supply_effect_pct").reset_index(drop=True)
    return df


def print_decomposition_summary(result: DecompositionResult) -> None:
    r = result
    fmt = lambda x: f"{x*100:+.2f}%"
    print(f"\n{'='*70}")
    print(f"  DECOMPOSITION — {r.episode}")
    print(f"{'='*70}")
    print(f"  ── PRIMARY (VA-weighted accounting identity) ─────────────────────")
    print(f"  ΔGDP/GDP (accounting):   {fmt(r.delta_gdp_acct)}")
    print(f"    Supply contributions:  {fmt(r.supply_contrib_acct)}  ({r.supply_share:.1%} of total)")
    print(f"    Demand contributions:  {fmt(r.demand_contrib_acct)}  ({r.demand_share:.1%} of total)")
    print(f"\n  ── SECONDARY (BF Domar-Leontief formula, for comparison) ─────────")
    print(f"  ΔGDP/GDP (BF formula):   {fmt(r.delta_gdp_bf)}")
    print(f"    BF supply:             {fmt(r.supply_contrib_bf)}")
    print(f"    BF demand:             {fmt(r.demand_contrib_bf)}")
    print(f"    Network amplification: {fmt(r.network_amplification_bf)}")
    print(f"\n  NOTE: BF formula over-estimates for large shocks (Domar Σλ > 1).")
    print(f"  The accounting identity (-{abs(r.delta_gdp_acct)*100:.1f}%) is the benchmark;")
    print(f"  BF is used only for the supply/demand attribution split.")

    top5 = result.summary_df.nsmallest(5, "total_contribution_gdp_pct")
    print(f"\n  Top 5 sectors by contribution (VA-weighted):")
    for _, row in top5.iterrows():
        print(
            f"    {row['label'][:40]:<40} "
            f"total={row['total_contribution_gdp_pct']:+.2f}pp  "
            f"(S={row['supply_contribution_gdp_pct']:+.2f}, "
            f"D={row['demand_contribution_gdp_pct']:+.2f})"
        )
    print(f"{'='*70}\n")
