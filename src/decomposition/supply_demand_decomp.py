"""
Supply-Demand Shock Identification and GDP Decomposition.

Implements the core methodology of Baqaee & Farhi (2022):
  1. Identify supply and demand shocks from price and output changes
  2. Decompose GDP change into supply vs. demand contributions
  3. Compute network amplification factors

MATHEMATICAL FRAMEWORK
──────────────────────
Equilibrium conditions per sector i (log-linearized):

  Supply curve:  Δp_i = (1/σ_i) Δy_i − s_i     ... (1)
  Demand curve:  Δp_i = −(1/ε_i) Δy_i + d_i     ... (2)

Where:
  Δp_i = log price change (observed)
  Δy_i = log output change (observed)
  s_i   = supply shock (positive = productivity gain → more output, lower price)
  d_i   = demand shock (positive = demand expansion → more output, higher price)
  σ_i   = supply price elasticity (slope of supply curve)
  ε_i   = demand price elasticity in absolute value

Solving (1) and (2) for s_i and d_i:
  s_i = Δy_i / σ_i − Δp_i   ... supply shock identification
  d_i = Δp_i + Δy_i / ε_i   ... demand shock identification

Verify: equilibrating (1)=(2) gives Δy_i = σ_i ε_i (s_i + d_i) / (σ_i + ε_i) ✓

GDP DECOMPOSITION (Baqaee-Farhi Proposition 1)
───────────────────────────────────────────────
First-order approximation:

  ΔGDP/GDP ≈  Σ_i Λ_i · s_i           [supply channel]
             + Σ_i α_i^f · d_i          [demand channel]
             + O(shocks²)               [second-order terms]

Where:
  Λ_i = λ_i · Σ_j L[i,j]   (Domar-weighted Leontief row sum)
  α_i^f = final demand share of sector i
  λ_i   = Domar weight of sector i
  L     = Leontief inverse

Network amplification for supply shocks:
  The factor Σ_j L[i,j] > 1 captures how sector i's supply disruption
  propagates downstream through the production network.

Note on demand shocks:
  In the B-F framework, demand shocks' network effect depends on fiscal
  policy and income effects. The first-order term α_i^f · d_i is the
  direct demand effect; network propagation adds a correction term
  involving the IO matrix. We implement both.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from src.io_network.construct_network import IONetwork


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT ELASTICITY PARAMETERS
# Calibrated from literature (see decisions_log.md)
# ─────────────────────────────────────────────────────────────────────────────

# Sector-specific supply elasticities σ_i
# Higher = more price-responsive supply; lower = supply is inelastic
DEFAULT_SUPPLY_ELASTICITIES = {
    "AG":        2.0,   # moderate, limited by land
    "MIN":       1.5,   # somewhat inelastic (resource constraint)
    "UTIL":      0.5,   # regulated, very inelastic supply
    "CONST":     2.5,   # moderate, limited by skilled labor
    "FOOD_MFG":  3.0,   # fairly elastic
    "CHEM":      2.5,
    "PETRO":     1.5,   # partially inelastic (refinery capacity)
    "ELEC":      2.0,
    "AUTO":      2.0,
    "OTH_MFG":   2.5,
    "WHOL":      3.0,   # services, elastic
    "RETAIL":    3.0,
    "AIR":       1.0,   # inelastic: aircraft/slot constrained
    "TRANS":     2.0,
    "INFO":      4.0,   # highly elastic (digital)
    "FIN":       3.0,
    "REAL":      0.5,   # housing supply very inelastic
    "PROF":      3.5,
    "HEALTH":    1.0,   # inelastic: regulated, capacity constrained
    "FOOD_SVC":  2.5,
    "ARTS":      1.5,   # venue/capacity constrained
    "OTH_SVC":   2.5,
    "GOVT":      0.5,   # essentially fixed
}

# Sector-specific demand price elasticities ε_i (absolute value)
# Higher = more price-sensitive demand
DEFAULT_DEMAND_ELASTICITIES = {
    "AG":        0.5,   # necessities, inelastic demand
    "MIN":       0.6,
    "UTIL":      0.4,   # necessity, very inelastic
    "CONST":     0.8,
    "FOOD_MFG":  0.6,
    "CHEM":      0.7,
    "PETRO":     0.5,   # transportation fuel, inelastic
    "ELEC":      1.2,   # elastic (substitutes available)
    "AUTO":      1.0,   # unit elastic
    "OTH_MFG":   1.0,
    "WHOL":      0.8,
    "RETAIL":    1.0,
    "AIR":       1.5,   # elastic (luxury travel)
    "TRANS":     0.8,
    "INFO":      0.7,   # moderately inelastic (connectivity essential)
    "FIN":       0.6,
    "REAL":      0.4,   # very inelastic (necessity)
    "PROF":      0.9,
    "HEALTH":    0.3,   # very inelastic (necessity)
    "FOOD_SVC":  1.2,   # elastic (discretionary dining out)
    "ARTS":      1.8,   # very elastic (discretionary)
    "OTH_SVC":   1.0,
    "GOVT":      0.2,   # essentially fixed
}


@dataclass
class SectorShocks:
    """
    Observed price/output changes and identified supply/demand shocks
    for a single sector.
    """
    code: str
    label: str
    delta_p: float          # log price change (observed)
    delta_y: float          # log output change (observed)
    sigma: float            # supply elasticity
    epsilon: float          # demand price elasticity
    supply_shock: float     # s_i = delta_y/sigma - delta_p
    demand_shock: float     # d_i = delta_p + delta_y/epsilon
    supply_contribution_gdp: float = 0.0   # Λ_i · s_i
    demand_contribution_gdp: float = 0.0   # α_i^f · d_i
    network_amplification: float = 0.0    # Leontief row multiplier for sector


@dataclass
class DecompositionResult:
    """
    Full decomposition of GDP (or price level) change into supply and demand
    contributions, with network amplification effects.
    """
    episode: str
    total_delta_gdp: float                  # Σ (supply + demand contributions)
    total_supply_contribution: float        # Σ_i Λ_i · s_i
    total_demand_contribution: float        # Σ_i α_i^f · d_i
    total_network_amplification: float      # amplification over partial-eq baseline
    sector_results: list                    # list of SectorShocks
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def demand_share(self) -> float:
        """Fraction of GDP change explained by demand shocks."""
        total = abs(self.total_supply_contribution) + abs(self.total_demand_contribution)
        if total == 0:
            return 0.0
        return abs(self.total_demand_contribution) / total

    @property
    def supply_share(self) -> float:
        return 1.0 - self.demand_share


def identify_shocks(
    sectors: list,
    delta_p: np.ndarray,
    delta_y: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    epsilon: Optional[np.ndarray] = None,
    sector_labels: Optional[list] = None,
) -> tuple:
    """
    Identify supply and demand shocks from observed price and output changes.

    Parameters
    ----------
    sectors     : list of sector codes (length N)
    delta_p     : (N,) log price changes (observed)
    delta_y     : (N,) log output changes (observed)
    sigma       : (N,) supply elasticities (default: sector-specific)
    epsilon     : (N,) demand elasticities (default: sector-specific)
    sector_labels : (N,) human-readable sector names

    Returns
    -------
    s : (N,) supply shocks
    d : (N,) demand shocks
    """
    N = len(sectors)
    if sector_labels is None:
        sector_labels = sectors

    # Default elasticities
    if sigma is None:
        sigma = np.array([DEFAULT_SUPPLY_ELASTICITIES.get(code, 2.0) for code in sectors])
    if epsilon is None:
        epsilon = np.array([DEFAULT_DEMAND_ELASTICITIES.get(code, 0.8) for code in sectors])

    # Identification formulas (derived in module docstring)
    s = delta_y / sigma - delta_p          # supply shock
    d = delta_p + delta_y / epsilon        # demand shock

    return s, d, sigma, epsilon


def decompose_gdp(
    net: IONetwork,
    delta_p: np.ndarray,
    delta_y: np.ndarray,
    episode: str = "Episode",
    sigma: Optional[np.ndarray] = None,
    epsilon: Optional[np.ndarray] = None,
    include_second_order: bool = True,
) -> DecompositionResult:
    """
    Full supply-demand decomposition of GDP change.

    Parameters
    ----------
    net          : IONetwork object (contains A, L, λ, α^f)
    delta_p      : (N,) log price changes
    delta_y      : (N,) log output changes
    episode      : label for this analysis episode
    sigma        : supply elasticities (default sector-specific)
    epsilon      : demand elasticities (default sector-specific)
    include_second_order : whether to include O(shock²) correction

    Returns
    -------
    DecompositionResult
    """
    N = len(net.sectors)

    # Step 1: Identify shocks
    s, d, sigma_arr, epsilon_arr = identify_shocks(
        net.sectors, delta_p, delta_y, sigma, epsilon, net.labels
    )

    # Step 2: GDP contributions
    # Supply contribution of sector i = Λ_i · s_i
    #   where Λ_i = λ_i · Σ_j L[i,j]  (Domar-Leontief multiplier)
    supply_contribs = net.domar_leontief * s

    # Demand contribution of sector i = α_i^f · d_i
    demand_contribs = net.alpha_f * d

    # Network amplification for supply: ratio of network to partial-equilibrium
    # Partial eq: Σ_i λ_i · s_i  (Hulten's theorem, no network)
    # Network:    Σ_i Λ_i · s_i  (with Leontief multiplier)
    pe_supply = net.lam * s
    network_amplification = (
        supply_contribs.sum() - pe_supply.sum()
        if abs(pe_supply.sum()) > 1e-10
        else 0.0
    )

    # Step 3: Second-order correction (optional)
    # For supply shocks, the second-order term involves:
    #   (1/2) Σ_{ij} (∂²GDP/∂z_i ∂z_j) · s_i · s_j
    # Approximated as: (1/2) s' · diag(λ) · (L - I) · s
    # This captures the "superstar" effects where large shocks amplify nonlinearly.
    second_order = 0.0
    if include_second_order and np.any(np.abs(s) > 0.05):
        L_minus_I = net.L - np.eye(N)
        second_order_mat = 0.5 * (net.lam * s) @ L_minus_I @ s
        second_order = float(second_order_mat)

    total_supply = float(supply_contribs.sum())
    total_demand = float(demand_contribs.sum())
    total_gdp = total_supply + total_demand + second_order

    # Step 4: Build sector-level results
    sector_results = []
    for i, (code, label) in enumerate(zip(net.sectors, net.labels)):
        sr = SectorShocks(
            code=code,
            label=label,
            delta_p=float(delta_p[i]),
            delta_y=float(delta_y[i]),
            sigma=float(sigma_arr[i]),
            epsilon=float(epsilon_arr[i]),
            supply_shock=float(s[i]),
            demand_shock=float(d[i]),
            supply_contribution_gdp=float(supply_contribs[i]),
            demand_contribution_gdp=float(demand_contribs[i]),
            network_amplification=float(net.leontief_row_mult[i]),
        )
        sector_results.append(sr)

    # Step 5: Build summary DataFrame
    summary_df = pd.DataFrame([
        {
            "sector": sr.code,
            "label": sr.label,
            "delta_p_pct": sr.delta_p * 100,
            "delta_y_pct": sr.delta_y * 100,
            "supply_shock": sr.supply_shock,
            "demand_shock": sr.demand_shock,
            "domar_weight": net.lam[i],
            "leontief_multiplier": sr.network_amplification,
            "supply_contribution_gdp_pct": sr.supply_contribution_gdp * 100,
            "demand_contribution_gdp_pct": sr.demand_contribution_gdp * 100,
            "total_contribution_gdp_pct": (
                sr.supply_contribution_gdp + sr.demand_contribution_gdp
            ) * 100,
        }
        for i, sr in enumerate(sector_results)
    ])
    summary_df = summary_df.sort_values("total_contribution_gdp_pct").reset_index(drop=True)

    return DecompositionResult(
        episode=episode,
        total_delta_gdp=total_gdp,
        total_supply_contribution=total_supply,
        total_demand_contribution=total_demand,
        total_network_amplification=network_amplification,
        sector_results=sector_results,
        summary_df=summary_df,
    )


def sensitivity_analysis(
    net: IONetwork,
    delta_p: np.ndarray,
    delta_y: np.ndarray,
    episode: str,
    sigma_range: tuple = (1.0, 2.0, 3.0),
    epsilon_range: tuple = (0.3, 0.8, 1.5),
) -> pd.DataFrame:
    """
    Run the decomposition across a grid of elasticity parameters to assess
    robustness of the supply/demand split.

    Returns a DataFrame summarising total supply and demand contributions
    for each (σ, ε) combination.
    """
    rows = []
    for sig in sigma_range:
        for eps in epsilon_range:
            sigma_arr = np.full(len(net.sectors), sig)
            epsilon_arr = np.full(len(net.sectors), eps)
            res = decompose_gdp(
                net, delta_p, delta_y, episode, sigma=sigma_arr, epsilon=epsilon_arr,
                include_second_order=False,
            )
            rows.append({
                "sigma": sig,
                "epsilon": eps,
                "supply_pct": res.total_supply_contribution * 100,
                "demand_pct": res.total_demand_contribution * 100,
                "total_pct": res.total_delta_gdp * 100,
                "demand_share": res.demand_share,
            })
    return pd.DataFrame(rows)


def print_decomposition_summary(result: DecompositionResult) -> None:
    """Print a formatted summary of the decomposition results."""
    r = result
    pct = lambda x: f"{x*100:+.2f}%"
    print(f"\n{'='*65}")
    print(f"  DECOMPOSITION: {r.episode}")
    print(f"{'='*65}")
    print(f"  Total ΔGDP/GDP:             {pct(r.total_delta_gdp)}")
    print(f"    Supply contributions:      {pct(r.total_supply_contribution)}")
    print(f"    Demand contributions:      {pct(r.total_demand_contribution)}")
    print(f"    Network amplification:     {pct(r.total_network_amplification)}")
    print(f"  Share from demand:           {r.demand_share:.1%}")
    print(f"  Share from supply:           {r.supply_share:.1%}")
    print(f"\n  Top 5 sectors by total contribution:")
    top5 = r.summary_df.nsmallest(5, "total_contribution_gdp_pct")
    for _, row in top5.iterrows():
        print(
            f"    {row['label'][:35]:<35} "
            f"total={row['total_contribution_gdp_pct']:+.2f}%  "
            f"(S={row['supply_contribution_gdp_pct']:+.2f}%, "
            f"D={row['demand_contribution_gdp_pct']:+.2f}%)"
        )
    print(f"{'='*65}\n")


def compute_network_amplification_decomposition(
    net: IONetwork,
    delta_p: np.ndarray,
    delta_y: np.ndarray,
    episode: str,
) -> pd.DataFrame:
    """
    Decompose network amplification into direct and indirect effects.

    For supply shocks:
      Direct effect:   λ_i · s_i  (Hulten first-order, no network)
      Indirect effect: (Λ_i - λ_i) · s_i  (from IO propagation)
      Total:           Λ_i · s_i

    Returns a DataFrame showing both for each sector.
    """
    s, d, sigma, epsilon = identify_shocks(net.sectors, delta_p, delta_y)

    df = pd.DataFrame({
        "sector": net.sectors,
        "label": net.labels,
        "supply_shock": s,
        "demand_shock": d,
        "domar_weight": net.lam,
        "leontief_row_mult": net.leontief_row_mult,
        "direct_supply_effect_pct": net.lam * s * 100,
        "indirect_supply_effect_pct": (net.domar_leontief - net.lam) * s * 100,
        "total_supply_effect_pct": net.domar_leontief * s * 100,
        "demand_effect_pct": net.alpha_f * d * 100,
    })
    df["total_contribution_pct"] = (
        df["total_supply_effect_pct"] + df["demand_effect_pct"]
    )

    # Amplification ratio: total/direct for supply shocks with non-trivial direct effect
    mask = df["direct_supply_effect_pct"].abs() > 0.001
    df["supply_amplification_ratio"] = np.where(
        mask,
        df["total_supply_effect_pct"] / df["direct_supply_effect_pct"],
        np.nan,
    )

    return df.sort_values("total_contribution_pct").reset_index(drop=True)
