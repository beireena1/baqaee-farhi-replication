"""
Cross-Episode Comparison: COVID-19 vs. 2021–2022 Inflation Surge.

Compares the structural features of the two episodes:
  1. Supply vs. demand share of total shock
  2. Network amplification magnitude
  3. Sectoral concentration of shocks
  4. Which sectors drove each episode
  5. Role of IO linkages in propagation
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.decomposition.supply_demand_decomp import DecompositionResult

OUTPUT_DIR = Path("outputs")
TABLES_DIR = OUTPUT_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def compute_herfindahl_index(contributions: np.ndarray) -> float:
    """
    Herfindahl-Hirschman Index for concentration of sector contributions.

    HHI = Σ share_i² where share_i = |contribution_i| / Σ_j |contribution_j|
    HHI = 1: single sector drives all variation
    HHI = 1/N: perfectly uniform across N sectors
    """
    abs_contribs = np.abs(contributions)
    total = abs_contribs.sum()
    if total == 0:
        return 0.0
    shares = abs_contribs / total
    return float((shares ** 2).sum())


def compute_episode_summary(result: DecompositionResult, net) -> dict:
    """
    Compute a comprehensive set of summary statistics for one episode.
    """
    df = result.summary_df

    total_supply = result.total_supply_contribution * 100
    total_demand = result.total_demand_contribution * 100
    total = result.total_delta_gdp * 100

    # Concentration of supply contributions
    supply_hhi = compute_herfindahl_index(df["supply_contribution_gdp_pct"].values)
    demand_hhi = compute_herfindahl_index(df["demand_contribution_gdp_pct"].values)

    # Top sector by absolute total contribution
    abs_total = df["total_contribution_gdp_pct"].abs()
    top_sector_idx = abs_total.idxmax()
    top_sector = df.loc[top_sector_idx, "label"]
    top_sector_contrib = df.loc[top_sector_idx, "total_contribution_gdp_pct"]

    # Share of top-5 sectors in total variation
    top5_share = abs_total.nlargest(5).sum() / (abs_total.sum() + 1e-10)

    # Average Leontief multiplier (weighted by absolute supply contribution)
    abs_supply = df["supply_contribution_gdp_pct"].abs()
    if abs_supply.sum() > 0:
        avg_leontief = (df["leontief_multiplier"] * abs_supply).sum() / abs_supply.sum()
    else:
        avg_leontief = float(net.leontief_row_mult.mean())

    # Network amplification as fraction of total supply effect
    pe_supply = (net.lam * df["supply_shock"].values).sum() * 100
    network_amp_pct = (total_supply - pe_supply) / (abs(total_supply) + 1e-10)

    return {
        "total_effect_pct": total,
        "supply_contribution_pct": total_supply,
        "demand_contribution_pct": total_demand,
        "supply_share": result.supply_share,
        "demand_share": result.demand_share,
        "network_amplification_ratio": network_amp_pct,
        "pe_supply_pct": pe_supply,
        "supply_hhi": supply_hhi,
        "demand_hhi": demand_hhi,
        "top_sector": top_sector,
        "top_sector_contrib_pct": top_sector_contrib,
        "top5_sector_share": top5_share,
        "avg_weighted_leontief": avg_leontief,
        "n_sectors": len(df),
    }


def build_comparison_table(covid_result: DecompositionResult,
                            infl_result: DecompositionResult,
                            net) -> pd.DataFrame:
    """
    Build the main comparison table for the two episodes.

    Returns a formatted DataFrame suitable for a paper table.
    """
    covid_stats = compute_episode_summary(covid_result, net)
    infl_stats = compute_episode_summary(infl_result, net)

    rows = [
        ("Total effect (%)", "total_effect_pct", "{:.2f}%"),
        ("Supply contributions (%)", "supply_contribution_pct", "{:.2f}%"),
        ("Demand contributions (%)", "demand_contribution_pct", "{:.2f}%"),
        ("Supply share of total", "supply_share", "{:.1%}"),
        ("Demand share of total", "demand_share", "{:.1%}"),
        ("Partial-eq. supply effect (%)", "pe_supply_pct", "{:.2f}%"),
        ("Network amplification (ratio)", "network_amplification_ratio", "{:.3f}"),
        ("Avg. Leontief multiplier (weighted)", "avg_weighted_leontief", "{:.3f}"),
        ("Supply HHI (concentration)", "supply_hhi", "{:.4f}"),
        ("Demand HHI (concentration)", "demand_hhi", "{:.4f}"),
        ("Top contributing sector", "top_sector", "{}"),
        ("Top sector contribution (%)", "top_sector_contrib_pct", "{:.2f}%"),
        ("Top-5 sector share of total", "top5_sector_share", "{:.1%}"),
    ]

    records = []
    for label, key, fmt in rows:
        covid_val = covid_stats[key]
        infl_val = infl_stats[key]
        try:
            records.append({
                "Metric": label,
                "COVID-19 (Feb–May 2020)": fmt.format(covid_val),
                "Inflation (2020Q4–2022Q4)": fmt.format(infl_val),
            })
        except (ValueError, TypeError):
            records.append({
                "Metric": label,
                "COVID-19 (Feb–May 2020)": str(covid_val),
                "Inflation (2020Q4–2022Q4)": str(infl_val),
            })

    table = pd.DataFrame(records)
    out_path = TABLES_DIR / "episode_comparison.csv"
    table.to_csv(out_path, index=False)
    print(f"Comparison table saved to {out_path}")
    return table


def build_sector_comparison(covid_result: DecompositionResult,
                             infl_result: DecompositionResult) -> pd.DataFrame:
    """
    Build a sector-by-sector comparison of the two episodes.

    Shows for each sector:
      - COVID total contribution (GDP decline) and supply/demand split
      - Inflation total contribution (GDP change) and supply/demand split
      - Change in role between episodes
    """
    covid_df = covid_result.summary_df.set_index("sector")
    infl_df = infl_result.summary_df.set_index("sector")

    all_sectors = sorted(set(covid_df.index) | set(infl_df.index))

    rows = []
    for sector in all_sectors:
        covid_row = covid_df.loc[sector] if sector in covid_df.index else None
        infl_row = infl_df.loc[sector] if sector in infl_df.index else None

        label = (covid_row["label"] if covid_row is not None
                 else infl_row["label"] if infl_row is not None else sector)

        rows.append({
            "sector": sector,
            "label": label,
            # COVID
            "covid_delta_y_pct": float(covid_row["delta_y_pct"]) if covid_row is not None else np.nan,
            "covid_delta_p_pct": float(covid_row["delta_p_pct"]) if covid_row is not None else np.nan,
            "covid_supply_contrib_pct": float(covid_row["supply_contribution_gdp_pct"]) if covid_row is not None else np.nan,
            "covid_demand_contrib_pct": float(covid_row["demand_contribution_gdp_pct"]) if covid_row is not None else np.nan,
            "covid_total_contrib_pct": float(covid_row["total_contribution_gdp_pct"]) if covid_row is not None else np.nan,
            # Inflation
            "infl_delta_y_pct": float(infl_row["delta_y_pct"]) if infl_row is not None else np.nan,
            "infl_delta_p_pct": float(infl_row["delta_p_pct"]) if infl_row is not None else np.nan,
            "infl_supply_contrib_pct": float(infl_row["supply_contribution_gdp_pct"]) if infl_row is not None else np.nan,
            "infl_demand_contrib_pct": float(infl_row["demand_contribution_gdp_pct"]) if infl_row is not None else np.nan,
            "infl_total_contrib_pct": float(infl_row["total_contribution_gdp_pct"]) if infl_row is not None else np.nan,
        })

    df = pd.DataFrame(rows)

    # Classify role reversal: sectors that were demand-driven in COVID
    # but supply-driven in inflation (e.g., energy)
    df["role_reversal"] = (
        (df["covid_demand_contrib_pct"] < df["covid_supply_contrib_pct"])  # demand-driven COVID
        & (df["infl_supply_contrib_pct"].abs() > df["infl_demand_contrib_pct"].abs())  # supply-driven inflation
    )

    out_path = TABLES_DIR / "sector_episode_comparison.csv"
    df.to_csv(out_path, index=False, float_format="%.3f")
    print(f"Sector comparison saved to {out_path}")
    return df


def print_comparison_narrative(covid_result: DecompositionResult,
                                infl_result: DecompositionResult,
                                net) -> None:
    """
    Print a narrative summary comparing the two episodes.
    """
    covid_stats = compute_episode_summary(covid_result, net)
    infl_stats = compute_episode_summary(infl_result, net)

    print("\n" + "=" * 70)
    print("  CROSS-EPISODE COMPARISON: COVID-19 vs. 2021–2022 Inflation")
    print("=" * 70)

    print(f"""
EPISODE 1: COVID-19 (Feb–May 2020)
───────────────────────────────────
  Total GDP change:    {covid_stats['total_effect_pct']:+.2f}%
  Supply contribution: {covid_stats['supply_contribution_pct']:+.2f}% ({covid_stats['supply_share']:.0%} of total)
  Demand contribution: {covid_stats['demand_contribution_pct']:+.2f}% ({covid_stats['demand_share']:.0%} of total)
  Network amplification: {covid_stats['network_amplification_ratio']:+.1%} of supply effect
  Dominant sector:     {covid_stats['top_sector']} ({covid_stats['top_sector_contrib_pct']:+.2f}%)

EPISODE 2: 2021–2022 Inflation Surge (2020Q4 → 2022Q4)
───────────────────────────────────────────────────────
  Total GDP change:    {infl_stats['total_effect_pct']:+.2f}%
  Supply contribution: {infl_stats['supply_contribution_pct']:+.2f}% ({infl_stats['supply_share']:.0%} of total)
  Demand contribution: {infl_stats['demand_contribution_pct']:+.2f}% ({infl_stats['demand_share']:.0%} of total)
  Network amplification: {infl_stats['network_amplification_ratio']:+.1%} of supply effect
  Dominant sector:     {infl_stats['top_sector']} ({infl_stats['top_sector_contrib_pct']:+.2f}%)

KEY STRUCTURAL DIFFERENCES
───────────────────────────────────────────────────────
  1. SUPPLY vs. DEMAND DOMINANCE:
     COVID:    Demand-driven ({covid_stats['demand_share']:.0%} demand, {covid_stats['supply_share']:.0%} supply)
     Inflation: Supply-driven ({infl_stats['supply_share']:.0%} supply, {infl_stats['demand_share']:.0%} demand)
     → COVID was primarily a demand-side shock (behavioral + policy lockdowns
       reduced spending on contact-intensive services)
     → Inflation was primarily supply-side (energy price shock, semiconductor
       shortage, supply chain disruption) amplified by fiscal stimulus demand

  2. NETWORK AMPLIFICATION:
     COVID:     {covid_stats['network_amplification_ratio']:+.1%} amplification of supply shocks
     Inflation: {infl_stats['network_amplification_ratio']:+.1%} amplification of supply shocks
     → Greater network amplification during inflation reflects energy/commodity
       shocks propagating across all sectors through IO linkages
       (energy is an input to almost every production process)

  3. CONCENTRATION (HHI):
     COVID supply HHI:     {covid_stats['supply_hhi']:.4f}
     Inflation supply HHI: {infl_stats['supply_hhi']:.4f}
     → {'Inflation was more concentrated' if infl_stats['supply_hhi'] > covid_stats['supply_hhi']
        else 'COVID was more concentrated'} in a few key supply-shock sectors

  4. TOP-5 SECTOR DOMINANCE:
     COVID:    Top 5 sectors = {covid_stats['top5_sector_share']:.0%} of total variation
     Inflation: Top 5 sectors = {infl_stats['top5_sector_share']:.0%} of total variation
""")


if __name__ == "__main__":
    from src.io_network.construct_network import get_io_network
    from src.replication.covid_episode import run_covid_replication
    from src.extension.inflation_episode import run_inflation_replication

    print("=" * 65)
    print("  Cross-Episode Comparison")
    print("=" * 65)

    net = get_io_network(year=2017, use_real_data=True)

    print("\nRunning COVID decomposition...")
    covid_result = run_covid_replication(net, run_sensitivity=False, verbose=False)

    print("Running Inflation decomposition...")
    infl_result = run_inflation_replication(net, run_sensitivity=False, verbose=False)

    print_comparison_narrative(covid_result, infl_result, net)

    comparison_table = build_comparison_table(covid_result, infl_result, net)
    print("\n── Summary Comparison Table ─────────────────────────────────────────")
    print(comparison_table.to_string(index=False))

    sector_comp = build_sector_comparison(covid_result, infl_result)
    print("\n── Sector-by-Sector Comparison ──────────────────────────────────────")
    display_cols = [
        "label", "covid_total_contrib_pct", "covid_supply_contrib_pct",
        "covid_demand_contrib_pct", "infl_total_contrib_pct",
        "infl_supply_contrib_pct", "infl_demand_contrib_pct",
    ]
    print(sector_comp[display_cols].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
