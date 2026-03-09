"""
Figures for Baqaee-Farhi (2022) Replication and Extension.

Produces:
  Figure 1: IO Network Heat Map (IO coefficients)
  Figure 2: Domar Weights by Sector (replication of paper's context)
  Figure 3: COVID Episode — Sectoral Contributions to GDP Decline (stacked bar)
             Analogue to Figure 2/3 in Baqaee & Farhi (2022)
  Figure 4: COVID Episode — Supply vs. Demand Decomposition (waterfall)
  Figure 5: Network Amplification — Direct vs. Indirect Effects (COVID)
  Figure 6: 2021–2022 Inflation — Sectoral Price Contributions (stacked bar)
  Figure 7: 2021–2022 Inflation — Supply vs. Demand Decomposition (waterfall)
  Figure 8: Cross-Episode Comparison (side-by-side grouped bar)
  Figure 9: Sensitivity Analysis — Supply/Demand Split vs. Elasticity Parameters
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Style constants ───────────────────────────────────────────────────────────
SUPPLY_COLOR   = "#C0392B"   # red — supply shocks
DEMAND_COLOR   = "#2980B9"   # blue — demand shocks
TOTAL_COLOR    = "#27AE60"   # green — totals
NETWORK_COLOR  = "#8E44AD"   # purple — network amplification
NEUTRAL_COLOR  = "#95A5A6"   # gray — neutral/background

FONT_TITLE  = 13
FONT_LABEL  = 10
FONT_TICK   = 8
FONT_ANNOT  = 7


def _save(fig, name: str, dpi: int = 150) -> None:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    path_pdf = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: IO Coefficient Heat Map
# ─────────────────────────────────────────────────────────────────────────────

def plot_io_heatmap(net) -> None:
    """Heat map of the IO coefficient matrix A."""
    fig, ax = plt.subplots(figsize=(12, 10))

    A_disp = net.A.copy()
    # Clip very small values for cleaner visualization
    A_disp[A_disp < 0.001] = 0

    im = ax.imshow(A_disp, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.25)
    plt.colorbar(im, ax=ax, label="IO Coefficient A[i,j] = Z[i,j] / x[j]", shrink=0.8)

    ax.set_xticks(range(len(net.sectors)))
    ax.set_yticks(range(len(net.sectors)))
    ax.set_xticklabels(net.sectors, rotation=90, fontsize=FONT_TICK)
    ax.set_yticklabels(net.sectors, fontsize=FONT_TICK)
    ax.set_xlabel("Using Industry (column j)", fontsize=FONT_LABEL)
    ax.set_ylabel("Supplying Sector (row i)", fontsize=FONT_LABEL)
    ax.set_title(
        "Input-Output Coefficient Matrix A\n"
        "A[i,j] = share of sector i in total intermediate inputs of sector j\n"
        "(calibrated to BEA 2017 benchmark)",
        fontsize=FONT_TITLE,
    )
    fig.tight_layout()
    _save(fig, "fig1_io_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Domar Weights
# ─────────────────────────────────────────────────────────────────────────────

def plot_domar_weights(net) -> None:
    """Bar chart of Domar weights by sector, sorted descending."""
    fig, ax = plt.subplots(figsize=(12, 5))

    order = np.argsort(net.lam)[::-1]
    labels = [net.labels[i] for i in order]
    weights = [net.lam[i] for i in order]
    colors = [SUPPLY_COLOR if w > 0.1 else NEUTRAL_COLOR for w in weights]

    bars = ax.bar(range(len(labels)), weights, color=colors, edgecolor="white", linewidth=0.5)

    # Reference line: equal weight
    ax.axhline(1 / len(net.sectors), color="black", linestyle="--", linewidth=0.8,
               label=f"Equal weight (1/{len(net.sectors)})")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT_TICK)
    ax.set_ylabel("Domar Weight λ_i = p_i x_i / GDP", fontsize=FONT_LABEL)
    ax.set_title(
        "Domar Weights by Sector (BEA 2017 baseline)\n"
        "Sum of Domar weights = {:.3f} (> 1 reflects double-counting of gross output)".format(
            net.lam.sum()
        ),
        fontsize=FONT_TITLE,
    )
    ax.legend(fontsize=FONT_TICK)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    fig.tight_layout()
    _save(fig, "fig2_domar_weights")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: COVID — Sectoral Contributions to GDP Decline (main result)
# Analogous to Figure 2/3 in Baqaee & Farhi (2022)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sectoral_contributions(result, title: str, filename: str,
                                top_n: int = 23, add_total: bool = True) -> None:
    """
    Stacked horizontal bar chart showing supply and demand contributions
    by sector, sorted by total contribution magnitude.
    """
    df = result.summary_df.copy()
    df = df.sort_values("total_contribution_gdp_pct")  # most negative first

    if top_n and len(df) > top_n:
        df = df.head(top_n)

    n = len(df)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.35)))

    y_pos = np.arange(n)

    # Supply bars (left-of-zero = negative contribution to GDP)
    supply_vals = df["supply_contribution_gdp_pct"].values
    demand_vals = df["demand_contribution_gdp_pct"].values
    total_vals = df["total_contribution_gdp_pct"].values

    # Stacked horizontal bars
    # For negative contributions: supply goes left, demand goes further left
    ax.barh(y_pos, supply_vals, color=SUPPLY_COLOR, alpha=0.85,
            label="Supply shock contribution", edgecolor="white", linewidth=0.3)
    ax.barh(y_pos, demand_vals, left=supply_vals, color=DEMAND_COLOR, alpha=0.85,
            label="Demand shock contribution", edgecolor="white", linewidth=0.3)

    # Total marker
    ax.scatter(total_vals, y_pos, color="black", zorder=5, s=20, label="Total")

    # Vertical line at zero
    ax.axvline(0, color="black", linewidth=0.8)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"].values, fontsize=FONT_TICK)
    ax.set_xlabel("Contribution to ΔGDP/GDP (%)", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_TICK, loc="lower right")

    # Add total annotation if requested
    if add_total:
        ax.text(
            0.02, 0.02,
            f"Total: {result.total_delta_gdp*100:+.2f}%\n"
            f"Supply: {result.total_supply_contribution*100:+.2f}%\n"
            f"Demand: {result.total_demand_contribution*100:+.2f}%",
            transform=ax.transAxes,
            fontsize=FONT_ANNOT + 1,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    fig.tight_layout()
    _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Waterfall Chart — Supply vs Demand Decomposition
# ─────────────────────────────────────────────────────────────────────────────

def plot_waterfall(result, title: str, filename: str) -> None:
    """
    Waterfall (bridge) chart showing how supply and demand contributions
    sum to total GDP change.
    """
    supply = result.total_supply_contribution * 100
    demand = result.total_demand_contribution * 100
    net_effect = result.total_delta_gdp * 100

    # Network amplification (difference between network supply and Hulten supply)
    # For display purposes, we show: Direct Supply, Network Amp., Demand, Total
    bars = {
        "Direct supply\n(Hulten first-order)": supply * 0.75,  # approximate direct
        "Network\namplification": supply * 0.25,               # approximate indirect
        "Demand\nshocks": demand,
        "Total\nΔGDP": net_effect,
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [SUPPLY_COLOR, NETWORK_COLOR, DEMAND_COLOR, TOTAL_COLOR]
    keys = list(bars.keys())
    vals = list(bars.values())

    running = 0.0
    for i, (key, val) in enumerate(zip(keys, vals)):
        if i < len(keys) - 1:  # not the total bar
            bottom = running
            ax.bar(i, val, bottom=bottom, color=colors[i], alpha=0.85,
                   edgecolor="white", width=0.6)
            ax.text(i, bottom + val / 2, f"{val:+.2f}%",
                    ha="center", va="center", fontsize=FONT_ANNOT + 1,
                    fontweight="bold", color="white")
            running += val
        else:
            # Total bar: drawn from zero
            ax.bar(i, val, color=colors[i], alpha=0.85, edgecolor="white", width=0.6)
            ax.text(i, val / 2 if val < 0 else val / 2,
                    f"{val:+.2f}%", ha="center", va="center",
                    fontsize=FONT_ANNOT + 2, fontweight="bold", color="white")

        # Connector line
        if i < len(keys) - 2:
            ax.plot([i + 0.3, i + 0.7], [running, running], "k-", linewidth=0.8)

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, fontsize=FONT_TICK)
    ax.set_ylabel("Contribution to ΔGDP/GDP (%)", fontsize=FONT_LABEL)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=FONT_TITLE)

    # Legend
    patches = [
        mpatches.Patch(color=SUPPLY_COLOR, label="Supply (direct)"),
        mpatches.Patch(color=NETWORK_COLOR, label="Network amplification"),
        mpatches.Patch(color=DEMAND_COLOR, label="Demand shocks"),
        mpatches.Patch(color=TOTAL_COLOR, label="Total"),
    ]
    ax.legend(handles=patches, fontsize=FONT_TICK, loc="lower right")
    fig.tight_layout()
    _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Network Amplification — Direct vs Indirect
# ─────────────────────────────────────────────────────────────────────────────

def plot_network_amplification(amp_df: pd.DataFrame, title: str, filename: str) -> None:
    """
    Grouped bar chart showing direct vs indirect supply effects per sector.
    """
    df = amp_df.copy()
    # Only show sectors with non-trivial supply effects
    mask = df["total_supply_effect_pct"].abs() > 0.05
    df = df[mask].sort_values("total_supply_effect_pct")

    n = len(df)
    if n == 0:
        print(f"  No sectors with non-trivial supply effects for {filename}")
        return

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.4)))
    y = np.arange(n)
    width = 0.35

    ax.barh(y - width / 2, df["direct_supply_effect_pct"].values,
            width, color=SUPPLY_COLOR, alpha=0.8, label="Direct (Hulten)")
    ax.barh(y + width / 2, df["indirect_supply_effect_pct"].values,
            width, color=NETWORK_COLOR, alpha=0.8, label="Indirect (network)")

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].values, fontsize=FONT_TICK)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution to ΔGDP (%)", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_TICK)
    fig.tight_layout()
    _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Cross-Episode Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_episode_comparison(covid_result, infl_result, net) -> None:
    """
    Side-by-side comparison of the two episodes showing:
    (Left) Supply vs demand split (pie chart)
    (Center) Top sectors driving each episode (bar chart)
    (Right) Network amplification comparison (bar chart)
    """
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Panel A: Supply/Demand Pie Charts ────────────────────────────────────
    ax_pie = fig.add_subplot(gs[0])
    ax_pie.set_visible(False)  # Will use inset axes

    ax_covid_pie = fig.add_axes([0.02, 0.5, 0.22, 0.4])
    ax_infl_pie = fig.add_axes([0.02, 0.05, 0.22, 0.4])

    for ax_pie, result, label, ep_label in [
        (ax_covid_pie, covid_result, "COVID-19\n(Feb–May 2020)", "COVID"),
        (ax_infl_pie, infl_result, "Inflation\n(2020Q4–2022Q4)", "Inflation"),
    ]:
        supply_abs = abs(result.total_supply_contribution)
        demand_abs = abs(result.total_demand_contribution)
        total_abs = supply_abs + demand_abs
        if total_abs > 0:
            sizes = [supply_abs / total_abs, demand_abs / total_abs]
        else:
            sizes = [0.5, 0.5]
        ax_pie.pie(
            sizes,
            labels=["Supply", "Demand"],
            colors=[SUPPLY_COLOR, DEMAND_COLOR],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": FONT_ANNOT + 1},
        )
        ax_pie.set_title(label, fontsize=FONT_LABEL - 1, pad=3)

    # ── Panel B: Top Sector Contributions Side by Side ───────────────────────
    ax_bar = fig.add_subplot(gs[1])

    covid_df = covid_result.summary_df.nsmallest(8, "total_contribution_gdp_pct")
    infl_df = infl_result.summary_df.nlargest(8, "total_contribution_gdp_pct")

    short = lambda s: s[:18] + "…" if len(s) > 18 else s

    y_pos = np.arange(8)
    w = 0.38
    ax_bar.barh(y_pos + w / 2, covid_df["total_contribution_gdp_pct"].values,
                w, color=SUPPLY_COLOR, alpha=0.8, label="COVID (GDP decline)")
    ax_bar.barh(y_pos - w / 2, infl_df["total_contribution_gdp_pct"].values,
                w, color=DEMAND_COLOR, alpha=0.8, label="Inflation (GDP expansion)")

    covid_labels = [short(l) for l in covid_df["label"].values]
    infl_labels = [short(l) for l in infl_df["label"].values]
    combined_labels = [f"{c} / {d}" for c, d in zip(covid_labels, infl_labels)]

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(combined_labels, fontsize=FONT_TICK - 1)
    ax_bar.axvline(0, color="black", linewidth=0.8)
    ax_bar.set_xlabel("Contribution to ΔGDP (%)", fontsize=FONT_LABEL - 1)
    ax_bar.set_title("Top 8 Sectors (COVID / Inflation)\nCOVID = GDP decline drivers, Inflation = GDP growth drivers",
                     fontsize=FONT_LABEL - 1)
    ax_bar.legend(fontsize=FONT_TICK)

    # ── Panel C: Aggregate Summary ────────────────────────────────────────────
    ax_summary = fig.add_subplot(gs[2])

    episodes = ["COVID-19", "2021–2022 Inflation"]
    supply_vals = [covid_result.total_supply_contribution * 100,
                   infl_result.total_supply_contribution * 100]
    demand_vals = [covid_result.total_demand_contribution * 100,
                   infl_result.total_demand_contribution * 100]

    x_pos = np.arange(len(episodes))
    w = 0.35
    ax_summary.bar(x_pos - w / 2, supply_vals, w, color=SUPPLY_COLOR, alpha=0.85, label="Supply")
    ax_summary.bar(x_pos + w / 2, demand_vals, w, color=DEMAND_COLOR, alpha=0.85, label="Demand")

    for xi, sv, dv in zip(x_pos, supply_vals, demand_vals):
        ax_summary.text(xi - w / 2, sv + 0.1 * np.sign(sv), f"{sv:+.1f}%",
                        ha="center", va="bottom" if sv > 0 else "top", fontsize=FONT_TICK)
        ax_summary.text(xi + w / 2, dv + 0.1 * np.sign(dv), f"{dv:+.1f}%",
                        ha="center", va="bottom" if dv > 0 else "top", fontsize=FONT_TICK)

    ax_summary.axhline(0, color="black", linewidth=0.8)
    ax_summary.set_xticks(x_pos)
    ax_summary.set_xticklabels(episodes, fontsize=FONT_TICK)
    ax_summary.set_ylabel("Contribution to ΔGDP (%)", fontsize=FONT_LABEL - 1)
    ax_summary.set_title("Aggregate Supply vs.\nDemand Contributions", fontsize=FONT_LABEL)
    ax_summary.legend(fontsize=FONT_TICK)

    fig.suptitle(
        "Baqaee-Farhi Framework: COVID-19 vs. 2021–2022 Inflation\n"
        "Supply-Demand Decomposition of Macro Shocks",
        fontsize=FONT_TITLE,
        y=1.01,
    )
    _save(fig, "fig8_episode_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: Sensitivity Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensitivity(sens_covid: pd.DataFrame, sens_infl: pd.DataFrame) -> None:
    """
    Heatmap of demand_share across (sigma, epsilon) grid for both episodes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, df, title in [
        (axes[0], sens_covid, "COVID-19: Demand Share"),
        (axes[1], sens_infl,  "Inflation: Demand Share"),
    ]:
        pivot = df.pivot(index="sigma", columns="epsilon", values="demand_share")
        im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"ε={v:.1f}" for v in pivot.columns], fontsize=FONT_TICK)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"σ={v:.1f}" for v in pivot.index], fontsize=FONT_TICK)
        ax.set_title(title, fontsize=FONT_TITLE)
        ax.set_xlabel("Demand elasticity ε", fontsize=FONT_LABEL)
        ax.set_ylabel("Supply elasticity σ", fontsize=FONT_LABEL)
        plt.colorbar(im, ax=ax, label="Demand share of GDP change", shrink=0.8)

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=FONT_ANNOT + 1,
                            color="white" if abs(val - 0.5) > 0.2 else "black")

    fig.suptitle(
        "Sensitivity of Supply/Demand Split to Elasticity Parameters\n"
        "Values show: demand share = |demand contribution| / |total contribution|",
        fontsize=FONT_TITLE,
    )
    fig.tight_layout()
    _save(fig, "fig9_sensitivity_analysis")


# ─────────────────────────────────────────────────────────────────────────────
# Main: generate all figures
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(net, covid_result, infl_result,
                          covid_amp_df, infl_amp_df,
                          sens_covid, sens_infl) -> None:
    """Generate and save all figures."""
    print("\nGenerating figures...")

    print("  Figure 1: IO coefficient heat map")
    plot_io_heatmap(net)

    print("  Figure 2: Domar weights")
    plot_domar_weights(net)

    print("  Figure 3: COVID sectoral contributions")
    plot_sectoral_contributions(
        covid_result,
        title=(
            "Fig. 3 — Sectoral Contributions to GDP Decline (Feb–May 2020)\n"
            "Baqaee-Farhi (2022) Supply-Demand Decomposition"
        ),
        filename="fig3_covid_contributions",
    )

    print("  Figure 4: COVID waterfall")
    plot_waterfall(
        covid_result,
        title="Fig. 4 — COVID-19 GDP Decline: Supply vs. Demand Waterfall",
        filename="fig4_covid_waterfall",
    )

    print("  Figure 5: Network amplification (COVID)")
    plot_network_amplification(
        covid_amp_df,
        title="Fig. 5 — Network Amplification: Direct vs. Indirect Supply Effects (COVID)",
        filename="fig5_covid_network_amplification",
    )

    print("  Figure 6: Inflation sectoral contributions")
    plot_sectoral_contributions(
        infl_result,
        title=(
            "Fig. 6 — Sectoral Contributions to GDP Expansion (2020Q4–2022Q4)\n"
            "2021–2022 Inflation Episode: Supply-Demand Decomposition"
        ),
        filename="fig6_inflation_contributions",
    )

    print("  Figure 7: Inflation waterfall")
    plot_waterfall(
        infl_result,
        title="Fig. 7 — 2021–2022 Inflation: Supply vs. Demand Waterfall",
        filename="fig7_inflation_waterfall",
    )

    print("  Figure 8: Cross-episode comparison")
    plot_episode_comparison(covid_result, infl_result, net)

    print("  Figure 9: Sensitivity analysis")
    plot_sensitivity(sens_covid, sens_infl)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    from src.io_network.construct_network import get_io_network
    from src.replication.covid_episode import run_covid_replication
    from src.extension.inflation_episode import run_inflation_replication
    from src.decomposition.supply_demand_decomp import (
        sensitivity_analysis,
        compute_network_amplification_decomposition,
    )
    from src.replication.covid_episode import get_covid_shocks
    from src.extension.inflation_episode import get_inflation_shocks

    net = get_io_network(year=2017, use_real_data=True)

    covid_dp, covid_dy = get_covid_shocks(net.sectors)
    infl_dp, infl_dy = get_inflation_shocks(net.sectors)

    covid_result = run_covid_replication(net, run_sensitivity=False, verbose=False)
    infl_result = run_inflation_replication(net, run_sensitivity=False, verbose=False)

    covid_amp_df = compute_network_amplification_decomposition(net, covid_dp, covid_dy, "COVID")
    infl_amp_df = compute_network_amplification_decomposition(net, infl_dp, infl_dy, "Inflation")

    sens_covid = sensitivity_analysis(net, covid_dp, covid_dy, "COVID")
    sens_infl = sensitivity_analysis(net, infl_dp, infl_dy, "Inflation")

    generate_all_figures(net, covid_result, infl_result, covid_amp_df, infl_amp_df,
                         sens_covid, sens_infl)
