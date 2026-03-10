"""
Baqaee & Farhi (2022) — Full Replication and Extension Pipeline.

Runs all steps in sequence:
  Step 1: Build IO network (BEA 2017 data, embedded or downloaded)
  Step 2: Replicate COVID-19 episode (Feb–May 2020)
  Step 3: Extend to 2021–2022 inflation surge
  Step 4: Cross-episode comparison
  Step 5: Generate all figures and tables

Usage:
    python main.py [--download-data] [--no-figures] [--verbose]

Options:
    --download-data   Attempt to download real BEA/BLS data (requires API keys)
    --no-figures      Skip figure generation
    --verbose         Print detailed output
"""

import sys
import argparse
from pathlib import Path

# Ensure src is on path when running from repo root
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baqaee-Farhi (2022) Replication Pipeline"
    )
    parser.add_argument(
        "--download-data", action="store_true",
        help="Attempt to download BEA/BLS data (requires API keys)",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip matplotlib figure generation",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed intermediate output",
    )
    return parser.parse_args()


def banner(text: str, width: int = 65) -> None:
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def main():
    args = parse_args()
    verbose = args.verbose

    banner("Baqaee & Farhi (2022) — Replication Pipeline")
    print(
        "Paper: Supply and Demand in Disaggregated Keynesian Economies\n"
        "       with an Application to the COVID-19 Crisis.\n"
        "       AER 112(5): 1397–1435, 2022.\n"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Optional: download data
    # ─────────────────────────────────────────────────────────────────────────
    if args.download_data:
        banner("Data Download")
        print("Attempting to download BEA and BLS data...")
        try:
            from src.data_download.download_bea import main as bea_main
            bea_main()
        except EnvironmentError as e:
            print(f"  BEA download skipped: {e}")
        try:
            from src.data_download.download_bls import main as bls_main
            bls_main()
        except Exception as e:
            print(f"  BLS download failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Build IO Network
    # ─────────────────────────────────────────────────────────────────────────
    banner("Step 1: IO Network Construction")
    print("Building IO network from BEA 2017 data...")

    from src.io_network.construct_network import (
        get_io_network, verify_network, save_network
    )

    net = get_io_network(year=2017, use_real_data=True)
    verify_network(net)
    save_network(net)

    summary = net.summary()
    print("\nSector summary (top 10 by gross output):")
    top10 = summary.nlargest(10, "gross_output_bn")
    print(top10[["label", "gross_output_bn", "value_added_bn", "domar_weight",
                  "leontief_row_mult"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))

    banner("Step 1 Complete")
    print(f"  Sectors: {len(net.sectors)}")
    print(f"  GDP (value added): ${net.gdp:,.0f} bn")
    print(f"  Σ Domar weights: {net.lam.sum():.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: COVID Replication
    # ─────────────────────────────────────────────────────────────────────────
    banner("Step 2: COVID-19 Replication (Feb–May 2020)")
    print("This replicates Table 2 and Figures 2–3 of Baqaee & Farhi (2022).")

    from src.replication.covid_episode import (
        run_covid_replication, generate_paper_table2, aggregate_contributions_by_type
    )

    covid_result = run_covid_replication(net, verbose=True, run_sensitivity=True)

    print("\nTable 2 Analogue:")
    table2 = generate_paper_table2(covid_result, net)
    print(table2.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nContributions by Sector Group:")
    agg = aggregate_contributions_by_type(covid_result)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    banner("Step 2 Complete — COVID Decomposition")
    print(f"  Total ΔGDP/GDP:  {covid_result.total_delta_gdp*100:+.2f}%")
    print(f"  Supply share:    {covid_result.supply_share:.1%}")
    print(f"  Demand share:    {covid_result.demand_share:.1%}")
    print(f"  Network amplification: {covid_result.total_network_amplification*100:+.2f}pp")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: 2021–2022 Inflation Extension
    # ─────────────────────────────────────────────────────────────────────────
    banner("Step 3: 2021–2022 Inflation Extension")
    print("Applying the Baqaee-Farhi framework to the post-pandemic inflation surge.")

    from src.extension.inflation_episode import (
        run_inflation_replication, classify_inflation_drivers
    )

    infl_result = run_inflation_replication(net, verbose=True, run_sensitivity=True)

    print("\nInflation Driver Classification:")
    classified = classify_inflation_drivers(infl_result)
    print(classified[["label", "supply_shock", "demand_shock",
                       "supply_contribution_gdp_pct",
                       "demand_contribution_gdp_pct", "inflation_type"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))
    classified.to_csv("outputs/tables/inflation_driver_classification.csv", index=False)

    banner("Step 3 Complete — Inflation Decomposition")
    print(f"  Total ΔGDP/GDP:  {infl_result.total_delta_gdp*100:+.2f}%")
    print(f"  Supply share:    {infl_result.supply_share:.1%}")
    print(f"  Demand share:    {infl_result.demand_share:.1%}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Cross-Episode Comparison
    # ─────────────────────────────────────────────────────────────────────────
    banner("Step 4: Cross-Episode Comparison")

    from src.extension.comparison import (
        build_comparison_table, build_sector_comparison, print_comparison_narrative
    )

    print_comparison_narrative(covid_result, infl_result, net)

    comp_table = build_comparison_table(covid_result, infl_result, net)
    print("\nSummary Comparison Table:")
    print(comp_table.to_string(index=False))

    sector_comp = build_sector_comparison(covid_result, infl_result)
    print(f"\nSector comparison saved to outputs/tables/sector_episode_comparison.csv")

    banner("Step 4 Complete")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Figures
    # ─────────────────────────────────────────────────────────────────────────
    if not args.no_figures:
        banner("Step 5: Figure Generation")

        from src.decomposition.supply_demand_decomp import (
            sensitivity_analysis,
            compute_network_amplification_decomposition,
        )
        from src.replication.covid_episode import get_covid_shocks
        from src.extension.inflation_episode import get_inflation_shocks
        from src.visualization.create_figures import generate_all_figures

        covid_dp, covid_dy = get_covid_shocks(net.sectors)
        infl_dp, infl_dy = get_inflation_shocks(net.sectors)

        covid_amp_df = compute_network_amplification_decomposition(
            net, covid_dp, covid_dy, "COVID"
        )
        infl_amp_df = compute_network_amplification_decomposition(
            net, infl_dp, infl_dy, "Inflation"
        )
        sens_covid = sensitivity_analysis(net, covid_dp, covid_dy, "COVID")
        sens_infl = sensitivity_analysis(net, infl_dp, infl_dy, "Inflation")

        generate_all_figures(
            net, covid_result, infl_result,
            covid_amp_df, infl_amp_df,
            sens_covid, sens_infl,
        )

        banner("Step 5 Complete")
        print("All figures saved to outputs/figures/")

    # ─────────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────────
    banner("Pipeline Complete")
    print("\nKey results:")
    print(f"\nCOVID-19 (Feb–May 2020):")
    print(f"  ΔGDP/GDP = {covid_result.total_delta_gdp*100:+.2f}%")
    print(f"  Demand-driven: {covid_result.demand_share:.0%}")
    print(f"  Supply-driven: {covid_result.supply_share:.0%}")

    print(f"\n2021–2022 Inflation (2020Q4–2022Q4):")
    print(f"  ΔGDP/GDP = {infl_result.total_delta_gdp*100:+.2f}%")
    print(f"  Supply-driven: {infl_result.supply_share:.0%}")
    print(f"  Demand-driven: {infl_result.demand_share:.0%}")

    print("\nOutput files:")
    from pathlib import Path
    tables = list(Path("outputs/tables").glob("*.csv"))
    figs = list(Path("outputs/figures").glob("*.png"))
    for f in sorted(tables):
        print(f"  [table] {f}")
    for f in sorted(figs):
        print(f"  [figure] {f}")

    print("\nSee decisions_log.md for all data and methodological decisions.")
    print("See README.md for full documentation.\n")


if __name__ == "__main__":
    main()
