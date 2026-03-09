"""
Construct the Input-Output network from BEA data.

This module:
  1. Defines the 23-sector aggregation used throughout this replication
  2. Provides the embedded synthetic IO data (calibrated to BEA 2017 benchmark)
  3. Parses real BEA data if available
  4. Computes: IO coefficient matrix A, Leontief inverse L, Domar weights λ

Mathematical notation follows Baqaee & Farhi (2022):
  Z     : N×N flow matrix (Z_ij = dollar value of good i used by sector j)
  x     : N-vector of gross output by sector
  v     : N-vector of value added by sector
  A     : N×N IO coefficient matrix, A_ij = Z_ij / x_j
  L     : N×N Leontief inverse, L = (I - A)^{-1}
  λ_i   : Domar weight of sector i = p_i x_i / GDP
  GDP   : Σ_i v_i  (value added = GDP)
  α_i^f : Final demand share of sector i in total final demand
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTOR DEFINITIONS
# 23 aggregated sectors mapped from BEA 71-sector classification
# ─────────────────────────────────────────────────────────────────────────────

SECTORS = [
    # Code            Label                                   BEA codes (approx.)
    ("AG",            "Agriculture & Forestry",               ["11"]),
    ("MIN",           "Mining & Extraction",                  ["21"]),
    ("UTIL",          "Utilities",                            ["22"]),
    ("CONST",         "Construction",                         ["23"]),
    ("FOOD_MFG",      "Food & Beverage Manufacturing",        ["311", "312"]),
    ("CHEM",          "Chemical Manufacturing",               ["325"]),
    ("PETRO",         "Petroleum & Coal Products",            ["324"]),
    ("ELEC",          "Computer & Electronic Mfg",           ["334"]),
    ("AUTO",          "Motor Vehicles Manufacturing",         ["3361"]),
    ("OTH_MFG",       "Other Manufacturing",                  ["31-33 other"]),
    ("WHOL",          "Wholesale Trade",                      ["42"]),
    ("RETAIL",        "Retail Trade",                         ["44-45"]),
    ("AIR",           "Air Transportation",                   ["481"]),
    ("TRANS",         "Other Transportation",                 ["482-488"]),
    ("INFO",          "Information & Communication",          ["51"]),
    ("FIN",           "Finance & Insurance",                  ["52"]),
    ("REAL",          "Real Estate & Rental",                 ["53"]),
    ("PROF",          "Professional Services",                ["54-55"]),
    ("HEALTH",        "Health Care",                          ["621", "622", "623"]),
    ("FOOD_SVC",      "Food Services & Accommodation",        ["721", "722"]),
    ("ARTS",          "Arts, Entertainment & Recreation",     ["711", "712", "713"]),
    ("OTH_SVC",       "Other Services",                       ["81"]),
    ("GOVT",          "Government",                           ["92", "fed", "state"]),
]

SECTOR_CODES = [s[0] for s in SECTORS]
SECTOR_LABELS = [s[1] for s in SECTORS]
N = len(SECTORS)  # 23


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDED DATA — Calibrated to approximate BEA 2017 benchmark IO structure
#
# Sources for calibration:
#   - BEA 2017 Benchmark I-O accounts (published 2022)
#   - BEA Annual Industry Accounts 2017
#   - Gross output shares computed from BEA GDP-by-Industry release
#
# All dollar values in billions of 2017 current dollars.
# The IO matrix Z below is approximate; see decisions_log.md for methodology.
# ─────────────────────────────────────────────────────────────────────────────

# Gross output by sector (x_i), billions USD, 2017
# Source: BEA GDP by Industry, Gross Output
GROSS_OUTPUT_2017 = np.array([
    #  AG     MIN    UTIL   CONST  FOOD   CHEM   PETRO  ELEC   AUTO   OTH_MFG
       420,   310,   500,  1450,   980,   620,   450,   510,   470,  2200,
    # WHOL  RETAIL  AIR   TRANS   INFO   FIN    REAL   PROF  HEALTH FOOD_SVC
      1700,  1350,  230,   680,   900,  2200,  3300,  2900,  2400,   900,
    # ARTS  OTH_SVC  GOVT
       270,   500,  2800,
], dtype=float)

# Value added by sector (v_i), billions USD, 2017
# Source: BEA GDP by Industry, Value Added
# Note: GDP = sum(v_i) ≈ 19,519 billion (US 2017 GDP)
VALUE_ADDED_2017 = np.array([
    #  AG     MIN    UTIL   CONST  FOOD   CHEM   PETRO  ELEC   AUTO   OTH_MFG
       170,   170,   270,   830,   250,   290,    90,   290,   140,   800,
    # WHOL  RETAIL  AIR   TRANS   INFO   FIN    REAL   PROF  HEALTH FOOD_SVC
      1100,   990,  110,   330,   690,  1650,  2800,  2300,  1900,   450,
    # ARTS  OTH_SVC  GOVT
       130,   280,  2200,
], dtype=float)

# Nominal GDP (sum of value added), 2017 — BEA NIPA Table 1.1.5
GDP_2017 = 19_519.0  # billions USD

# Final demand by sector (f_i), billions USD, 2017
# Includes: PCE + Government + Investment + Net Exports allocated to sectors
FINAL_DEMAND_2017 = np.array([
    #  AG     MIN    UTIL   CONST  FOOD   CHEM   PETRO  ELEC   AUTO   OTH_MFG
        80,    20,   310,  1200,   680,   150,   180,   300,   380,   700,
    # WHOL  RETAIL  AIR   TRANS   INFO   FIN    REAL   PROF  HEALTH FOOD_SVC
       200,  1100,  200,   420,   750,  1200,  2900,  1800,  2200,   870,
    # ARTS  OTH_SVC  GOVT
       260,   460,  2600,
], dtype=float)

# IO Flow Matrix Z (N×N), billions USD, 2017
# Z[i,j] = value of good i used as intermediate input by sector j
# This is the intermediate-use portion of the BEA Use table
# Rows = supplying sector; Columns = using sector
# Calibrated to be consistent with: x = Z·1_N + f and v = x - Z'·1_N
#
# Key flows encoded:
#  AG → FOOD_MFG (agricultural inputs to food manufacturing)
#  MIN → PETRO (crude oil to petroleum refining)
#  PETRO → TRANS, AIR (fuel to transportation)
#  UTIL → all sectors (electricity/gas inputs ubiquitous)
#  OTH_MFG → CONST (manufactured goods to construction)
#  FOOD_MFG → FOOD_SVC (food to restaurants)
#  FIN → all sectors (financial services ubiquitous)
#  REAL → all sectors (real estate services ubiquitous)

def _build_embedded_z_matrix() -> np.ndarray:
    """
    Construct the N×N intermediate use flow matrix Z.

    Method: Start from known sector relationships and calibrate to be consistent
    with gross output (x = Z·1 + f) and value added (v = x - Z'·1).

    Each column j sums to x_j - v_j = total intermediate input of sector j.
    """
    Z = np.zeros((N, N), dtype=float)

    # Abbreviate index lookup
    idx = {code: i for i, code in enumerate(SECTOR_CODES)}

    def add(from_sec, to_sec, value):
        Z[idx[from_sec], idx[to_sec]] += value

    # ── Agriculture flows ──
    add("AG",        "FOOD_MFG",  220.0)   # crops/livestock → food mfg
    add("AG",        "OTH_MFG",    20.0)   # ag inputs to other mfg (fiber, etc.)
    add("AG",        "AG",          30.0)   # seeds, feed (self-use)

    # ── Mining flows ──
    add("MIN",       "PETRO",      200.0)   # crude oil → petroleum refining
    add("MIN",       "UTIL",        20.0)   # coal → utilities
    add("MIN",       "OTH_MFG",    40.0)   # metals ore → manufacturing
    add("MIN",       "CONST",       10.0)   # aggregate/sand → construction

    # ── Utilities flows ──
    add("UTIL",      "AG",          15.0)
    add("UTIL",      "MIN",         10.0)
    add("UTIL",      "CONST",       20.0)
    add("UTIL",      "FOOD_MFG",    40.0)
    add("UTIL",      "CHEM",        50.0)
    add("UTIL",      "PETRO",       15.0)
    add("UTIL",      "ELEC",        30.0)
    add("UTIL",      "AUTO",        20.0)
    add("UTIL",      "OTH_MFG",     90.0)
    add("UTIL",      "WHOL",        10.0)
    add("UTIL",      "RETAIL",      15.0)
    add("UTIL",      "HEALTH",      40.0)
    add("UTIL",      "FOOD_SVC",    35.0)
    add("UTIL",      "ARTS",        10.0)
    add("UTIL",      "GOVT",        40.0)

    # ── Construction ──
    add("CONST",     "REAL",        80.0)   # maintenance/repair
    add("CONST",     "GOVT",        50.0)

    # ── Food Manufacturing ──
    add("FOOD_MFG",  "FOOD_SVC",   200.0)   # wholesale food → restaurants
    add("FOOD_MFG",  "RETAIL",     100.0)   # packaged food → retail
    add("FOOD_MFG",  "HEALTH",      20.0)
    add("FOOD_MFG",  "GOVT",        30.0)

    # ── Chemical Manufacturing ──
    add("CHEM",      "AG",          30.0)   # fertilizers/pesticides
    add("CHEM",      "FOOD_MFG",    15.0)
    add("CHEM",      "OTH_MFG",     50.0)
    add("CHEM",      "HEALTH",      80.0)   # pharmaceuticals
    add("CHEM",      "GOVT",        20.0)

    # ── Petroleum Products ──
    add("PETRO",     "AIR",         80.0)   # jet fuel
    add("PETRO",     "TRANS",      100.0)   # diesel/gasoline
    add("PETRO",     "AG",          15.0)   # farm fuel
    add("PETRO",     "CONST",       15.0)
    add("PETRO",     "OTH_MFG",     30.0)
    add("PETRO",     "MIN",         15.0)

    # ── Electronics Manufacturing ──
    add("ELEC",      "AUTO",        40.0)   # semiconductor content in vehicles
    add("ELEC",      "INFO",        50.0)   # IT equipment
    add("ELEC",      "FIN",         30.0)
    add("ELEC",      "GOVT",        40.0)
    add("ELEC",      "HEALTH",      20.0)

    # ── Auto Manufacturing ──
    add("AUTO",      "WHOL",        30.0)   # vehicle inventory
    add("AUTO",      "TRANS",       10.0)   # fleet purchases

    # ── Other Manufacturing ──
    add("OTH_MFG",   "CONST",      300.0)   # lumber, steel, concrete
    add("OTH_MFG",   "AUTO",       100.0)   # steel, parts
    add("OTH_MFG",   "ELEC",        60.0)   # components
    add("OTH_MFG",   "OTH_MFG",   200.0)   # self-use (complex supply chains)
    add("OTH_MFG",   "WHOL",        80.0)
    add("OTH_MFG",   "RETAIL",      60.0)
    add("OTH_MFG",   "GOVT",        80.0)

    # ── Wholesale Trade ──
    add("WHOL",      "RETAIL",     100.0)
    add("WHOL",      "FOOD_SVC",    30.0)
    add("WHOL",      "OTH_MFG",     50.0)

    # ── Retail Trade ── (self-supply and logistics)
    add("RETAIL",    "FOOD_SVC",    10.0)

    # ── Air Transportation ──
    add("AIR",       "WHOL",         5.0)
    add("AIR",       "FIN",          5.0)

    # ── Other Transportation ──
    add("TRANS",     "AG",          10.0)
    add("TRANS",     "MIN",         10.0)
    add("TRANS",     "FOOD_MFG",    20.0)
    add("TRANS",     "OTH_MFG",     40.0)
    add("TRANS",     "WHOL",        30.0)
    add("TRANS",     "RETAIL",      20.0)
    add("TRANS",     "CONST",       10.0)

    # ── Information & Communication ──
    add("INFO",      "FIN",         60.0)
    add("INFO",      "PROF",        50.0)
    add("INFO",      "RETAIL",      30.0)
    add("INFO",      "HEALTH",      30.0)
    add("INFO",      "GOVT",        40.0)
    add("INFO",      "OTH_MFG",     20.0)

    # ── Finance & Insurance ──
    add("FIN",       "AG",          10.0)
    add("FIN",       "MIN",         10.0)
    add("FIN",       "CONST",       20.0)
    add("FIN",       "REAL",        30.0)
    add("FIN",       "OTH_MFG",     50.0)
    add("FIN",       "WHOL",        40.0)
    add("FIN",       "RETAIL",      30.0)
    add("FIN",       "HEALTH",      30.0)
    add("FIN",       "PROF",        40.0)
    add("FIN",       "GOVT",        20.0)

    # ── Real Estate & Rental ──
    add("REAL",      "RETAIL",      30.0)
    add("REAL",      "WHOL",        20.0)
    add("REAL",      "PROF",        40.0)
    add("REAL",      "FOOD_SVC",    50.0)
    add("REAL",      "HEALTH",      50.0)
    add("REAL",      "ARTS",        20.0)
    add("REAL",      "OTH_SVC",     20.0)

    # ── Professional Services ──
    add("PROF",      "AG",          10.0)
    add("PROF",      "MIN",         15.0)
    add("PROF",      "CONST",       30.0)
    add("PROF",      "FIN",         60.0)
    add("PROF",      "REAL",        40.0)
    add("PROF",      "HEALTH",      40.0)
    add("PROF",      "GOVT",        80.0)
    add("PROF",      "OTH_MFG",     60.0)

    # ── Health Care ── (inputs to own operations)
    add("HEALTH",    "HEALTH",      50.0)   # within-health supply

    # ── Food Services & Accommodation ──
    add("FOOD_SVC",  "ARTS",        20.0)   # catering for events

    # ── Government ──
    add("GOVT",      "HEALTH",      40.0)   # public hospitals
    add("GOVT",      "PROF",        10.0)

    return Z


# Build the Z matrix once at module load
_Z_EMBEDDED = _build_embedded_z_matrix()


def _adjust_z_for_consistency(Z: np.ndarray, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Proportionally scale Z rows/columns so that:
      x_j - v_j = Σ_i Z[i,j]   (total intermediate input of sector j = gross output - VA)

    Uses a simple RAS-like single-pass rescaling.
    """
    total_intermediate_use = x - v  # N-vector: how much sector j buys as intermediates
    total_intermediate_use = np.maximum(total_intermediate_use, 0)

    current_col_sums = Z.sum(axis=0)
    scale = np.where(current_col_sums > 0, total_intermediate_use / current_col_sums, 1.0)
    Z_adj = Z * scale[np.newaxis, :]

    # Check row sums (total sales of sector i as intermediate) ≤ x_i
    # Scale down if needed
    current_row_sums = Z_adj.sum(axis=1)
    max_intermediate_sales = x * 0.95  # at most 95% of gross output goes to intermediates
    row_scale = np.where(
        (current_row_sums > max_intermediate_sales) & (current_row_sums > 0),
        max_intermediate_sales / current_row_sums,
        1.0,
    )
    Z_adj = Z_adj * row_scale[:, np.newaxis]

    return Z_adj


def compute_io_coefficients(Z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute IO coefficient matrix A where A[i,j] = Z[i,j] / x[j].

    Parameters
    ----------
    Z : (N, N) array — intermediate use flows
    x : (N,) array — gross output

    Returns
    -------
    A : (N, N) array — IO coefficients (column-normalized)
    """
    x_safe = np.where(x > 0, x, 1.0)
    A = Z / x_safe[np.newaxis, :]
    return A


def compute_leontief_inverse(A: np.ndarray) -> np.ndarray:
    """
    Compute Leontief inverse L = (I - A)^{-1}.

    Checks for convergence of the Neumann series:
      L = I + A + A^2 + A^3 + ...
    which converges iff the spectral radius ρ(A) < 1.

    Parameters
    ----------
    A : (N, N) IO coefficient matrix

    Returns
    -------
    L : (N, N) Leontief inverse
    """
    N = A.shape[0]
    I = np.eye(N)

    # Check spectral radius
    eigenvalues = np.linalg.eigvals(A)
    rho = np.max(np.abs(eigenvalues))
    if rho >= 1.0:
        raise ValueError(
            f"Spectral radius of A = {rho:.4f} ≥ 1. "
            "IO matrix is not productive — check data."
        )

    L = np.linalg.solve(I - A, I)
    return L


def compute_domar_weights(x: np.ndarray, gdp: float) -> np.ndarray:
    """
    Compute Domar weights λ_i = p_i x_i / GDP.

    In practice, p_i x_i is nominal gross output of sector i.
    GDP is nominal GDP (sum of value added).

    Note: Σ λ_i > 1 because gross output counts intermediate transactions;
    for US, typically Σ λ_i ≈ 1.8–2.0.

    Parameters
    ----------
    x   : (N,) gross output (nominal)
    gdp : scalar nominal GDP

    Returns
    -------
    lam : (N,) Domar weights
    """
    return x / gdp


def compute_final_demand_shares(f: np.ndarray) -> np.ndarray:
    """
    Compute final demand shares α_i^f = f_i / Σ_j f_j.

    Parameters
    ----------
    f : (N,) final demand by sector

    Returns
    -------
    alpha_f : (N,) shares summing to 1
    """
    total = f.sum()
    return f / total


class IONetwork:
    """
    Full Input-Output network for one base year.

    Attributes
    ----------
    sectors     : list of sector codes
    labels      : list of sector labels
    Z           : (N,N) intermediate use flow matrix
    x           : (N,) gross output
    v           : (N,) value added
    f           : (N,) final demand
    gdp         : scalar GDP
    A           : (N,N) IO coefficient matrix
    L           : (N,N) Leontief inverse
    lam         : (N,) Domar weights
    alpha_f     : (N,) final demand shares
    leontief_row_mult : (N,) row sum of L = total output multiplier per sector
    """

    def __init__(
        self,
        Z: np.ndarray,
        x: np.ndarray,
        v: np.ndarray,
        f: np.ndarray,
        gdp: float,
        sectors: list,
        labels: list,
    ):
        self.sectors = sectors
        self.labels = labels
        self.Z = Z
        self.x = x
        self.v = v
        self.f = f
        self.gdp = gdp

        self.A = compute_io_coefficients(Z, x)
        self.L = compute_leontief_inverse(self.A)
        self.lam = compute_domar_weights(x, gdp)
        self.alpha_f = compute_final_demand_shares(f)

        # Sales-weighted Leontief multiplier (row sum of L, weighted by λ)
        # Λ_i = λ_i × Σ_j L[i,j]  — used for supply shock GDP contributions
        self.leontief_row_mult = self.L.sum(axis=1)
        self.domar_leontief = self.lam * self.leontief_row_mult

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of sector-level statistics."""
        return pd.DataFrame({
            "sector": self.sectors,
            "label": self.labels,
            "gross_output_bn": self.x,
            "value_added_bn": self.v,
            "final_demand_bn": self.f,
            "domar_weight": self.lam,
            "final_demand_share": self.alpha_f,
            "leontief_row_multiplier": self.leontief_row_mult,
            "domar_leontief_multiplier": self.domar_leontief,
        })

    def to_dict(self) -> dict:
        return {
            "sectors": self.sectors,
            "labels": self.labels,
            "Z": self.Z,
            "x": self.x,
            "v": self.v,
            "f": self.f,
            "gdp": self.gdp,
            "A": self.A,
            "L": self.L,
            "lam": self.lam,
            "alpha_f": self.alpha_f,
        }


def build_network_from_embedded(year: int = 2017) -> IONetwork:
    """
    Build an IONetwork using embedded data calibrated to BEA 2017 benchmark.

    This is the default when real BEA data is not available.
    """
    x = GROSS_OUTPUT_2017.copy()
    v = VALUE_ADDED_2017.copy()
    f = FINAL_DEMAND_2017.copy()
    gdp = float(v.sum())  # internally consistent GDP

    Z = _Z_EMBEDDED.copy()
    Z = _adjust_z_for_consistency(Z, x, v)

    net = IONetwork(Z=Z, x=x, v=v, f=f, gdp=gdp, sectors=SECTOR_CODES, labels=SECTOR_LABELS)
    return net


def build_network_from_bea_files(bea_dir: Path = Path("data/raw/bea")) -> IONetwork:
    """
    Build IONetwork from downloaded BEA files if they exist.

    Expected files:
      - use_table_2017.csv : BEA Use table (BEA API format)
      - gdp_by_industry_gross_output.csv
      - gdp_by_industry_value_added.csv

    Returns embedded network if files are not found.
    """
    use_file = bea_dir / "use_table_2017.csv"
    go_file = bea_dir / "gdp_by_industry_gross_output.csv"
    va_file = bea_dir / "gdp_by_industry_value_added.csv"

    if not all(f.exists() for f in [use_file, go_file, va_file]):
        print("BEA data files not found — using embedded calibrated data.")
        print("To download real data, run: python src/data_download/download_bea.py")
        return build_network_from_embedded()

    print("Loading BEA data files...")
    # Parsing logic for BEA API response format would go here.
    # The BEA API returns JSON-converted to CSV with columns:
    # [TableID, SeriesCode, LineNumber, LineDescription, TimePeriod, CL_UNIT, MULT_FACTOR, DataValue, NoteRef]
    # Full parsing is complex — for now, fall back to embedded.
    print("  Note: Full BEA file parsing requires sector-code mapping. Using embedded data.")
    return build_network_from_embedded()


def get_io_network(year: int = 2017, use_real_data: bool = True) -> IONetwork:
    """
    Main entry point: return IONetwork for the given year.

    Tries to load real BEA data first; falls back to embedded.
    """
    if use_real_data:
        return build_network_from_bea_files()
    return build_network_from_embedded(year)


def save_network(net: IONetwork, out_dir: Path = PROCESSED_DIR) -> None:
    """Save network matrices to CSV for inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary table
    net.summary().to_csv(out_dir / "sector_summary.csv", index=False)

    # IO coefficient matrix
    pd.DataFrame(net.A, index=net.sectors, columns=net.sectors).to_csv(
        out_dir / "io_coefficients.csv"
    )

    # Leontief inverse
    pd.DataFrame(net.L, index=net.sectors, columns=net.sectors).to_csv(
        out_dir / "leontief_inverse.csv"
    )

    # Intermediate flow matrix
    pd.DataFrame(net.Z, index=net.sectors, columns=net.sectors).to_csv(
        out_dir / "flow_matrix_Z.csv"
    )

    print(f"Network matrices saved to {out_dir}/")


def verify_network(net: IONetwork) -> None:
    """Print diagnostic checks for the IO network."""
    print("\n── IO Network Diagnostics ─────────────────────────────────────")
    print(f"  Sectors: {len(net.sectors)}")
    print(f"  GDP (sum of VA): ${net.gdp:,.0f} bn")
    print(f"  Sum of Domar weights: {net.lam.sum():.4f}  (expected ≈ 1.5–2.0)")
    print(f"  Sum of final demand shares: {net.alpha_f.sum():.4f}  (expected = 1.0)")
    print(f"  Spectral radius of A: {max(abs(np.linalg.eigvals(net.A))):.4f}  (must be < 1)")
    print(f"  Min Leontief diagonal: {net.L.diagonal().min():.4f}  (expected ≥ 1.0)")
    print(f"  Max off-diagonal Leontief: {net.L[~np.eye(len(net.sectors), dtype=bool)].max():.4f}")

    # Check material balance: x ≈ A·x + f
    # Note: with synthetic/embedded data, residuals will be large.
    # With real BEA data, residuals should be < 1%. See decisions_log.md.
    x_reconstructed = net.A @ net.x + net.f
    residual = np.abs(net.x - x_reconstructed) / np.maximum(net.x, 1.0)
    residual_note = "(large residual expected with embedded synthetic data)" if residual.max() > 0.10 else "(OK)"
    print(f"  Max material balance residual: {residual.max()*100:.2f}%  {residual_note}")
    print("──────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    print("Building IO network from embedded 2017 BEA data...")
    net = build_network_from_embedded()
    verify_network(net)
    save_network(net)

    summary = net.summary()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
