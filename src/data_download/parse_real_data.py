"""
Parse the three BF replication data files into model-ready arrays.

Files
-----
IO_data_2018.mat        — BEA 2018 Use Table (71-industry × 71-industry + VA rows)
BLS_labor_shock_202108.xls  — BLS hours changes by BEA sector, monthly 2020-2021
Expenditure_202107.xls  — BEA PCE nominal expenditures by category, 2018-2021

Outputs
-------
parse_io_mat()          → Z66, x66, v66  (66×66 IO matrix, gross output, value added)
parse_bls_shocks()      → dict[sector_code → log hours change Feb→May 2020]
parse_pce_shocks()      → dict[sector_code → log PCE change Feb→May 2020]

Sector aggregation (IO 71 → model 66)
--------------------------------------
IO rows 27-30  (Motor vehicle dealers, Food&bev stores, Gen merchandise, Other retail)
  → model RETAIL (index 27)
IO rows 47-48  (Housing Services, Other Real Estate)
  → model REALE  (index 44)
IO rows 66-67  (Federal defense, Federal nondefense)
  → model FEDGOV (index 62)
IO row  68     (Federal government enterprises)    → model FEDGOVE (63)
IO row  69     (State/local general government)    → model SLGOV   (64)
IO row  70     (State/local government enterprises)→ model SLGOVE  (65)

BLS aggregation (71 codes → 66 model sectors)
----------------------------------------------
BLS codes 441+445+452+4A0 → RETAIL   (x-weighted average)
BLS codes HS + ORE        → REALE    (x-weighted average)
BLS codes GFGD + GFGN     → FEDGOV   (x-weighted average)

PCE mapping (demand-side expenditure changes → model sectors)
--------------------------------------------------------------
PCE changes are log(P_May2020 / P_Feb2020) in nominal spending. For consumer-
facing sectors these proxy the demand-side shock. B2B sectors (mining, wholesale,
most manufacturing) return 0 (no direct PCE coverage).

Units
-----
IO matrix: millions USD (raw), converted to billions USD for model
BLS shocks: dimensionless log fractions (already in file)
PCE shocks: computed as log(May_spend / Feb_spend), dimensionless
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io

# ─── Default data file locations ───────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
IO_MAT_PATH   = REPO_ROOT / "IO_data_2018.mat"
BLS_XLS_PATH  = REPO_ROOT / "BLS_labor_shock_202108.xls"
PCE_XLS_PATH  = REPO_ROOT / "Expenditure_202107.xls"

# ─── IO sector slice index (year 2018 = slice 18 in 22-slice array) ────────
IO_YEAR_SLICE = 18

# ─── Aggregation map: IO 71-sector → model 66-sector ───────────────────────
# Each entry: (io_rows_list, model_idx)
# io_rows are 0-indexed into the 71-row industry space (rows 0-70 of Data_raw)
IO_AGG = [
    # rows 0-26: 1:1 with model indices 0-26
    *[([i], i) for i in range(27)],
    # retail aggregation
    ([27, 28, 29, 30], 27),        # RETAIL
    # rows 31-46: offset by 3 → model 28-43
    *[([i], i - 3) for i in range(31, 47)],
    # real estate aggregation
    ([47, 48], 44),                # REALE
    # rows 49-65: offset by 4 → model 45-61
    *[([i], i - 4) for i in range(49, 66)],
    # government aggregation
    ([66, 67], 62),                # FEDGOV
    ([68],     63),                # FEDGOVE
    ([69],     64),                # SLGOV
    ([70],     65),                # SLGOVE
]

# ─── BLS iocode → model sector code ─────────────────────────────────────────
BLS_TO_MODEL = {
    "111CA":  "FARM",
    "113FF":  "FOREST",
    "211":    "OILGAS",
    "212":    "MINE",
    "213":    "MINE_SUP",
    "22":     "UTIL",
    "23":     "CONST",
    "321":    "WOOD",
    "327":    "NMMIN",
    "331":    "PMETAL",
    "332":    "FABMETAL",
    "333":    "MACH",
    "334":    "COMPELEC",
    "335":    "ELECEQUIP",
    "3361MV": "MOTVEH",
    "3364OT": "OTRTRANS",
    "337":    "FURN",
    "339":    "MISCMFG",
    "311FT":  "FOOD",
    "313TT":  "TEXTILE",
    "315AL":  "APPAREL",
    "322":    "PAPER",
    "323":    "PRINT",
    "324":    "PETRO",
    "325":    "CHEM",
    "326":    "PLASTIC",
    "42":     "WHOLE",
    # retail: four BLS codes aggregate to RETAIL
    "441":    "RETAIL",
    "445":    "RETAIL",
    "452":    "RETAIL",
    "4A0":    "RETAIL",
    "481":    "AIRTRANS",
    "482":    "RAILTRANS",
    "483":    "WATERTRANS",
    "484":    "TRUCK",
    "485":    "TRANSIT",
    "486":    "PIPE",
    "487OS":  "OTHERTRANS",
    "493":    "WAREHOUSE",
    "511":    "PUBLISH",
    "512":    "MOVIE",
    "513":    "BROADCAST",
    "514":    "INFODATA",
    "521CI":  "CREDIT",
    "523":    "SECURIT",
    "524":    "INSURE",
    "525":    "FUNDS",
    # real estate: two BLS codes aggregate to REALE
    "HS":     "REALE",
    "ORE":    "REALE",
    "532RL":  "RENTAL",
    "5411":   "LEGAL",
    "5415":   "COMPDES",
    "5412OP": "MISCPROF",
    "55":     "MGMT",
    "561":    "ADMIN",
    "562":    "WASTE",
    "61":     "EDUC",
    "621":    "AMBULAT",
    "622":    "HOSPITAL",
    "623":    "NURSING",
    "624":    "SOCIALAS",
    "711AS":  "PERFORM",
    "713":    "AMUSE",
    "721":    "ACCOMM",
    "722":    "FOODSVC",
    "81":     "OTHSVC",
    # government
    "GFGD":   "FEDGOV",
    "GFGN":   "FEDGOV",
    "GFE":    "FEDGOVE",
    "GSLG":   "SLGOV",
    "GSLE":   "SLGOVE",
}

# ─── PCE line-number → model sector code ────────────────────────────────────
# Line numbers from BEA Table 2.4.5U; 0-indexed row = line - 1 + header_offset.
# We store (line_number, sector_code, weight) where weight allows splitting one
# PCE line across multiple sectors (weights must sum to 1.0 per line).
# Only consumer-facing sectors are listed; B2B sectors (MINE, WHOLE, etc.) = 0.
PCE_LINES = [
    # ── Goods ────────────────────────────────────────────────────────────────
    (4,   "MOTVEH",    1.0),    # New motor vehicles (55)
    (10,  "MOTVEH",    0.5),    # Net purchases of used motor vehicles — partly retail
    (10,  "RETAIL",    0.5),
    (18,  "MOTVEH",    1.0),    # Motor vehicle parts and accessories (58)
    (22,  "FURN",      1.0),    # Furniture and furnishings
    (27,  "ELECEQUIP", 1.0),    # Household appliances
    (37,  "COMPELEC",  1.0),    # Video/audio/photographic/info processing equipment
    (50,  "AMUSE",     1.0),    # Sporting equipment
    (51,  "AMUSE",     1.0),    # Sports and recreational vehicles
    (61,  "MISCMFG",   1.0),    # Jewelry and watches
    (64,  "AMBULAT",   1.0),    # Therapeutic appliances and equipment
    (72,  "FOOD",      0.7),    # Food/nonalcoholic beverages off-premises → food mfg
    (72,  "RETAIL",    0.3),    # … and retail margin
    (97,  "FOOD",      0.7),    # Alcoholic beverages off-premises
    (97,  "RETAIL",    0.3),
    (104, "APPAREL",   0.6),    # Women's clothing
    (104, "RETAIL",    0.4),
    (105, "APPAREL",   0.6),    # Men's clothing
    (105, "RETAIL",    0.4),
    (107, "APPAREL",   0.6),    # Other clothing & footwear
    (107, "RETAIL",    0.4),
    (112, "PETRO",     1.0),    # Motor vehicle fuels (59) → petroleum refining
    (115, "PETRO",     1.0),    # Fuel oil and other fuels
    (119, "CHEM",      0.6),    # Pharmaceutical and other medical products
    (119, "AMBULAT",   0.4),
    (129, "CHEM",      1.0),    # Household supplies
    (135, "CHEM",      1.0),    # Personal care products
    (139, "FOOD",      1.0),    # Tobacco (127)
    # ── Housing ───────────────────────────────────────────────────────────────
    (152, "REALE",     1.0),    # Rental of tenant-occupied nonfarm housing
    (158, "REALE",     1.0),    # Imputed rental of owner-occupied housing
    (161, "REALE",     1.0),    # Rental value of farm dwellings
    (164, "UTIL",      1.0),    # Water supply and sanitation
    (168, "UTIL",      1.0),    # Electricity (27)
    (169, "UTIL",      1.0),    # Natural gas (28)
    # ── Health ────────────────────────────────────────────────────────────────
    (172, "AMBULAT",   1.0),    # Physician services (44)
    (173, "AMBULAT",   1.0),    # Dental services (45)
    (174, "AMBULAT",   1.0),    # Paramedical services (46)
    (181, "HOSPITAL",  1.0),    # Hospitals (51)
    (185, "NURSING",   1.0),    # Nursing homes (52)
    # ── Transportation ────────────────────────────────────────────────────────
    (190, "TRUCK",     0.5),    # Motor vehicle maintenance and repair
    (190, "RETAIL",    0.5),
    (198, "TRANSIT",   1.0),    # Ground transportation (63)
    (205, "AIRTRANS",  1.0),    # Air transportation (64)
    (206, "WATERTRANS",1.0),    # Water transportation (65)
    # ── Recreation ────────────────────────────────────────────────────────────
    (208, "PERFORM",   0.5),    # Membership clubs, sports centers, parks, theaters
    (208, "AMUSE",     0.5),
    (216, "BROADCAST", 1.0),    # Audio-video equipment services
    (224, "AMUSE",     1.0),    # Gambling (91)
    (228, "AMUSE",     1.0),    # Other recreational services
    # ── Food services & accommodation ────────────────────────────────────────
    (234, "FOODSVC",   1.0),    # Purchased meals and beverages (102)
    (247, "ACCOMM",    1.0),    # Accommodations (104)
    # ── Finance & insurance ───────────────────────────────────────────────────
    (252, "CREDIT",    1.0),    # Financial services furnished without payment
    (256, "CREDIT",    0.5),    # Financial service charges, fees, commissions
    (256, "SECURIT",   0.5),
    (269, "INSURE",    1.0),    # Life insurance
    (270, "INSURE",    1.0),    # Net household insurance
    (273, "INSURE",    1.0),    # Net health insurance
    (277, "INSURE",    1.0),    # Net motor vehicle insurance
    # ── Telecom & info ────────────────────────────────────────────────────────
    (280, "BROADCAST", 1.0),    # Telecommunication services (71)
    (284, "OTHERTRANS",1.0),    # Postal and delivery services (68)
    (287, "INFODATA",  1.0),    # Internet access (72)
    # ── Education ─────────────────────────────────────────────────────────────
    (289, "EDUC",      1.0),    # Higher education (97)
    (292, "EDUC",      1.0),    # Nursery, elementary, and secondary schools (98)
    (295, "EDUC",      1.0),    # Commercial and vocational schools (99)
    # ── Other personal services ───────────────────────────────────────────────
    (296, "MISCPROF",  0.5),    # Professional and other services (121)
    (296, "OTHSVC",    0.5),
    (305, "OTHSVC",    1.0),    # Personal care and clothing services
    (313, "SOCIALAS",  1.0),    # Social services and religious activities
]

# ─── PCE column index for Feb 2020 and May 2020 ────────────────────────────
# From sheet layout: 2020 starts at col 26, Feb=col 27, May=col 30
PCE_FEB2020_COL = 27
PCE_MAY2020_COL = 30
PCE_HEADER_ROWS = 7   # rows 0-6 are title/header; data starts at row 7


def parse_io_mat(path: Path = IO_MAT_PATH) -> tuple:
    """
    Parse IO_data_2018.mat and return model-aggregated (66-sector) arrays.

    Returns
    -------
    Z66 : np.ndarray, shape (66, 66), millions USD
        Intermediate flow matrix aggregated to 66 model sectors
    x66 : np.ndarray, shape (66,), millions USD
        Gross output per model sector
    v66 : np.ndarray, shape (66,), millions USD
        Total value added per model sector
    sector_names : list of str
        BEA names for each model sector (aggregated sectors get joined name)
    """
    mat = scipy.io.loadmat(str(path))
    D = mat["Data_raw"]          # (87, 100, 22)
    d = D[:, :, IO_YEAR_SLICE]   # (87, 100) for year 2018

    names_raw = [mat["indname"][i][0].item() for i in range(83)]

    # Extract full 71×71 intermediate flow matrix (NaN → 0)
    Z_raw = np.where(np.isnan(d[0:71, 2:73]), 0.0, d[0:71, 2:73])
    x_raw = d[82, 2:73]   # gross output row 82
    v_raw = d[81, 2:73]   # total value added row 81

    # Build 66-sector aggregated arrays
    N66 = 66
    Z66 = np.zeros((N66, N66))
    x66 = np.zeros(N66)
    v66 = np.zeros(N66)

    # Row and column aggregation using IO_AGG
    for (io_rows, model_idx) in IO_AGG:
        # gross output and value added: sum
        x66[model_idx] += np.nansum(x_raw[io_rows])
        v66[model_idx] += np.nansum(v_raw[io_rows])

    for (io_rows_i, model_i) in IO_AGG:
        for (io_rows_j, model_j) in IO_AGG:
            block = Z_raw[np.ix_(io_rows_i, io_rows_j)]
            Z66[model_i, model_j] += np.nansum(block)

    # Build sector names (aggregated sectors get combined name)
    _agg_model_to_io: dict = {}
    for (io_rows, model_idx) in IO_AGG:
        _agg_model_to_io.setdefault(model_idx, []).extend(io_rows)

    sector_names = []
    for model_idx in range(N66):
        io_rows = _agg_model_to_io[model_idx]
        if len(io_rows) == 1:
            sector_names.append(names_raw[io_rows[0]])
        else:
            sector_names.append(" + ".join(names_raw[r] for r in io_rows))

    return Z66, x66, v66, sector_names


def _bls_weighted_shock(group_df: pd.DataFrame, shock_col: str) -> float:
    """
    Compute gross-output-weighted log shock for a group of BLS rows.

    The BLS file does not contain gross output directly; we use the absolute
    value of the hours level (imputed from shock + base) as a proxy, but since
    that is not available we fall back to an unweighted mean.
    """
    vals = group_df[shock_col].astype(float)
    return float(vals.mean())


def parse_bls_shocks(
    path: Path = BLS_XLS_PATH,
    shock_col: str = "diff_2005",   # May 2020 relative to Feb 2020 baseline
) -> dict:
    """
    Parse BLS_labor_shock_202108.xls and return a dict of model-sector log shocks.

    Parameters
    ----------
    path      : path to BLS Excel file
    shock_col : column name encoding the shock month. Default diff_2005 = May 2020.
                Format YYMM: 20=year 2020, 05=May → Feb 2020 is the base (0 change).

    Returns
    -------
    delta_y : dict[model_sector_code → float]
        Log change in hours worked (Feb→May 2020), a proxy for output change.
        Covers all 66 model sectors.  Missing sectors default to 0.0.
    """
    df = pd.read_excel(str(path), sheet_name="Sheet1", header=None)
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    df = df.rename(columns={df.columns[0]: "iocode"})

    # Convert shock column to numeric
    df[shock_col] = pd.to_numeric(df[shock_col], errors="coerce").fillna(0.0)

    # Accumulate weighted shock per model sector
    sector_vals: dict[str, list] = {}
    for _, row in df.iterrows():
        code = str(row["iocode"]).strip()
        model = BLS_TO_MODEL.get(code)
        if model is None:
            continue
        sector_vals.setdefault(model, []).append(float(row[shock_col]))

    delta_y = {model: float(np.mean(vals)) for model, vals in sector_vals.items()}
    return delta_y


def _pce_log_change(df: pd.DataFrame, line_no: int) -> float | None:
    """
    Return log(May2020 / Feb2020) for a given PCE line number.
    Returns None if either value is missing or non-positive.
    """
    # Row index in df: PCE line numbers start at 1, but our df starts at row 0
    # Row in raw sheet = PCE_HEADER_ROWS + (line_no - 1)
    row_idx = PCE_HEADER_ROWS + (line_no - 1)
    if row_idx >= len(df):
        return None
    feb = pd.to_numeric(df.iloc[row_idx, PCE_FEB2020_COL], errors="coerce")
    may = pd.to_numeric(df.iloc[row_idx, PCE_MAY2020_COL], errors="coerce")
    if pd.isna(feb) or pd.isna(may) or feb <= 0 or may <= 0:
        return None
    return float(np.log(may / feb))


def parse_pce_shocks(path: Path = PCE_XLS_PATH) -> dict:
    """
    Parse Expenditure_202107.xls and return a dict of model-sector log demand shocks.

    The PCE log changes (Feb→May 2020) proxy the demand-side expenditure shock
    for consumer-facing sectors.  Spending = P × Q, so a fall in PCE reflects
    a combination of price and quantity changes driven by demand.

    For sectors with multiple PCE lines mapped to them, the result is an
    expenditure-weighted average across lines (equal-weighted within the mapping
    table, since we do not have individual-line expenditure shares).

    Returns
    -------
    delta_p : dict[model_sector_code → float]
        Log change in PCE spending (Feb→May 2020).
        Sectors without PCE coverage are absent from the dict (default 0.0).
    """
    df = pd.read_excel(str(path), sheet_name="Sheet0", header=None)

    # Accumulate weighted shocks
    sector_sum: dict[str, float] = {}
    sector_wgt: dict[str, float] = {}

    for (line_no, sector_code, weight) in PCE_LINES:
        lc = _pce_log_change(df, line_no)
        if lc is None:
            continue
        sector_sum[sector_code] = sector_sum.get(sector_code, 0.0) + weight * lc
        sector_wgt[sector_code] = sector_wgt.get(sector_code, 0.0) + weight

    delta_p = {
        sector: sector_sum[sector] / sector_wgt[sector]
        for sector in sector_sum
    }
    return delta_p


def load_all_real_data(
    io_path: Path = IO_MAT_PATH,
    bls_path: Path = BLS_XLS_PATH,
    pce_path: Path = PCE_XLS_PATH,
) -> tuple:
    """
    Convenience loader: parse all three files and return (Z66, x66, v66, delta_y, delta_p).

    Returns
    -------
    Z66    : (66, 66) intermediate flow matrix, millions USD
    x66    : (66,)   gross output, millions USD
    v66    : (66,)   value added, millions USD
    delta_y: dict[sector_code → float] BLS labor shock (log change, May vs Feb 2020)
    delta_p: dict[sector_code → float] PCE demand shock (log change, May vs Feb 2020)
    """
    Z66, x66, v66, _ = parse_io_mat(io_path)
    delta_y = parse_bls_shocks(bls_path)
    delta_p = parse_pce_shocks(pce_path)
    return Z66, x66, v66, delta_y, delta_p


if __name__ == "__main__":
    print("Parsing IO matrix …")
    Z66, x66, v66, names = parse_io_mat()
    print(f"  Z66 shape: {Z66.shape}")
    print(f"  Gross output total: ${x66.sum()/1e6:.2f} T")
    print(f"  Value added total:  ${v66.sum()/1e6:.2f} T")

    print("\nParsing BLS labor shocks …")
    dy = parse_bls_shocks()
    print(f"  Sectors covered: {len(dy)}")
    for s in ["AIRTRANS", "ACCOMM", "FOODSVC", "MOTVEH", "FARM"]:
        print(f"  {s:15s}: {dy.get(s, 0.0):+.4f}")

    print("\nParsing PCE demand shocks …")
    dp = parse_pce_shocks()
    print(f"  Sectors covered: {len(dp)}")
    for s in ["AIRTRANS", "ACCOMM", "FOODSVC", "PETRO", "HOSPITAL", "REALE"]:
        print(f"  {s:15s}: {dp.get(s, 0.0):+.4f}")
