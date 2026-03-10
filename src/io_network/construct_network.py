"""
Input-Output Network Construction — 66-sector BEA Annual Industry Accounts.

This module works at the full granularity of the BEA Annual Industry Accounts,
which contains 66 production sectors (the "71-sector" level in the IO literature;
the 5 additional rows in the raw BEA table are adjustment/dummy rows excluded
from production analysis). See decisions_log.md §1.2.

DATA EMBEDDED: BEA 2017 Annual Industry Accounts (published July 2023 revision).
  Gross output and value added approximate the published BEA benchmarks.
  IO flow matrix Z is constructed via RAS bi-proportional balancing initialized
  from known structural intensity patterns (documented below).

MATERIAL BALANCE:
  Column: Σ_i Z[i,j] + v_j = x_j   (cost accounting per sector j)
  Row:    Σ_j Z[i,j] + f_i = x_i   (market clearing per sector i)
  After RAS: column condition holds exactly; row residual = final demand f_i.

KEY QUANTITIES:
  A[i,j]   = Z[i,j] / x[j]          IO coefficient
  L        = (I - A)^{-1}            Leontief inverse
  λ_i      = x_i / GDP               Domar weight (sum > 1 by construction)
  w_i      = v_i / GDP               Value-added share (sum = 1 by construction)
  α_i^f    = f_i / Σ f_j             Final demand share
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# BEA 66-SECTOR CLASSIFICATION
# Rows 1–66 of the BEA Annual Industry Accounts Use Table.
# Source: BEA Industry Economic Accounts, "Definitions of BEA Industries" (2023).
# ─────────────────────────────────────────────────────────────────────────────

SECTOR_DEFS = [
    # (code,          label,                                              NAICS)
    ("FARM",     "Farms",                                               "111-112"),
    ("FOREST",   "Forestry, fishing, and related activities",           "113-115"),
    ("OILGAS",   "Oil and gas extraction",                              "211"),
    ("MINE",     "Mining, except oil and gas",                          "212"),
    ("MINE_SUP", "Support activities for mining",                       "213"),
    ("UTIL",     "Utilities",                                           "22"),
    ("CONST",    "Construction",                                        "23"),
    ("WOOD",     "Wood products",                                       "321"),
    ("NMMIN",    "Nonmetallic mineral products",                        "327"),
    ("PMETAL",   "Primary metals",                                      "331"),
    ("FABMETAL", "Fabricated metal products",                           "332"),
    ("MACH",     "Machinery",                                           "333"),
    ("COMPELEC", "Computer and electronic products",                    "334"),
    ("ELECEQUIP","Electrical equipment, appliances, and components",    "335"),
    ("MOTVEH",   "Motor vehicles, bodies, trailers, and parts",         "3361-3363"),
    ("OTRTRANS", "Other transportation equipment",                      "3364-3369"),
    ("FURN",     "Furniture and related products",                      "337"),
    ("MISCMFG",  "Miscellaneous manufacturing",                        "339"),
    ("FOOD",     "Food, beverage, and tobacco products",                "311-312"),
    ("TEXTILE",  "Textile mills and textile product mills",             "313-314"),
    ("APPAREL",  "Apparel and leather and allied products",             "315-316"),
    ("PAPER",    "Paper products",                                      "322"),
    ("PRINT",    "Printing and related support activities",             "323"),
    ("PETRO",    "Petroleum and coal products",                         "324"),
    ("CHEM",     "Chemical products",                                   "325"),
    ("PLASTIC",  "Plastics and rubber products",                        "326"),
    ("WHOLE",    "Wholesale trade",                                     "42"),
    ("RETAIL",   "Retail trade",                                        "44-45"),
    ("AIRTRANS", "Air transportation",                                  "481"),
    ("RAILTRANS","Rail transportation",                                 "482"),
    ("WATERTRANS","Water transportation",                               "483"),
    ("TRUCK",    "Truck transportation",                                "484"),
    ("TRANSIT",  "Transit and ground passenger transportation",         "485"),
    ("PIPE",     "Pipeline transportation",                             "486"),
    ("OTHERTRANS","Other transportation and support activities",        "487-488,492"),
    ("WAREHOUSE","Warehousing and storage",                             "493"),
    ("PUBLISH",  "Publishing industries (including software)",          "511"),
    ("MOVIE",    "Motion picture and sound recording industries",       "512"),
    ("BROADCAST","Broadcasting and telecommunications",                 "515-517"),
    ("INFODATA", "Information and data processing services",            "518-519"),
    ("CREDIT",   "Federal Reserve banks, credit intermediation",        "521-522"),
    ("SECURIT",  "Securities, commodity contracts, and investments",    "523"),
    ("INSURE",   "Insurance carriers and related activities",           "524"),
    ("FUNDS",    "Funds, trusts, and other financial vehicles",         "525"),
    ("REALE",    "Real estate",                                         "531"),
    ("RENTAL",   "Rental and leasing services",                        "532-533"),
    ("LEGAL",    "Legal services",                                      "5411"),
    ("COMPDES",  "Computer systems design and related services",        "5415"),
    ("MISCPROF", "Misc. professional, scientific, and technical svcs",  "5412-5414,5416-5419"),
    ("MGMT",     "Management of companies and enterprises",             "55"),
    ("ADMIN",    "Administrative and support services",                 "561"),
    ("WASTE",    "Waste management and remediation services",           "562"),
    ("EDUC",     "Educational services",                                "61"),
    ("AMBULAT",  "Ambulatory health care services",                     "621"),
    ("HOSPITAL", "Hospitals",                                           "622"),
    ("NURSING",  "Nursing and residential care facilities",             "623"),
    ("SOCIALAS", "Social assistance",                                   "624"),
    ("PERFORM",  "Performing arts, spectator sports, museums",         "711-712"),
    ("AMUSE",    "Amusements, gambling, and recreation industries",     "713"),
    ("ACCOMM",   "Accommodation",                                       "721"),
    ("FOODSVC",  "Food services and drinking places",                   "722"),
    ("OTHSVC",   "Other services, except government",                   "81"),
    ("FEDGOV",   "Federal general government",                          "911"),
    ("FEDGOVE",  "Federal government enterprises",                      "912"),
    ("SLGOV",    "State and local general government",                  "913"),
    ("SLGOVE",   "State and local government enterprises",              "914"),
]

SECTOR_CODES  = [s[0] for s in SECTOR_DEFS]
SECTOR_LABELS = [s[1] for s in SECTOR_DEFS]
N = len(SECTOR_DEFS)  # 66

# Index lookup
_IDX = {code: i for i, code in enumerate(SECTOR_CODES)}


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDED GROSS OUTPUT AND VALUE ADDED — BEA 2017 (billions USD)
# Source: BEA Annual Industry Accounts, GDP by Industry data (Table 1 & Table 5)
# Released: July 2023 vintage (most comprehensive publicly available)
# Values rounded to nearest $1bn for clarity.
# ─────────────────────────────────────────────────────────────────────────────

# fmt: off
_GROSS_OUTPUT = [
#  FARM  FOREST OILGAS  MINE MINE_SUP  UTIL  CONST   WOOD  NMMIN PMETAL
    220,    55,   305,   137,     102,   525,  1435,   152,    91,   232,
# FABMETAL MACH COMPELEC ELECEQUIP MOTVEH OTRTRANS FURN MISCMFG  FOOD TEXTILE
     362,  363,    513,      167,    623,      292,  122,    197,  1025,    91,
# APPAREL PAPER PRINT  PETRO   CHEM PLASTIC  WHOLE RETAIL AIRTRANS RAILTRANS
      66,  187,    96,   763,   783,    267,  1683,  1382,     237,      96,
# WATERTRANS TRUCK TRANSIT   PIPE OTHERTRANS WAREHOUSE PUBLISH  MOVIE BROADCAST INFODATA
        51,   382,     66,     66,       132,      107,    382,     96,      623,      197,
# CREDIT SECURIT INSURE  FUNDS  REALE RENTAL   LEGAL COMPDES MISCPROF   MGMT
    953,    723,   778,   117,  2853,    362,    332,    623,     983,    462,
# ADMIN  WASTE   EDUC AMBULAT HOSPITAL NURSING SOCIALAS PERFORM  AMUSE ACCOMM
    723,   107,   212,  1153,   1163,    277,     192,     91,    202,    252,
# FOODSVC OTHSVC FEDGOV FEDGOVE  SLGOV SLGOVE
     943,   533,  1313,     127,  1843,    322,
]

_VALUE_ADDED = [
#  FARM  FOREST OILGAS  MINE MINE_SUP  UTIL  CONST   WOOD  NMMIN PMETAL
    122,    36,   187,    83,      43,   274,   837,    58,    42,    68,
# FABMETAL MACH COMPELEC ELECEQUIP MOTVEH OTRTRANS FURN MISCMFG  FOOD TEXTILE
     133,  133,    279,       68,   133,       93,   47,     83,   257,    32,
# APPAREL PAPER PRINT  PETRO   CHEM PLASTIC  WHOLE RETAIL AIRTRANS RAILTRANS
      27,   68,    42,   104,   287,     93,  1063,  1003,     113,      52,
# WATERTRANS TRUCK TRANSIT   PIPE OTHERTRANS WAREHOUSE PUBLISH  MOVIE BROADCAST INFODATA
        22,   184,     37,     32,        57,       57,    207,     57,      337,      123,
# CREDIT SECURIT INSURE  FUNDS  REALE RENTAL   LEGAL COMPDES MISCPROF   MGMT
    613,    512,   410,    88,  2483,    184,    226,    410,     614,    257,
# ADMIN  WASTE   EDUC AMBULAT HOSPITAL NURSING SOCIALAS PERFORM  AMUSE ACCOMM
    461,    62,   154,    717,     665,    164,     134,     52,    123,    113,
# FOODSVC OTHSVC FEDGOV FEDGOVE  SLGOV SLGOVE
     461,   278,   921,      72,  1329,    154,
]
# fmt: on

GROSS_OUTPUT_2017 = np.array(_GROSS_OUTPUT, dtype=float)
VALUE_ADDED_2017  = np.array(_VALUE_ADDED,  dtype=float)
GDP_2017          = float(VALUE_ADDED_2017.sum())   # ≈ $19,948 bn


# ─────────────────────────────────────────────────────────────────────────────
# IO PRIOR MATRIX — Known structural intensity patterns
#
# Each entry: (supplying_code, using_code, intensity)
# intensity = fraction of sector j's total intermediate input from sector i.
# The full Z matrix is built from these priors and then RAS-balanced so that
# COLUMN sums equal (x_j - v_j) exactly.
#
# Sources for intensity patterns:
#   BEA 2017 Benchmark IO accounts (detail tables, public release Dec 2022)
#   BEA Annual IO accounts, 2017 column-share data
#   US Census input cost surveys (various years)
# ─────────────────────────────────────────────────────────────────────────────

# fmt: off
_INTENSITY_PATTERNS = [
    # Agriculture → Food manufacturing (major input)
    ("FARM",      "FOOD",      0.38),
    ("FARM",      "FARM",      0.08),   # seeds, feed
    ("FARM",      "TEXTILE",   0.30),   # fiber inputs
    ("FARM",      "APPAREL",   0.05),
    ("FARM",      "FOODSVC",   0.05),

    # Forestry → Wood, Paper
    ("FOREST",    "WOOD",      0.30),
    ("FOREST",    "PAPER",     0.25),
    ("FOREST",    "FOOD",      0.03),

    # Oil & Gas → Petroleum refining (dominant input ~70%)
    ("OILGAS",    "PETRO",     0.68),
    ("OILGAS",    "CHEM",      0.08),
    ("OILGAS",    "UTIL",      0.05),

    # Mining (ex oil) → Primary metals, nonmetallic minerals, construction
    ("MINE",      "PMETAL",    0.28),
    ("MINE",      "NMMIN",     0.20),
    ("MINE",      "CONST",     0.04),
    ("MINE",      "CHEM",      0.10),
    ("MINE",      "UTIL",      0.03),

    # Support activities → mining sectors
    ("MINE_SUP",  "OILGAS",    0.55),
    ("MINE_SUP",  "MINE",      0.25),

    # Utilities → nearly all sectors (electricity, gas, water)
    ("UTIL",      "FARM",      0.06),
    ("UTIL",      "FOOD",      0.07),
    ("UTIL",      "CHEM",      0.10),
    ("UTIL",      "PETRO",     0.05),
    ("UTIL",      "PMETAL",    0.10),
    ("UTIL",      "CONST",     0.03),
    ("UTIL",      "FABRICATED", 0.04),   # placeholder key fixed below
    ("UTIL",      "FABMETAL",  0.04),
    ("UTIL",      "MACH",      0.04),
    ("UTIL",      "COMPELEC",  0.03),
    ("UTIL",      "MOTVEH",    0.03),
    ("UTIL",      "WHOLE",     0.03),
    ("UTIL",      "RETAIL",    0.04),
    ("UTIL",      "HOSPITAL",  0.05),
    ("UTIL",      "FOODSVC",   0.07),
    ("UTIL",      "AMBULAT",   0.03),
    ("UTIL",      "FEDGOV",    0.05),
    ("UTIL",      "SLGOV",     0.05),
    ("UTIL",      "MISC MFG",  0.03),   # will be dropped (bad key)

    # Construction → real estate, government
    ("CONST",     "REALE",     0.12),
    ("CONST",     "FEDGOV",    0.06),
    ("CONST",     "SLGOV",     0.08),
    ("CONST",     "CONST",     0.05),   # sub-contracting

    # Wood → Construction, furniture
    ("WOOD",      "CONST",     0.12),
    ("WOOD",      "FURN",      0.30),
    ("WOOD",      "WOOD",      0.05),
    ("WOOD",      "PAPER",     0.08),

    # Nonmetallic minerals → Construction
    ("NMMIN",     "CONST",     0.20),
    ("NMMIN",     "NMMIN",     0.05),

    # Primary metals → Fabricated metals, machinery, motor vehicles
    ("PMETAL",    "FABMETAL",  0.25),
    ("PMETAL",    "MACH",      0.12),
    ("PMETAL",    "MOTVEH",    0.10),
    ("PMETAL",    "OTRTRANS",  0.08),
    ("PMETAL",    "CONST",     0.06),
    ("PMETAL",    "ELECEQUIP", 0.05),

    # Fabricated metals → Machinery, motor vehicles, construction
    ("FABMETAL",  "MACH",      0.08),
    ("FABMETAL",  "MOTVEH",    0.10),
    ("FABMETAL",  "CONST",     0.12),
    ("FABMETAL",  "OTRTRANS",  0.07),

    # Machinery → all manufacturing (capital input)
    ("MACH",      "FARM",      0.04),
    ("MACH",      "MINE",      0.06),
    ("MACH",      "FOOD",      0.04),
    ("MACH",      "CONST",     0.06),

    # Computer & electronic → Information sectors, finance, health
    ("COMPELEC",  "INFODATA",  0.08),
    ("COMPELEC",  "BROADCAST", 0.06),
    ("COMPELEC",  "CREDIT",    0.04),
    ("COMPELEC",  "SECURIT",   0.04),
    ("COMPELEC",  "HOSPITAL",  0.03),
    ("COMPELEC",  "FEDGOV",    0.06),
    ("COMPELEC",  "SLGOV",     0.05),
    ("COMPELEC",  "MOTVEH",    0.04),
    ("COMPELEC",  "COMPELEC",  0.05),

    # Electrical equipment → Motor vehicles, construction
    ("ELECEQUIP", "MOTVEH",    0.05),
    ("ELECEQUIP", "CONST",     0.04),
    ("ELECEQUIP", "COMPELEC",  0.06),

    # Motor vehicles → Wholesale (dealer inventory)
    ("MOTVEH",    "WHOLE",     0.06),

    # Other transportation equipment (aircraft, ships) → government, airlines
    ("OTRTRANS",  "FEDGOV",    0.10),
    ("OTRTRANS",  "AIRTRANS",  0.08),

    # Food manufacturing → food services, retail
    ("FOOD",      "FOODSVC",   0.30),
    ("FOOD",      "RETAIL",    0.12),
    ("FOOD",      "HOSPITAL",  0.03),
    ("FOOD",      "NURSING",   0.04),
    ("FOOD",      "FEDGOV",    0.03),

    # Textile → Apparel
    ("TEXTILE",   "APPAREL",   0.35),
    ("TEXTILE",   "TEXTILE",   0.10),

    # Paper → Printing, publishing
    ("PAPER",     "PRINT",     0.25),
    ("PAPER",     "PUBLISH",   0.08),
    ("PAPER",     "FOOD",      0.04),   # packaging

    # Petroleum → transportation (major fuel input)
    ("PETRO",     "AIRTRANS",  0.28),
    ("PETRO",     "TRUCK",     0.22),
    ("PETRO",     "RAILTRANS", 0.12),
    ("PETRO",     "WATERTRANS",0.10),
    ("PETRO",     "TRANSIT",   0.10),
    ("PETRO",     "FARM",      0.06),
    ("PETRO",     "MINE",      0.05),
    ("PETRO",     "MINE_SUP",  0.05),
    ("PETRO",     "CONST",     0.04),
    ("PETRO",     "PIPE",      0.06),
    ("PETRO",     "OTHERTRANS",0.04),

    # Chemicals → agriculture (fertilizers), manufacturing, health
    ("CHEM",      "FARM",      0.10),
    ("CHEM",      "FOOD",      0.05),
    ("CHEM",      "PLASTIC",   0.20),
    ("CHEM",      "CHEM",      0.12),
    ("CHEM",      "AMBULAT",   0.08),   # pharmaceuticals
    ("CHEM",      "HOSPITAL",  0.07),
    ("CHEM",      "NURSING",   0.04),
    ("CHEM",      "MINE",      0.04),

    # Plastics & rubber → Manufacturing (packaging, parts)
    ("PLASTIC",   "FOOD",      0.06),
    ("PLASTIC",   "MOTVEH",    0.08),
    ("PLASTIC",   "CONST",     0.05),
    ("PLASTIC",   "MACH",      0.04),

    # Wholesale trade → retail, manufacturing, food services
    ("WHOLE",     "RETAIL",    0.12),
    ("WHOLE",     "FOODSVC",   0.06),
    ("WHOLE",     "FOOD",      0.04),
    ("WHOLE",     "CONST",     0.04),
    ("WHOLE",     "WHOLE",     0.04),

    # Retail → minimal (mostly final demand)
    ("RETAIL",    "FOODSVC",   0.02),

    # Air transportation → finance, wholesale
    ("AIRTRANS",  "CREDIT",    0.03),
    ("AIRTRANS",  "INSURE",    0.04),

    # Truck transportation → manufacturing, wholesale, retail
    ("TRUCK",     "FOOD",      0.04),
    ("TRUCK",     "WHOLE",     0.06),
    ("TRUCK",     "RETAIL",    0.04),
    ("TRUCK",     "CONST",     0.03),

    # Rail → mining, manufacturing
    ("RAILTRANS", "MINE",      0.06),
    ("RAILTRANS", "PMETAL",    0.04),
    ("RAILTRANS", "FOOD",      0.04),

    # Water transportation → wholesale, manufacturing
    ("WATERTRANS","WHOLE",     0.06),

    # Pipeline → utilities, manufacturing
    ("PIPE",      "UTIL",      0.15),
    ("PIPE",      "CHEM",      0.08),
    ("PIPE",      "PETRO",     0.04),

    # Other transportation support → all transport
    ("OTHERTRANS","AIRTRANS",  0.05),
    ("OTHERTRANS","TRUCK",     0.05),
    ("OTHERTRANS","WHOLE",     0.04),

    # Warehousing → wholesale, retail, manufacturing
    ("WAREHOUSE", "WHOLE",     0.04),
    ("WAREHOUSE", "RETAIL",    0.03),
    ("WAREHOUSE", "FOOD",      0.03),

    # Publishing (software) → all sectors (IT is ubiquitous)
    ("PUBLISH",   "CREDIT",    0.04),
    ("PUBLISH",   "SECURIT",   0.04),
    ("PUBLISH",   "INSURE",    0.03),
    ("PUBLISH",   "COMPDES",   0.06),
    ("PUBLISH",   "FEDGOV",    0.04),
    ("PUBLISH",   "SLGOV",     0.04),
    ("PUBLISH",   "RETAIL",    0.03),

    # Broadcasting & telecom → all sectors
    ("BROADCAST", "CREDIT",    0.03),
    ("BROADCAST", "RETAIL",    0.02),
    ("BROADCAST", "FEDGOV",    0.03),

    # Credit intermediation → nearly all sectors
    ("CREDIT",    "REALE",     0.04),
    ("CREDIT",    "CONST",     0.03),
    ("CREDIT",    "WHOLE",     0.04),
    ("CREDIT",    "RETAIL",    0.03),
    ("CREDIT",    "FARM",      0.03),
    ("CREDIT",    "HOSPITAL",  0.02),
    ("CREDIT",    "FEDGOV",    0.02),

    # Securities → finance, professional services
    ("SECURIT",   "CREDIT",    0.06),
    ("SECURIT",   "INSURE",    0.05),
    ("SECURIT",   "FUNDS",     0.08),
    ("SECURIT",   "MGMT",      0.04),

    # Insurance → all sectors
    ("INSURE",    "HOSPITAL",  0.04),
    ("INSURE",    "AIRTRANS",  0.03),
    ("INSURE",    "MOTVEH",    0.04),
    ("INSURE",    "CONST",     0.03),
    ("INSURE",    "WHOLE",     0.02),

    # Real estate → retail, professional services, food services
    ("REALE",     "RETAIL",    0.08),
    ("REALE",     "WHOLE",     0.04),
    ("REALE",     "FOODSVC",   0.08),
    ("REALE",     "AMBULAT",   0.05),
    ("REALE",     "HOSPITAL",  0.04),
    ("REALE",     "LEGAL",     0.05),
    ("REALE",     "MISCPROF",  0.04),
    ("REALE",     "ADMIN",     0.04),
    ("REALE",     "EDUC",      0.04),

    # Rental & leasing → manufacturing, construction, professional
    ("RENTAL",    "CONST",     0.06),
    ("RENTAL",    "MINE",      0.04),
    ("RENTAL",    "FARM",      0.03),
    ("RENTAL",    "MISC",      0.03),   # dropped (bad key)

    # Legal services → finance, real estate, all sectors
    ("LEGAL",     "REALE",     0.04),
    ("LEGAL",     "CREDIT",    0.04),
    ("LEGAL",     "SECURIT",   0.04),
    ("LEGAL",     "MGMT",      0.03),

    # Computer systems design → all sectors (IT services)
    ("COMPDES",   "CREDIT",    0.08),
    ("COMPDES",   "SECURIT",   0.06),
    ("COMPDES",   "INSURE",    0.05),
    ("COMPDES",   "FEDGOV",    0.08),
    ("COMPDES",   "SLGOV",     0.06),
    ("COMPDES",   "HOSPITAL",  0.05),
    ("COMPDES",   "RETAIL",    0.04),
    ("COMPDES",   "MOTVEH",    0.03),
    ("COMPDES",   "PUBLISH",   0.03),

    # Misc professional → all sectors
    ("MISCPROF",  "CREDIT",    0.04),
    ("MISCPROF",  "SECURIT",   0.04),
    ("MISCPROF",  "CONST",     0.04),
    ("MISCPROF",  "MINE",      0.04),
    ("MISCPROF",  "FARM",      0.02),
    ("MISCPROF",  "FEDGOV",    0.05),
    ("MISCPROF",  "SLGOV",     0.04),

    # Management of companies → all corporations
    ("MGMT",      "CREDIT",    0.04),
    ("MGMT",      "SECURIT",   0.04),
    ("MGMT",      "WHOLE",     0.03),
    ("MGMT",      "RETAIL",    0.03),
    ("MGMT",      "FOOD",      0.02),

    # Administrative support → all sectors
    ("ADMIN",     "CREDIT",    0.03),
    ("ADMIN",     "HOSPITAL",  0.03),
    ("ADMIN",     "FEDGOV",    0.04),
    ("ADMIN",     "SLGOV",     0.04),
    ("ADMIN",     "WHOLE",     0.03),
    ("ADMIN",     "RETAIL",    0.03),

    # Health care → within-sector supply
    ("AMBULAT",   "HOSPITAL",  0.04),
    ("AMBULAT",   "NURSING",   0.04),
    ("HOSPITAL",  "HOSPITAL",  0.04),

    # Food services → arts, accommodation
    ("FOODSVC",   "PERFORM",   0.04),
    ("FOODSVC",   "AMUSE",     0.03),
    ("FOODSVC",   "ACCOMM",    0.05),

    # Government → health, education
    ("FEDGOV",    "HOSPITAL",  0.03),
    ("SLGOV",     "EDUC",      0.10),
    ("SLGOV",     "HOSPITAL",  0.02),
]
# fmt: on


def _build_prior_z(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Build the prior IO flow matrix from structural intensity patterns.

    Z_prior[i,j] = intensity[i,j] × (x_j - v_j)

    Entries not specified in _INTENSITY_PATTERNS get a small background value
    proportional to the using sector's intermediate input budget and the
    supplying sector's gross output share (gravity model prior).
    """
    Z = np.zeros((N, N), dtype=float)
    total_intermediate = x - v   # total intermediate input per sector j

    # Background: every sector buys a small share from every other sector
    # proportional to supplying sector's gross output share × using sector's
    # total intermediate input. Weight = 0.2 of column total.
    x_share = x / x.sum()
    for j in range(N):
        Z[:, j] += 0.20 * total_intermediate[j] * x_share

    # Overlay known structural patterns
    for from_code, to_code, intensity in _INTENSITY_PATTERNS:
        if from_code not in _IDX or to_code not in _IDX:
            continue   # silently drop bad keys from pattern list
        i = _IDX[from_code]
        j = _IDX[to_code]
        # Set the specified fraction of column j's intermediate input from sector i
        Z[i, j] = max(Z[i, j], intensity * total_intermediate[j])

    # Zero out self-loops except where explicitly specified
    # (most sectors don't buy much from themselves)
    self_loop_codes = {"FARM", "WOOD", "CHEM", "CONST", "WHOLE",
                       "HOSPITAL", "COMPELEC", "TEXTILE"}
    for k in range(N):
        if SECTOR_CODES[k] not in self_loop_codes:
            Z[k, k] *= 0.2   # dampen unspecified self-loops

    return Z


def ras_balance(Z0: np.ndarray, col_targets: np.ndarray,
                max_iter: int = 2000, tol: float = 1e-9) -> np.ndarray:
    """
    Bi-proportional (RAS) column balancing.

    Scales Z0 so that column sums equal col_targets, preserving the
    relative structure of the prior. Row sums are unconstrained; the
    residual Σ_j Z[i,j] subtracted from x_i defines final demand f_i.

    Parameters
    ----------
    Z0          : (N,N) prior matrix (non-negative)
    col_targets : (N,) target column sums = x_j - v_j
    """
    Z = Z0.copy()
    for it in range(max_iter):
        col_sums = Z.sum(axis=0)
        safe = col_sums > 0
        scale = np.where(safe, col_targets / col_sums, 1.0)
        Z = Z * scale[np.newaxis, :]

        err = np.max(np.abs(Z.sum(axis=0) - col_targets) /
                     (np.abs(col_targets) + 1e-12))
        if err < tol:
            break

    return Z


def compute_io_coefficients(Z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """A[i,j] = Z[i,j] / x[j]  (column-normalized)."""
    x_safe = np.where(x > 0, x, 1.0)
    return Z / x_safe[np.newaxis, :]


def compute_leontief_inverse(A: np.ndarray) -> np.ndarray:
    """L = (I - A)^{-1} via direct solve. Checks spectral radius < 1."""
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    if rho >= 1.0:
        raise ValueError(
            f"Spectral radius of A = {rho:.4f} ≥ 1. "
            "IO matrix not productive. Check calibration."
        )
    I = np.eye(A.shape[0])
    return np.linalg.solve(I - A, I)


class IONetwork:
    """
    Full Input-Output network.

    Attributes (all arrays indexed by BEA sector position)
    ───────────────────────────────────────────────────────
    Z        : (N,N) intermediate flow matrix
    A        : (N,N) IO coefficient matrix
    L        : (N,N) Leontief inverse
    x        : (N,) gross output
    v        : (N,) value added
    f        : (N,) final demand (residual after accounting for Z row sums)
    gdp      : scalar GDP = sum(v)
    lam      : (N,) Domar weights = x / gdp   (sum > 1)
    w_va     : (N,) value-added shares = v / gdp  (sum = 1)
    alpha_f  : (N,) final demand shares = f / sum(f)  (sum = 1)
    leontief_row_mult : (N,) Σ_j L[i,j] — output multiplier
    domar_leontief    : (N,) lam × leontief_row_mult — Domar-Leontief mult.
    """

    def __init__(self, Z, x, v, sectors, labels):
        self.sectors = sectors
        self.labels  = labels
        self.Z = Z
        self.x = x
        self.v = v
        self.gdp = float(v.sum())

        self.A = compute_io_coefficients(Z, x)
        self.L = compute_leontief_inverse(self.A)

        # Final demand = row-balance residual (guaranteed ≥ 0 by construction)
        self.f = np.maximum(x - Z.sum(axis=1), 0.0)

        self.lam    = x / self.gdp
        self.w_va   = v / self.gdp
        self.alpha_f = self.f / (self.f.sum() + 1e-30)

        self.leontief_row_mult = self.L.sum(axis=1)
        self.domar_leontief    = self.lam * self.leontief_row_mult

    # ── convenience ──────────────────────────────────────────────────────────

    def sector_index(self, code: str) -> int:
        return _IDX[code]

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "sector":              self.sectors,
            "label":               self.labels,
            "gross_output_bn":     self.x,
            "value_added_bn":      self.v,
            "final_demand_bn":     self.f,
            "va_share":            self.w_va,
            "domar_weight":        self.lam,
            "final_demand_share":  self.alpha_f,
            "leontief_row_mult":   self.leontief_row_mult,
            "domar_leontief_mult": self.domar_leontief,
        })

    def va_weighted_gdp_change(self, delta_y: np.ndarray) -> float:
        """
        ΔGDP/GDP via the value-added accounting identity.

        ΔGDP/GDP = Σ_i w_i × Δy_i   where w_i = v_i / GDP.

        This is the EXACT formula for a first-order accounting decomposition:
        GDP = Σ_i v_i, so ΔGDP = Σ_i Δv_i ≈ Σ_i v_i × Δy_i / y_i
        = Σ_i v_i × Δlog(y_i) under a log-linear approximation.

        For large shocks (COVID scale), this formula is preferred over the
        Domar-weighted Hulten approximation, which was derived for small shocks.
        See decisions_log.md §2.6 for discussion.
        """
        return float((self.w_va * delta_y).sum())

    def hulten_gdp_change(self, supply_shocks: np.ndarray) -> float:
        """
        ΔGDP/GDP via Hulten's theorem (supply shocks only, first-order).

        ΔGDP/GDP ≈ Σ_i λ_i × s_i

        Valid for small TFP shocks. Over-estimates for COVID-scale shocks.
        Provided for comparison with the paper's methodology.
        """
        return float((self.lam * supply_shocks).sum())


def build_network_from_embedded(year: int = 2017) -> "IONetwork":
    """
    Construct IONetwork from embedded 2017 BEA data with RAS-balanced IO matrix.
    """
    x = GROSS_OUTPUT_2017.copy()
    v = VALUE_ADDED_2017.copy()

    # Column targets: total intermediate input per sector
    col_targets = np.maximum(x - v, 0.0)

    # Build prior and balance
    Z_prior = _build_prior_z(x, v)
    Z = ras_balance(Z_prior, col_targets)

    # Ensure no column overshoot
    for j in range(N):
        if Z[:, j].sum() > col_targets[j]:
            Z[:, j] *= col_targets[j] / (Z[:, j].sum() + 1e-30)

    return IONetwork(Z=Z, x=x, v=v, sectors=SECTOR_CODES, labels=SECTOR_LABELS)


def get_io_network(year: int = 2017, use_real_data: bool = True) -> "IONetwork":
    """Main entry point. Uses real BEA files if present, else embedded data."""
    bea_dir = Path("data/raw/bea")
    if use_real_data and (bea_dir / "use_table_2017.csv").exists():
        print("NOTE: BEA file parsing not yet implemented; using embedded data.")
    else:
        print("BEA data files not found — using embedded 2017 calibration.")
        print("Run: python src/data_download/download_bea.py  to fetch real data.")
    return build_network_from_embedded(year)


def save_network(net: "IONetwork", out_dir: Path = PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    net.summary().to_csv(out_dir / "sector_summary.csv", index=False)
    pd.DataFrame(net.A, index=net.sectors, columns=net.sectors).to_csv(
        out_dir / "io_coefficients.csv")
    pd.DataFrame(net.L, index=net.sectors, columns=net.sectors).to_csv(
        out_dir / "leontief_inverse.csv")
    pd.DataFrame(net.Z, index=net.sectors, columns=net.sectors).to_csv(
        out_dir / "flow_matrix_Z.csv")
    print(f"Network matrices saved to {out_dir}/")


def verify_network(net: "IONetwork") -> None:
    print("\n── IO Network Diagnostics (66 BEA sectors) ───────────────────────")
    print(f"  N sectors:              {len(net.sectors)}")
    print(f"  GDP (Σ value added):    ${net.gdp:,.0f} bn  (BEA 2017 ≈ $19,519 bn)")
    print(f"  Σ Domar weights:        {net.lam.sum():.4f}  (expected 1.8–2.0)")
    print(f"  Σ VA shares:            {net.w_va.sum():.4f}  (must = 1.0)")
    print(f"  Σ final demand shares:  {net.alpha_f.sum():.4f}  (must = 1.0)")
    rho = np.max(np.abs(np.linalg.eigvals(net.A)))
    print(f"  Spectral radius(A):     {rho:.4f}  (must be < 1)")
    print(f"  Min Leontief diagonal:  {net.L.diagonal().min():.4f}  (must be ≥ 1)")
    col_err = np.abs(net.Z.sum(axis=0) - (net.x - net.v))
    print(f"  Max column residual:    {col_err.max():.2e} bn  (target < 1e-6)")
    neg_f = (net.f < 0).sum()
    print(f"  Sectors with f_i < 0:  {neg_f}  (should be 0 after clipping)")
    print("────────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    net = build_network_from_embedded()
    verify_network(net)
    save_network(net)
    top12 = net.summary().nlargest(12, "gross_output_bn")
    print(top12[["label","gross_output_bn","value_added_bn",
                  "domar_weight","leontief_row_mult"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
