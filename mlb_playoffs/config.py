"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"          # raw downloads (never re-fetched once present)
MANUAL_DIR = DATA_DIR / "manual"        # user-edited inputs: brackets, injuries, overrides
EXTERNAL_DIR = DATA_DIR / "external"    # optional pybaseball / statsapi pulls
MODEL_DIR = DATA_DIR / "models"
CHART_DIR = ROOT / "charts"

FIRST_SEASON = 2015
LAST_SEASON = 2025          # last season with complete Retrosheet data
SEASONS = list(range(FIRST_SEASON, LAST_SEASON + 1))

RETRO_BASE = "https://raw.githubusercontent.com/chadwickbureau/retrosheet/master"

# Retrosheet postseason event-file suffixes (missing ones are skipped).
POSTSEASON_EVENT_FILES = (
    ["WS", "ALCS", "NLCS", "ALD1", "ALD2", "NLD1", "NLD2", "ALWC", "NLWC"]
    + [f"{lg}W{i}" for lg in ("AL", "NL") for i in range(1, 5)]
)
POSTSEASON_GAMELOGS = {"WC": "GLWC.TXT", "DS": "GLDV.TXT", "LCS": "GLLC.TXT", "WS": "GLWS.TXT"}

# Fixed wOBA linear weights (roughly the 2015-2025 average; constant so the
# metric is comparable across seasons and needs no future information).
WOBA_WEIGHTS = {"BB": 0.69, "HBP": 0.72, "1B": 0.89, "2B": 1.27, "3B": 1.62, "HR": 2.10}
FIP_CONSTANT = 3.15

# Retrosheet uses franchise codes; map to familiar abbreviations for display.
TEAM_DISPLAY = {
    "ANA": "LAA", "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CHA": "CWS",
    "CHN": "CHC", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET", "HOU": "HOU",
    "KCA": "KC", "LAN": "LAD", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYA": "NYY",
    "NYN": "NYM", "OAK": "ATH", "ATH": "ATH", "PHI": "PHI", "PIT": "PIT", "SDN": "SD",
    "SEA": "SEA", "SFN": "SF", "SLN": "STL", "TBA": "TB", "TEX": "TEX", "TOR": "TOR",
    "WAS": "WSH",
}
TEAM_NAMES = {
    "ANA": "Angels", "ARI": "Diamondbacks", "ATL": "Braves", "BAL": "Orioles", "BOS": "Red Sox",
    "CHA": "White Sox", "CHN": "Cubs", "CIN": "Reds", "CLE": "Guardians", "COL": "Rockies",
    "DET": "Tigers", "HOU": "Astros", "KCA": "Royals", "LAN": "Dodgers", "MIA": "Marlins",
    "MIL": "Brewers", "MIN": "Twins", "NYA": "Yankees", "NYN": "Mets", "OAK": "Athletics",
    "ATH": "Athletics", "PHI": "Phillies", "PIT": "Pirates", "SDN": "Padres", "SEA": "Mariners",
    "SFN": "Giants", "SLN": "Cardinals", "TBA": "Rays", "TEX": "Rangers", "TOR": "Blue Jays",
    "WAS": "Nationals",
}
# The Athletics changed Retrosheet code in 2025; treat them as one franchise.
FRANCHISE_ALIASES = {"OAK": "ATH"}

for _d in (DATA_DIR, CACHE_DIR, MANUAL_DIR, EXTERNAL_DIR, MODEL_DIR, CHART_DIR):
    _d.mkdir(parents=True, exist_ok=True)
