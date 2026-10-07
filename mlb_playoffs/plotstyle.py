"""Shared matplotlib style: one validated categorical palette, recessive chrome."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#b8b7b1"
# Categorical slots in fixed order (validated CVD-safe; first three all-pairs).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Diverging pair for "favors home" vs "favors away".
POS, NEG = "#2a78d6", "#e34948"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False,
        "legend.fontsize": 9, "text.color": TEXT, "font.size": 10, "lines.linewidth": 2,
    })


apply()
