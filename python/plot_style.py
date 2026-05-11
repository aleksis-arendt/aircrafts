import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "figures"
MATPLOTLIB_CACHE_DIR = PROJECT_ROOT / ".matplotlib-cache"


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    font_dir = Path(r"C:\Program Files\MiKTeX\fonts\opentype\public\lm")
    for name in (
        "lmroman10-regular.otf",
        "lmroman10-italic.otf",
        "lmroman10-bold.otf",
        "lmroman10-bolditalic.otf",
    ):
        font_path = font_dir / name
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt
