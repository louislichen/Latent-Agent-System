"""Select a font for the benchmark plots' Chinese labels."""

import os
from pathlib import Path


def cjk_font_family():
    from matplotlib import font_manager

    override = os.environ.get("LATENT_PLOT_FONT")
    candidates = [override] if override else [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            font_manager.fontManager.addfont(candidate)
            return font_manager.FontProperties(fname=candidate).get_name()
    if override:
        raise FileNotFoundError(f"LATENT_PLOT_FONT does not exist: {override}")
    return "sans-serif"
