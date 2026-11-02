"""
config.py — every data/output path used by the TAKAHE scripts, in one place.

Set the environment variable TAKAHE_DATA to the folder holding your data
(Bedmap3, MEaSUREs v2, picked layers, ...) or edit DATA_DIR below.
Files keep the same relative layout they had on the original machine
(e.g. <DATA_DIR>/Picked_PSTs_20261014/horallpst, <DATA_DIR>/Margin Layers/...).
"""
import os

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("TAKAHE_DATA", os.path.join(REPO_DIR, "data"))
OUT_DIR = os.environ.get("TAKAHE_OUT", os.path.join(REPO_DIR, "outputs"))


def p(*parts):
    """Path inside DATA_DIR."""
    return os.path.join(DATA_DIR, *parts)


def out(*parts):
    """Path inside OUT_DIR (created on demand)."""
    path = os.path.join(OUT_DIR, *parts)
    os.makedirs(os.path.dirname(path) if os.path.splitext(path)[1] else path, exist_ok=True)
    return path


# Core gridded datasets (see README for download links)
BM3_BED = p("bm3_bed.tif")                                   # Bedmap3 bed, EPSG:3031
BM3_SURF = p("bm3_surface.tif")                              # Bedmap3 surface, EPSG:3031
MV2_VEL = p("antarctica_ice_velocity_450m_v2.nc")           # MEaSUREs InSAR velocity v2
WD2014 = p("WD2014_Chronology_BuizertSigl.tab")              # WAIS Divide WD2014 chronology
