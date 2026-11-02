# TAKAHE — englacial layer analysis at Mt. Takahe, Thwaites Glacier

Code behind *"A case for late-Holocene subglacial volcanism at Mt. Takahe, Thwaites Glacier, West Antarctica"* (Jason Bott, UT Austin / UTIG honors thesis, 2026).

It works on radar isochrones picked from the AGASEA 2004/05 ice-penetrating radar survey. The steps are:

1. Correct the picked layers to elevation and date them with the WAIS Divide WD2014 chronology.
2. Map the layers across Takahe's flank.
3. Compare them with surface velocity, strain and basal-melt models.
4. Invert drawdown and differential thickening of the layers for basal melt and geothermal heat flux.

## Workflow

```
picked layers (Landmark exports, one folder per line: X47a, Y17a, ...)
        │
        ▼
correct_layers.produce_layer_df ──► correct_layers_to_radar_surface   (depth → elevation, firn correction)
        │                                   │
        │                                   ▼
        │                        layer_age_estimator / Core_Upload     (ages from WD2014 at line Y67a)
        ▼
Layer_Analysis_Master.py   ◄── bm3_upload (Bedmap3)  ◄── Mv2_uploader (MEaSUREs v2 velocity)
        │
        ├── strain_tools / smooth_strains ........ surface strain rates (flow-aligned), back-tracking
        ├── stat_melt / layer_correlation_plot .... frictional basal melt (Beem et al. 2010) vs. layers
        ├── layer_shape_v2 ........................ shortening, max slope, fold extrema, layer spacing
        ├── Vertical_Velocity_Profile ............. vertical velocity through the column
        ├── nye_drawdown .......................... Nye-with-melt layer prediction, infill → heat flux
        └── layer_extent_maps / layer_extent_map .. map-view figures of every dated isochrone
                                                    │
                                                    ▼
                                    heatflux_compare.py (Takahe vs. other volcanic/geothermal systems)
```

Run `Layer_Analysis_Master.py` cell by cell (Spyder `# %%` cells). It is the entry point. Edit `PST_folder_list` there to choose the lines.

## Modules

| File | What it does |
|---|---|
| **Entry points** | |
| `Layer_Analysis_Master.py` | Master workflow: load data, correct layers, set limits per line, run the analyses |
| `MapFigureMaker.py` | Map figures of horizons around Takahe |
| `Shear_Layers_Plot.py` | Earlier all-in-one shear-margin layer analysis (superseded by the master script) |
| `Landmark_Layers.py` | Loads and plots Landmark horizon exports (Muldoon/Leuro, FFR layers) |
| **Loading data** | |
| `bm3_upload.py`, `uploader.py`, `data_uploader_no_topo.py` | Bedmap3 grids and other gridded datasets → interpolators |
| `Mv2_uploader.py` | MEaSUREs v2 InSAR surface velocity |
| `Core_Upload.py` | WAIS Divide WD2014 age–depth chronology |
| `dataframe_merger_horizons.py` | Merges horizon CSVs into one dataframe |
| `Flowband_Functions.py` | Shared helpers: tif readers, interpolators, flowline tracer, 3D plots |
| **Layers** | |
| `correct_layers.py` | Builds layer dataframes per line, picks surface/bed/base, along-track distance, analysis limits |
| `correct_layers_to_radar_surface.py` | Depth-to-elevation correction referenced to the radar surface (+ firn) |
| `layer_age_estimator.py` | Assigns ages to layers from the core chronology |
| `layer_shape_v2.py` | Shortening, max slope, fold extrema, interval spacing (`layer_shape_functions.py` = older version) |
| `layer_correlation_plot.py` | Multi-line figure: layers vs. velocity, strain, melt |
| `layer_extent_maps.py`, `layer_extent_map.py` | Map-view figures of every picked isochrone (elevation / fractional depth) |
| **Ice flow, strain, melt** | |
| `strain_tools.py` | Strain rates (with uncertainty), vertical strain, back-tracking along flow |
| `smooth_strains.py` | Strain rates from a smoothed velocity field |
| `Velocity_Normal_Vectors.py` | Flow-normal vectors and the original strain calculation (see Known issues) |
| `SurfaceStrainCalcsIncomplete.py` | Early strain prototype |
| `Vertical_Velocity_Profile.py` | Vertical velocity profiles along a line |
| `stat_melt.py` | Frictional basal melt, deterministic or Monte Carlo (after Beem et al. 2010) |
| `Nye_Melt.py` | Nye / Dansgaard–Johnsen fits to dated layers in one ice column (Fahnestock et al. 2001) |
| `nye_drawdown.py` | Predicted vs. observed layer depth with basal melt; infill → melt rate → heat flux |
| **Figures and utilities** | |
| `heatflux_compare.py` | Log-scale comparison of the Takahe heat flux with other systems |
| `Takahe_Map.py`, `Volcano_Profile.py`, `Radar_Plots.py` | Context maps, volcano profiles, radar-derived products |
| `colormaps.py`, `Projection_Swapper.py`, `CoordConverter-Inc.py`, `BaseFilePicker.py` | Small utilities |
| `config.py` | Every data and output path, in one place |
| **From Flowband Code** | |
| `layers_3D.py` | 3D view of the Bedmap3 bed and ice surface with the dated radar isochrones |
| `nye_melt_at_x.py` | Interactive Nye-with-melt fit at one along-track position (uses `Nye_Melt.py`) |
| `time_flowline.py` | Traces ice through the velocity field by time (where was this ice T years ago?) |
| `Flowband_Tracing_Not_Interactive.py` | Flowline/flowband tracing with GHF, RACMO, ALBMAP and Bedmap inputs (Sticky Spot work) |
| `THW_BedPicks.py`, `Profile_Plot.py`, `Plot_Layers.py` | Thwaites bed picks, flowline profiles, 2.5D model layer output |
| `Horizon_CSV_Scraper.py` | Splits a multi-horizon CSV export into one dataframe per horizon |
| `Magma_Viscosity_Plots.py`, `Strk_Dip_Matcher.py` | Small side scripts (magma viscosity plot, strike/dip matching) |

## Setup

```bash
git clone <this repo>
cd TAKAHE
pip install -r requirements.txt
```

### Data

Data are not included in this repo. Put them in a folder and point the code at it:

```bash
# Windows (PowerShell):  $env:TAKAHE_DATA = "C:\path\to\data"
export TAKAHE_DATA=/path/to/data      # default: ./data
export TAKAHE_OUT=/path/to/outputs    # default: ./outputs
```

`config.py` expects these files directly in the data folder:

| File | Source |
|---|---|
| `bm3_bed.tif`, `bm3_surface.tif` | Bedmap3 (Pritchard et al. 2025), EPSG:3031 |
| `antarctica_ice_velocity_450m_v2.nc` | MEaSUREs InSAR-Based Antarctica Ice Velocity Map v2 (Rignot, Mouginot & Scheuchl 2017), NSIDC-0484 |
| `WD2014_Chronology_BuizertSigl.tab` | WAIS Divide WD2014 chronology (Buizert et al. 2015; Sigl et al. 2016), PANGAEA |
| `Picked_PSTs_20261014/horallpst/`, `Margin Layers/<line>/` | Picked radar layers (AGASEA 2004/05, UTIG) — not public |

Other scripts reference further files under the same folder (`config.p("...")`). Search for `config.p(` to see them.

## Key numbers (X47a)

| Quantity | Value |
|---|---|
| Layer ages (WD2014, tied at Y67a) | D1 0.592, D2 1.185, D7 1.778, D4 2.224, D6 4.260 ka |
| Infill, D2–D7 | 1.74× trough ÷ flank, 101 m excess over 593 yr → ≥ 171 mm/yr → ≥ 1.66 W m⁻² |
| Drawdown inversion | median 2.44 W m⁻², peak 4.4 W m⁻² |

## Known issues

- `Velocity_Normal_Vectors.Strain_Deliverer`: `np.gradient` returns `[d/dy, d/dx]`, so `dudx` and `dvdy` are swapped there, and the rotation uses the normal rather than the flow direction. Use `strain_tools.strain_rates` instead (units 1/yr).
- The docstring of `smooth_strains.py` says `from smooth_strain import ...`. The module is `smooth_strains`.
- Several scripts are Spyder cell scripts that run on import. Import the functions you need rather than whole scripts.

## License

MIT — see [LICENSE](LICENSE).
