# Code for Palmo et al.

Code and derived data for the manuscript figures.

```
code/airborne/       FIREX-AQ inputs, steady-state model, FNR thresholds, P(O3) network
code/satellite/      Williams Flats TROPOMI, 3 August 2019
code/wrf-les-chem/   Williams Flats WRF-LES-Chem
data/                Tables and model files read by the notebooks
notebooks/           One notebook per figure
figs/                Figures
```

The FIREX-AQ 1 Hz merge, TROPOMI L2 NO2, HCHO, and CO, and the WRF-LES-Chem output of Wang et al. (2021) are not included.

`code/airborne/steady_state_po3.jl` needs Julia 1.10 and [HSSModel.jl](https://github.com/joshua-laughner/HSSModel.git). VOC rate constants are in `code/airborne/config/Xu2021_VOCs_clean.csv`.

| Figure | Notebook |
|---|---|
| 1 | Schematic |
| 2 | `notebooks/fig02_plume_structure.ipynb` |
| 3 | `notebooks/fig03_ozone_regimes.ipynb` |
| 4 | `notebooks/fig04_tropomi_williams.ipynb` |
| 5 | `notebooks/fig05_les_nn.ipynb` |
| S1 | `notebooks/figs01_flight_tracks.ipynb` |
| S2 | Schematic |
| S3 | `notebooks/figs03_smoke_coordinates.ipynb` |
| S4 | `notebooks/figs04_hcho_vocr.ipynb` |
| S5 | `notebooks/figs05_voc_contributions.ipynb` |
| S6 | `notebooks/figs06_column_distributions.ipynb` |
| S7 | `notebooks/figs07_fnr_thresholds.ipynb` |
| S8 | `notebooks/figs08_po3_by_regime.ipynb` |
| S9 | `notebooks/figs09_nn_heldout.ipynb` |
| S10 | `notebooks/figs10_shap.ipynb` |
| S11 | `notebooks/figs11_les_cross_section.ipynb` |
| S12 | `notebooks/figs12_les_columns.ipynb` |
| S13 | `notebooks/figs13_plume_thickness.ipynb` |
