# cyclonedata-SST-GCM
Dataset featuring  cyclone with sea-surface temperature (SST) history using general circulation model (GCM)




## Relevant literature

* Hsu, W. C., Patricola, C. M., & Chang, P. (2019). The impact of climate model sea surface temperature biases on tropical cyclone simulations. Climate Dynamics, 53, 173-192.
* Vecchi, G. A., & Soden, B. J. (2007). Effect of remote sea surface temperature change on tropical cyclone potential intensity. Nature, 450(7172), 1066-1070.
* Zhao, M., & Held, I. M. (2012). TC-permitting GCM simulations of hurricane frequency response to sea surface temperature anomalies projected for the late-twenty-first century. Journal of Climate, 25(8), 2995-3009.
* Camargo, S. J., Barnston, A. G., & Zebiak, S. E. (2005). A statistical assessment of tropical cyclone activity in atmospheric general circulation models. Tellus A: Dynamic Meteorology and Oceanography, 57(4), 589-604.
* Palmer, T. N., & Zhaobo, S. (1985). A modelling and observational study of the relationship between sea surface temperature in the north‐west Atlantic and the atmospheric general circulation. Quarterly Journal of the Royal Meteorological Society, 111(470), 947-975.
  
# Cyclone and Sea Surface Temperature Analysis

This project provides tools for analyzing the relationship between tropical cyclones and sea surface temperatures (SST). The analysis is primarily focused on the South Indian Ocean region. 

## Table of Contents

- [Requirements](#requirements)
- [Functions](#functions)
  - [merge_cyclone_sst](#merge_cyclone_sst)
  - [plot_cyclone_tracks_cartopy](#plot_cyclone_tracks_cartopy)
  - [plot_sst_SI](#plot_sst_si)
  - [plot_month_sst_3r](#plot_month_sst_3r)
  - [plot_month_category](#plot_month_category)

## Requirements

The following Python libraries are required to run the scripts:
- pandas
- xarray
- matplotlib
- cartopy
- seaborn

You can install these packages using pip:
```bash
pip install pandas xarray matplotlib cartopy seaborn

## Functions

### `merge_cyclone_sst`

This function merges cyclone data with sea surface temperature (SST) data from a netCDF file. 

#### Parameters:
- `cyclone_path` (str): Path to the cyclone CSV file.
- `sst_path` (str): Path to the SST netCDF file.
- `output_file_path` (str): Path to save the output CSV file.

#### Usage:
```python
merge_cyclone_sst('path_to_cyclone.csv', 'path_to_sst.nc', 'output.csv')

### `plot_cyclone_tracks_cartopy`

This function plots cyclone tracks on a map using Cartopy.

#### Parameters:
- `path` (str): Path to the CSV file containing cyclone data.

#### Usage:
````python
plot_cyclone_tracks_cartopy(path)

### `plot_sst_SI`

This function plots the average sea surface temperature over the South Indian Ocean.

#### Parameters:
- `path` (str): Path to the SST netCDF file.

#### Usage:
````python
plot_sst_SI('path_to_sst.nc')

### `plot_month_sst_3r`

This function calculates and plots the monthly average SST for three regions in the South Indian Ocean.

#### Parameters:
- `path_csv` (str): Path to save the combined CSV file.
- `path_fig` (str): Path to save the output figure.
- `path1` (str): Path to save the CSV file for region 1.
- `path2` (str): Path to save the CSV file for region 2.
- `path3` (str): Path to save the CSV file for region 3.

#### Usage:
````python
plot_month_sst_3r('path_to_combined.csv', 'path_to_figure.png', 'path_to_region1.csv', 'path_to_region2.csv', 'path_to_region3.csv')


### `plot_month_category`

This function plots the monthly distribution of tropical cyclone categories.

#### Parameters:
- `path_in` (str): Path to the input CSV file.
- `fig_out` (str): Path to save the output figure.

#### Usage:
````python
plot_month_category('path_to_input.csv', 'path_to_figure.png')









