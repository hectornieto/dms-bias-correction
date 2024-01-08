# DMS Bias Correction

## Synopsis

This project contains the *Python* code for enhancing the dynamic LST range of sharpened LST scenes, by fusing them with Landsat LST imagery. 

The project consists of: 

1. a lower-level module `dms_bias_correction.scale_lst.py` with the basic functions needed for the bias correction approach 
2. a higher-level module `dms_bias_correction.landsat_collection_2_helper.py`for easily running the correction by reading [Landsat Collection 2 Level 2](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products)

## Installation

Download the project to your local system, enter the download directory and then type

`python setup.py install` 

if you want to install pyTSEB and its low-level modules in your Python distribution. 

The following Python libraries will be required:

- Numpy
- Scipy
- scikit-image
- GDAL
- pyDMS, at [https://github.com/radosuav/pyDMS](https://github.com/radosuav/pyDMS)

With `conda`, you can create a complete environment with
```
conda env create -f environment.yml
```

## Code Example
### High-level example

The easiest way to get a feeling of the sharpening enhancement is throught the example script [test_dms_correction](./test_dms_correction.py)
In a terminal shell, navigate to your working folder and type

- `python test_dms_correction.py` 

The script will read all the Landsat images and compute the reference dynamic range that will be used to correct the sharpened S3 LST image. Then it will compare the corrected and uncorrected DMS image to a Landsat scene acquired on the same date and plot the results in [test/dms_corr_best_case](./test/dms_corr_best_case)

### Low-level example

## Main Scientific References
When using this sofware please cite the following refrences:

- R. Guzinski, H. Nieto, R. Ramo-Sánchez, J.M. Sánchez, I. Joma, R. Zitouna-Chebbi, O. Roupsard, R. and López-Urrea, Improving field-scale crop actual evapotranspiration monitoring with Sentinel-3, Sentinel-2, and Landsat data fusion (2023) International Journal of Applied Earth Observation and Geoinformation", volume 125, art. No. 103587, doi :10.1016/j.jag.2023.103587
- J. M. Sánchez, J. M. Galve, H. Nieto and R. Guzinski, Assessment of High-Resolution LST Derived From the Synergy of Sentinel-2 and Sentinel-3 in Agricultural Areas, (2034) IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 17, pp. 916-928, 2024, doi: 10.1109/JSTARS.2023.3335896.


## Tests
The folder *./test* contains an example for running the correction of a DMS sharpened image, at [test/dms_raw](./test/dms_raw), using a time series of downloaded Landsat Collection 2 Level 2 scenes, located at [test/landsat](./test/landsat). The output will be stored at [test/dms_corr_best_case](./test/dms_corr_best_case) and could be compared to the actual LST Landsat scene of the same date in [test/landsat_reference](./test/landsat_reference)

## Contributors
- **Hector Nieto** ([hector.nieto@ica.csic.es](mailto:hector.nieto@ica.csic.es), [hector.nieto.solana@gmail.com](mailto:hector.nieto.solana@gmail.com)) main developer
- **Radoslaw Guzinski** main developer, tester


## License
dms-bias-correction:  enhancing the dynamic LST range of sharpened LST scenes

Copyright 2023 Hector Nieto and contributors.
    
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
