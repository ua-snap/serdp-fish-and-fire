# SERDP Fish and Fire pipelines and utilities

This repository is currently for SNAP internal use only. 

This codebase is for SNAP's technical assistance work on the SERDP Fish and Fire project (internal name). Currently it provides a single data pipeline for producing monthly summaries of key variables in the 1km WRF outputs created for this project, which can be found at `postprocess_wrf/monthly_summaries.ipynb`.

In the future, it may make sense to include the [`align-wrf-modis`](https://github.com/ua-snap/align-wrf-modis) repository as a subdirectory here for simplicity. New pipeline efforts such as these should be added to this repo as separate subdirectories in the main directory. 
