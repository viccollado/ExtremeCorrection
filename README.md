# Extreme Correction

This repository contains the complete code for the Extreme Correction technique, along with example notebooks that demonstrate its usage. The Extreme Correction method is designed to correct extreme value data, particularly focusing on significant wave height (Hs) datasets. 

The code is also implemented in the [Bluemath](https://github.com/GeoOcean/BlueMath_tk) Python package developed by [GeoOcean](https://github.com/GeoOcean). 

## Repository Structure
- `src`: This folder contains the core implementation of the Extreme Correction technique and other utils functions employed in the code. The provided class performs the correction without dividing the data into clusters (Weather Types - WT).

- `Notebooks`: This notebook demonstrates the application of the correction technique for Santoña, Spain in two different variables, significant wave height ($H_s$) and wave peak period ($T_p$). The step-by-step process illustrates how the method refines extreme value estimates.


## Example Workflow

The correction process typically involves the following steps:

- Load your dataset (e.g., Hs data).
- Initiate the `ExtremeCorrection` class.
- Fit the model to the data and apply the correction.
- Generate and analyze the corrected figures, which will be saved automatically in the selected folder.