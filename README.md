# Upper-tail correction of multivariate synthetic environmental series using annual maxima

This repository contains the complete code for the Upper-tail correction technique described in this [brief report](https://rdcu.be/fbVgK) published in Stochastic Environmental Research and Risk Assessment. The notebooks folder contains the examples used in the paper. 

The code is also implemented in the [Bluemath](https://github.com/GeoOcean/BlueMath_tk) Python package developed by [GeoOcean](https://github.com/GeoOcean). 

## Repository Structure
- `src`: This folder contains the core implementation of the Upper-tail correction technique and other utils functions employed in the code. The provided class performs the correction without stratification.

- `Notebooks`: These notebook demonstrates the application of the correction technique for Santoña, Spain in two different variables, significant wave height ($H_s$) and wave peak period ($T_p$). The step-by-step process illustrates how the method refines extreme value estimates.

## Example Workflow

The correction process typically involves the following steps:

- Load your dataset (e.g., Hs data).
- Initiate the `ExtremeCorrection` class.
- Fit the model to the data and apply the correction.
- Generate and analyze the corrected figures, which will be saved automatically in the selected folder.

# Citation

If you use Upper-tail correction technique in your research, please cite it using the following BibTeX entry:

```bib
@article{Collado2026,
  author    = {Víctor Collado and Fernando J. Méndez and Roberto Mínguez},
  title     = {Upper-tail correction of multivariate synthetic environmental series using annual maxima},
  journal   = {Stochastic Environmental Research and Risk Assessment},
  year      = {2026},
  month     = apr,
  day       = {1},
  volume    = {40},
  number    = {4},
  pages     = {92},
  issn      = {1436-3259},
  doi       = {10.1007/s00477-026-03215-0},
  url       = {https://doi.org/10.1007/s00477-026-03215-0}
}
```
