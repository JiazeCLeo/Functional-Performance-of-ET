# Functional-Performance-of-ET

This project aims to apply four causal discovery methods to time series data in order to analyse the functional performance of an evapotranspiration system. The file [Four Causal Discovery Methods Demo.py](./Four%20Causal%20Discovery%20Methods%20Demo.py) involves an artificial time series dataset. This system contains nonlinear interactions and indirect coupling, providing a benchmark for causal method comparison. The mathematical equations are shown here:
```
x1(t) = x1(t-1) * (3.9 - 3.9 x1(t-1)) + ξ1
x2(t) = x2(t-1) * (3.6 - 0.4 x1(t-1) - 3.6 x2(t-1)) + ξ2
x3(t) = x3(t-1) * (3.6 - 0.4 x2(t-1) - 3.6 x3(t-1)) + ξ3
x4(t) = x4(t-1) * (3.8 - 0.35 x3(t-1) - 3.8 x4(t-1)) + ξ4
x5(t) = x5(t-1) * (3.2 - 0.4 x2(t-1) - 3.6 x5(t-1)) + ξ5
```
We provided four causal discovery methods including:
- Granger Causality (GC)
- Transfer Entropy (TE)
- PCMCI
- Convergent Cross Mapping (CCM)

The required Python dependencies are specified in the file [environment.yml](./environment.yml), including:
- numpy
- pandas
- matplotlib
- scipy
- statsmodels
- pyinform
- seaborn
- tigramite
Readers are encouraged to explore the demo workflow and adapt these causal discovery methods to their own datasets and research applications.
---

# Evapotranspiration dataset

This research utilizes two datasets:

- **Temporal Functional Performance Analysis Data (30-Minute Interval)**
- **OpenET Functional Performance Data (Daily Interval)**

The processed datasets used in this study are publicly available via [HydroShare](https://www.hydroshare.org/resource/0ef3eda3534f44a6bbd65786d57222ea/).

Users are encouraged to download the data directly from HydroShare and follow the demo workflow provided in this repository.

---

# Latent Heat Model

This study applies the **Priestley–Taylor (PT)** and **Surface Flux Equilibrium (SFE)** methods to simulate latent heat flux.

The model implementations are provided in [LEfunctions.py](./LEfunctions.py).

### Implemented Functions

- `fun_LEPT`  
  Implements the **Priestley–Taylor (PT)** method for latent heat estimation.

- `fun_LESFE`  
  Implements the original **Surface Flux Equilibrium (SFE)** formulation, which neglects ground heat flux.

- `fun_LESFE_Mod`  
  Implements a modified SFE formulation that incorporates ground heat flux, enabling a consistent and comparable evaluation across models.
