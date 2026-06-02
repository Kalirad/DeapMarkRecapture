# The Exact Hypergeometric Posterior Method for Accurate Inference of Population Size from Mark–Recapture Data

This repository contains the code, data, and analysis scripts necessary to reproduce
the results, simulations, and figures presented in the paper _The Exact Hypergeometric
Posterior Method for Accurate Inference of Population Size from Mark–Recapture Data_,
by Danial Mirzaee, Seyed Amir Malekpour, and Ata Kalirad (DOI: pending publication).

---

## Installation

The `ehpmarkrecap` package can be installed directly from GitHub:

```bash
pip install git+https://github.com/YOUR_USERNAME/ehpmarkrecap.git
```

### Basic usage

```python
from ehpmarkrecap import ehp, ehp_plot

# Single dataset, unbounded
res = ehp((100, 100, 10), K=False, alpha=0.05)
print(res["mode"], res["median"], res["ci_low"], res["ci_high"])

# With bounded K and heterogeneity
res = ehp((20, 20, 5), K=3000, phi=0.7, omega=1.5, alpha=0.05)
ehp_plot(res)
```

---

## Reproducing the Results

### Prerequisites

Python 3.12, NumPy 2.1, SciPy 1.14.

### Repository Structure

* **`ehpmarkrecap/`**: Installable Python package with the core EHP functions.
* **`OtherMethods.py`**: Implementations of alternative methods used for comparison.
* **`Simulation.py`**: Script used to run the simulations.
* **`figures_and_tables.ipynb`**: Jupyter Notebook to generate all figures and tables.
* **`ci_efficiency_all_results_table_7.csv`**: Results for Table 7.
* **`data/`**: Empirical datasets used in case studies:
    * `Data_Khelifa_et_al._2021_Sci.Rep.xlsx`
    * `Hinneberg_et_al_2022_Multi_Surveyor_CMR_resultsfile.xlsx`
    * `Rhinoceros_Auklet_North_American_Pacific_Coast_(GLS)-tracks.csv`