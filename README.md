# The Exact Hypergeometric Posterior Method for Accurate Inference of Population Size from Mark–Recapture Data

This repository contains the code, data, and analysis scripts necessary to reproduce the results, simulations, and figures presented in the paper _The Exact Hypergeometric Posterior Method for Accurate
Inference of Population Size from Mark–Recapture Data_, by Danial Mirzaee, Seyed Amir Malekpour, and
Ata Kalirad (DOI: pending publication).

---

## Repository Structure

The file organization is structured to separate raw data, methodological scripts, simulation runs, and final outputs:

* **`ehpmarkrecap.py`**: The core Python module containing the primary functions and algorithms for the mark-recapture methodology.
* **`OtherMethods.py`**: A module containing implementations of alternative methods used for comparative performance in the study.
* **`Simulation.py`**: The script used to run the simulations.
* **`figures_and_tables.ipynb`**: A Jupyter Notebook that ingests the simulation results and raw data to generate the exact figures and tables found in the manuscript.
* **`ci_efficiency_all_results_table_7.csv`**: Results used to produce Table 7 in the paper.
* **`data/`**: A directory containing empirical datasets used for case studies, including:
    * `Data_Khelifa_et_al._2021_Sci.Rep.xlsx`
    * `Hinneberg_et_al_2022_Multi_Surveyor_CMR_resultsfile.xlsx`
    * `Rhinoceros_Auklet_North_American_Pacific_Coast_(GLS)-tracks.csv`

---

## Prerequisites and Installation

To run the code in this repository, you will need Python 3.12, NumPy 2.1, and SciPy 1.14.
