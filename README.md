# Modeling Cutaneous Leishmaniasis: Climate Change & Epidemiology (Brazil)

##  Project Overview
This project establishes an "End-to-End" pipeline to model the impact of climate change (Scenario SSP3-7.0) on the incidence of Cutaneous Leishmaniasis in Brazil. By fusing epidemiological data (SINAN), demographic stats (IBGE), and bioclimatic variables (WorldClim), we predict risk shifts for the period 2041–2060.

**Key Features:**
* **Source-Agnostic:** Code adaptable for other regions (e.g., North Africa).
* **Machine Learning Ensemble:** Voting Regressor (XGBoost, Random Forest, Ridge).
* **Explainable AI:** SHAP value analysis to decode local risk drivers.
* **Interactive Dashboard:** HTML-based visualization of incidence projections.

##  Repository Structure

The core logic is located in the `src/` folder, organized as a sequential pipeline:

* `01_climate_prep.py` - Processing WorldClim raster data.
* `02_build_dataset.py` - Merging health, demo, and climate data.
* `03_add_demographics.py` - Calculating incidence rates.
* `06_train_model.py` - Training the Voting Regressor Ensemble.
* `09_response_curves.py` & `10_final_dashboard.py` - Analysis & Visualization.

## Key Results

### Feature Importance
The model identified **Temperature Seasonality (bio_4)** and **Annual Precipitation (bio_12)** as the primary drivers for defining risk areas.
![Feature Importance](results/importance_Classification.png)

### Future Projection (SSP3-7.0)
The projection reveals a complex redistribution:
* **Decline** in central Brazil (too hot/dry for vectors).

##  Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
2. **Run scripts in numerical ordner from `src/`**
   