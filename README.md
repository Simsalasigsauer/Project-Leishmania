# Modeling Cutaneous Leishmaniasis: Climate Change & Epidemiology (Brazil)

![Dashboard Preview](results/dashboard_preview.png)

## Project Overview
This project establishes an "End-to-End" pipeline to model the impact of climate change (Scenario **SSP3-7.0**) on the incidence of Cutaneous Leishmaniasis in Brazil. By fusing epidemiological data, demographic statistics, and bioclimatic variables, we predict risk shifts for the period **2041–2060**.

### Data Sources
* **Health (Response):** DATASUS / SINAN (Reported cases)
* **Demographics:** IBGE (Population estimates for incidence calculation)
* **Climate (Predictors):** WorldClim (Historical data & CMIP6 Projections)
* **Geometry:** GADM Level 2 (Brazilian Municipalities)

### Key Features
* **Source-Agnostic:** The Python code is adaptable for other regions (e.g., North Africa) by swapping shapefiles and health registers.
* **Machine Learning Ensemble:** Utilizes a Voting Regressor combining **XGBoost**, **Random Forest**, and **Ridge Regression**.
* **Explainable AI:** Integration of **SHAP values** to decode local risk drivers (e.g., identifying if heat or humidity drives the risk).
* **Interactive Dashboard:** HTML-based visualization of incidence projections and causal analysis.

---

## Repository Structure

The core logic is located in the `src/` folder, designed as a sequential pipeline:

* `01_climate_prep.py` - Processing and aggregating WorldClim raster data using zonal statistics.
* `02_build_dataset.py` - Merging health data, demographics, and climate predictors.
* `03_add_demographics.py` - Calculating normalized incidence rates (Cases per 100k).
* `06_train_model.py` - Training and evaluating the Voting Regressor Ensemble.
* `09_response_curves.py` - Analyzing ecological dependencies (PDPs/ALE).
* `10_final_dashboard.py` - Generating the interactive HTML visualization.

---

## Key Results

### 1. Feature Importance
The classification model identified **Temperature Seasonality (`bio_4`)** and **Annual Precipitation (`bio_12`)** as the primary drivers for defining risk areas. Stability is key for the vector's survival.

![Feature Importance](results/importance_Classification.png)

### 2. Future Projection (SSP3-7.0)
The projection for 2041–2060 reveals a complex redistribution of disease risk:
* **Decline in Central Brazil:** Regions like Mato Grosso may see a decrease in cases as temperatures exceed the physiological limit of the sandfly and conditions become too arid.
* **Increase in the South & Coast:** Previously cooler regions are warming up, creating new suitable habitats for the vector.
* **Amazon Region:** Remains highly endemic with specific hotspots intensifying due to persistent humidity.

---

## Usage

### 1. Install Dependencies
Ensure you have Python installed:
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
Execute the scripts sequentially from the `src/` folder:
```bash
python src/01_climate_prep.py
python src/02_build_dataset.py
# ... run scripts 03 to 09 ...
python src/10_final_dashboard.py
```


