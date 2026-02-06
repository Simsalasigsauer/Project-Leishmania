import pandas as pd
import geopandas as gpd
import numpy as np
import os
import folium
from folium.plugins import DualMap # Wichtig für den Vergleich!
import pickle
from xgboost import XGBRegressor

# PFADE
BASE_DIR = r"C:\Users\Thomas\Forschungsprojekt- Leishmaniose"
CACHE_FILE = os.path.join(BASE_DIR, "data", "processed", "temp_shap_data.pkl")
OUTPUT_HTML = os.path.join(BASE_DIR, "results", "dashboard_VALIDATION_SIDE_BY_SIDE.html")

def run_validation():
    print("--- Erstelle Validierungs-Karte (Realität vs. Modell) ---")
    
    if not os.path.exists(CACHE_FILE):
        return

    with open(CACHE_FILE, 'rb') as f:
        data = pickle.load(f)
        full_gdf = data['full_gdf']
        features_train = data['features_train']

    # Modell nachtrainieren (auf Basis der aktuellen Daten)
    train_df = full_gdf.dropna(subset=features_train + ['Faelle', 'incidence'])
    X = train_df[features_train]
    y = np.log1p(train_df['incidence'])
    
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    
    full_gdf['pred_incidence_2022'] = np.expm1(model.predict(full_gdf[features_train].fillna(0)))
    
    # Karte bauen: DualMap (Split Screen)
    m = folium.plugins.DualMap(location=[-14.2, -51.9], zoom_start=4, layout='vertical')
    
    # Farbskala 
    vmax = np.percentile(full_gdf['incidence'], 95) # Ausreißer kappen für bessere Farben
    cmap = folium.LinearColormap(['white', 'orange', 'red', 'purple'], vmin=0, vmax=vmax, caption="Inzidenz")

    # Linke Karte: ECHTE DATEN
    folium.GeoJson(
        full_gdf,
        name="Realität (Gemeldet)",
        style_function=lambda x: {
            'fillColor': cmap(x['properties']['incidence']),
            'color': 'transparent', 'fillOpacity': 0.7
        },
        tooltip=folium.GeoJsonTooltip(fields=['NAME_2', 'incidence'], aliases=['Ort:', 'Echt:'])
    ).add_to(m.m1)
    
    # Rechte Karte: KI VORHERSAGE
    folium.GeoJson(
        full_gdf,
        name="KI Modell (Gelernt)",
        style_function=lambda x: {
            'fillColor': cmap(x['properties']['pred_incidence_2022']),
            'color': 'transparent', 'fillOpacity': 0.7
        },
        tooltip=folium.GeoJsonTooltip(fields=['NAME_2', 'pred_incidence_2022'], aliases=['Ort:', 'Modell:'])
    ).add_to(m.m2)

    m.add_child(cmap)
    m.save(OUTPUT_HTML)
    print(f"✅ Validierungskarte gespeichert: {OUTPUT_HTML}")

if __name__ == "__main__":
    run_validation()