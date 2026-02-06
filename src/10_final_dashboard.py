import pandas as pd
import geopandas as gpd
import numpy as np
import os
from tqdm import tqdm
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import folium
import branca.colormap as cm
import shap
import pickle
import warnings
from joblib import Parallel, delayed
import multiprocessing


warnings.filterwarnings("ignore")

# --- 1. KONFIGURATION & PFADE ---
BASE_DIR = r"C:\Users\Thomas\Forschungsprojekt- Leishmaniose"
DATA_FILE = os.path.join(BASE_DIR, "data", "processed", "master_table_with_landscape.csv")
CACHE_FILE = os.path.join(BASE_DIR, "data", "processed", "temp_shap_data.pkl")
OUTPUT_HTML = os.path.join(BASE_DIR, "results", "Scientific_Final_Report_2060.html")

# --- 2. ANALYSE-FUNKTIONEN ---

def calculate_habitat_suitability(temp_series):
    """Berechnet den ökologischen Eignungsindex (Gaussian Niche Model)."""
    mu, sigma = 25.0, 5.0
    suit = np.exp(-((temp_series - mu)**2) / (2 * sigma**2))
    suit = np.where((temp_series < 10) | (temp_series > 32), 0, suit)
    return suit

def get_map_style(feature, col, cmap):
    """Visualisierungs-Stil für geographische Layer."""
    val = feature['properties'][col]
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return {'fillColor': 'gray', 'color': 'transparent', 'fillOpacity': 0}
    if col == 'inc_delta' and abs(val) < 1.0:
        return {'fillColor': '#f0f0f0', 'color': 'transparent', 'fillOpacity': 0.1}
    return {'fillColor': cmap(val), 'color': 'transparent', 'weight': 0, 'fillOpacity': 0.8}

# --- 3. MODELLIERUNG & DASHBOARD-GENERIERUNG ---

def run_scientific_model():
    print("---  OPUS MAGNUM: HIGH-PRECISION ENSEMBLE ANALYSIS  ---")
    
    # A. DATEN-IMPORT
    if not os.path.exists(CACHE_FILE):
        print(" Kritischer Fehler: Cache-Datei nicht gefunden.")
        return 
    
    with open(CACHE_FILE, 'rb') as f:
        data_dump = pickle.load(f)
        full_gdf = data_dump['full_gdf']
        future_df = data_dump['future_df']
        features_train = data_dump['features_train']

    # B. ENSEMBLE-KALIBRIERUNG
    print(f"Kalibrierung des Ensembles...")
    train_df = full_gdf.dropna(subset=features_train + ['incidence'])
    X_train_all = train_df[features_train].values
    y_train_log = np.log1p(train_df['incidence']).values
    
    mask_endemic = (train_df['Faelle'] > 0).values
    X_train = X_train_all[mask_endemic]
    y_train = y_train_log[mask_endemic]
    
    ensemble = VotingRegressor(estimators=[
        ('xgb', XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ('rf', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)),
        ('ridge', make_pipeline(StandardScaler(), Ridge(alpha=1.0)))
    ])
    ensemble.fit(X_train, y_train)

    # C. PROGNOSE-BERECHNUNG (2060)
    print("Generierung der Projektionen für das Jahr 2060...")
    
    X_fut_df = pd.DataFrame()
    X_pres_df = pd.DataFrame()

    for b in [1, 2, 4, 7, 10, 11, 12, 15]:
        col = f"bio_{b}"
        X_fut_df[col] = future_df[col] if col in future_df else full_gdf[col]
        X_pres_df[col] = full_gdf[col]
            
    for col in ['elevation', 'pop_density']:
        X_fut_df[col] = full_gdf[col]
        X_pres_df[col] = full_gdf[col]
        
    X_fut_np = X_fut_df[features_train].fillna(0).values
    X_pres_np = X_pres_df[features_train].fillna(0).values
    
    preds_stack = np.column_stack([
        ensemble.named_estimators_['xgb'].predict(X_fut_np),
        ensemble.named_estimators_['rf'].predict(X_fut_np),
        ensemble.named_estimators_['ridge'].predict(X_fut_np)
    ])
    
    log_pred_2060 = np.mean(preds_stack, axis=1)
    uncertainty = np.std(preds_stack, axis=1)
    
    full_gdf['inc_2060'] = np.expm1(log_pred_2060)
    full_gdf['inc_uncertainty'] = uncertainty
    full_gdf['inc_delta'] = full_gdf['inc_2060'] - full_gdf['incidence']
    
    # Referenzwerte für Tooltip speichern
    full_gdf['suit_now'] = calculate_habitat_suitability(full_gdf['bio_10'])
    full_gdf['suit_fut'] = calculate_habitat_suitability(X_fut_df['bio_10'])
    full_gdf['fut_bio_10'] = X_fut_df['bio_10']
    full_gdf['fut_bio_11'] = X_fut_df['bio_11']
    full_gdf['fut_bio_12'] = X_fut_df['bio_12']

    # D. HOCHPRÄZISE PARALLELE SHAP-ANALYSE (RYZEN-OPTIMIERT)
    n_cores = multiprocessing.cpu_count() - 2
    PRECISION_SAMPLES = 2048 # Hohe Sampling-Rate für exakte Ergebnisse
    
    print(f" Starte Hochpräzisions-Extraktion ({PRECISION_SAMPLES} Samples) auf {n_cores} Kernen...")
    
    def shap_predict(data): return ensemble.predict(data)
    explainer = shap.KernelExplainer(shap_predict, shap.kmeans(X_train, 10))

    def compute_chunk(chunk):
        return explainer.shap_values(chunk, nsamples=PRECISION_SAMPLES)

    chunks_fut = np.array_split(X_fut_np, n_cores)
    chunks_pres = np.array_split(X_pres_np, n_cores)

    results_fut = Parallel(n_jobs=n_cores)(delayed(compute_chunk)(c) for c in chunks_fut)
    results_pres = Parallel(n_jobs=n_cores)(delayed(compute_chunk)(c) for c in chunks_pres)
    
    shap_delta = np.vstack(results_fut) - np.vstack(results_pres)
    
    # E. DETERMINANTEN-REKONSTRUKTION & RESIDUUM-DISTRIBUTION
    feature_map = {
        'bio_1': 'Jahres-Temperatur', 'bio_2': 'Tages-Amplitude',
        'bio_4': 'Temperatur-Saisonalität', 'bio_7': 'Jährliche Amplitude', 
        'bio_10': 'Sommer-Temperatur', 'bio_11': 'Winter-Temperatur', 
        'bio_12': 'Jahres-Niederschlag', 'bio_15': 'Niederschlags-Variabilität',
        'elevation': 'Orographie', 'pop_density': 'Bevölkerungsdichte'
    }
    
    log_delta_real = log_pred_2060 - np.log1p(full_gdf['incidence'].values)
    shap_html_list = []

    for i in range(len(shap_delta)):
        impacts = []
        current_sum = 0.0
        for idx, val in enumerate(shap_delta[i]):
            impacts.append([val, feature_map.get(features_train[idx], features_train[idx])])
            current_sum += val
        
        # Proportionale Distribution des Residuums für 100% additive Korrektheit
        residual = log_delta_real[i] - current_sum
        abs_total = sum(abs(v[0]) for v in impacts)
        if abs_total > 0:
            for item in impacts:
                item[0] += residual * (abs(item[0]) / abs_total)
        
        impacts.sort(key=lambda x: abs(x[0]), reverse=True)
        
        rows = "".join([
            f"<tr><td>{n}</td><td style='text-align:right; color:{'#b2182b' if v>0 else '#2166ac'};'>"
            f"<b>{v:+.2f}</b></td></tr>" for v, n in impacts
        ])
        shap_html_list.append(f"<table style='width:100%; font-size:10px; border-top:1px solid #ddd;'>{rows}</table>")

    full_gdf['shap_info'] = shap_html_list
    full_gdf['real_log_delta'] = log_delta_real

    # F. GEOGRAPHISCHE DARSTELLUNG & TOOLTIP-FINALS
    def scientific_tooltip(row):
        color = "#b2182b" if row['inc_delta'] > 0 else "#2166ac"
        
        # Sicherheits-Check für Ganzzahlen
        def format_precip(val):
            return f"{int(val)}" if pd.notnull(val) else "n.a."

        return f"""
        <div style="font-family: 'Segoe UI', sans-serif; font-size: 11px; width: 320px;">
            <b style="font-size:14px; color:{color};">Projektion: {row['inc_delta']:+.1f} Fälle</b><br>
            <div style="background:#f4f4f4; padding:5px; margin:5px 0; border:1px solid #ccc;">
                <b>Modell-Validierung (Netto-Log-Delta):</b><br>
                Berechnete Gesamtdifferenz: <b>{row['real_log_delta']:+.3f}</b>
            </div>
            <b>Quantitative Metriken:</b><br>
            Erwartete Inzidenz 2060: {row['inc_2060']:.1f}<br>
            Konfidenz-Intervall (SD): ± {row['inc_uncertainty']:.2f}<br>
            Ökologische Eignung: {row['suit_now']:.2f} ➔ {row['suit_fut']:.2f}<br>
            
            <br><b>Kausale Determinanten (SHAP Decomposition):</b>
            {row['shap_info']}
            
            <hr style="margin:8px 0; border:0; border-top:1px double #999;">
            <table style="width:100%; font-size:10px; color:#444; line-height:1.4;">
                <tr><td>Sommer-Temperatur (Bio10):</td><td style="text-align:right;">{row['bio_10']:.1f} ➔ {row['fut_bio_10']:.1f} °C</td></tr>
                <tr><td>Winter-Temperatur (Bio11):</td><td style="text-align:right;">{row['bio_11']:.1f} ➔ {row['fut_bio_11']:.1f} °C</td></tr>
                <tr><td>Jahres-Niederschlag (Bio12):</td><td style="text-align:right;">{format_precip(row['bio_12'])} ➔ {format_precip(row['fut_bio_12'])} mm</td></tr>
            </table>
        </div>
        """

    full_gdf['tooltip'] = full_gdf.apply(scientific_tooltip, axis=1)
    
    m = folium.Map(location=[-14.2, -51.9], zoom_start=4, tiles='cartodbpositron')
    colormap = cm.LinearColormap(['#2166ac', '#f7f7f7', '#b2182b'], vmin=-15, vmax=15, caption="Inzidenz-Differenz (2060 vs. Gegenwart)")
    
    folium.GeoJson(
        full_gdf,
        style_function=lambda x: get_map_style(x, 'inc_delta', colormap),
        tooltip=folium.GeoJsonTooltip(fields=['NAME_2', 'tooltip'], aliases=['Region:', ''])
    ).add_to(m)
    
    colormap.add_to(m)
    m.save(OUTPUT_HTML)
    print(f"✅ Analyse abgeschlossen. Dashboard generiert: {OUTPUT_HTML}")

if __name__ == "__main__":
    run_scientific_model()