import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
import os
import pickle
from tqdm import tqdm

# --- PFADE ---
BASE_DIR = r"C:\Users\Thomas\Forschungsprojekt- Leishmaniose"
CACHE_FILE = os.path.join(BASE_DIR, "data", "processed", "temp_shap_data.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_curves():
    print("--- Erstelle 'World Class' Response Curves (Einheitliche Achsen) ---")
    
    # 1. Cache laden 
    if not os.path.exists(CACHE_FILE):
        print(" Bitte erst V4 oder V11 laufen lassen für den Cache!")
        return

    with open(CACHE_FILE, 'rb') as f:
        data = pickle.load(f)
        full_gdf = data['full_gdf']
        features_train = data['features_train']

    # 2. Modell trainieren
    print("Trainiere Modell für Analyse...")
    train_df = full_gdf.dropna(subset=features_train + ['Faelle', 'incidence'])
    X = train_df[features_train]
    y = np.log1p(train_df['incidence'])
    
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 3. Kurven berechnen
    base_row = X.median().to_frame().T 
    
    results = {}
    global_max_y = 0
    
    print("Berechne Kurven...")
    for feat in tqdm(features_train):
        # Min und Max Werte für dieses Feature im Datensatz
        min_val = X[feat].min()
        max_val = X[feat].max()
        
        # 100 Schritte von Min bis Max
        x_vals = np.linspace(min_val, max_val, 100)
        y_vals = []
        
        for val in x_vals:
            # Wir nehmen die Durchschnitts-Gemeinde und ändern nur DIESEN Wert
            temp_row = base_row.copy()
            temp_row[feat] = val
            
            # Vorhersage (Log zurückrechnen!)
            pred_log = model.predict(temp_row)[0]
            pred_inc = np.expm1(pred_log)
            y_vals.append(pred_inc)
            
        results[feat] = (x_vals, y_vals)
        
        # Max Y merken für einheitliche Achse
        if max(y_vals) > global_max_y:
            global_max_y = max(y_vals)

    print(f"Maximaler Ausschlag (Y-Achse wird fixiert auf): {global_max_y:.1f}")

    # 4. Plotten
    num_feats = len(features_train)
    cols = 3
    rows = (num_feats // cols) + 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()
    
    # Mapping 
    nice_names = {
        'bio_1': 'Jahres-Temp (°C)', 'bio_2': 'Tagesschwankung',
        'bio_4': 'Saisonalität (Klima-Chaos)', 'bio_7': 'Temp-Jahresrange', 
        'bio_10': 'Sommerhitze (Max Quartal)', 'bio_11': 'Winterkälte (Min Quartal)', 
        'bio_12': 'Jahresniederschlag (mm)', 'bio_15': 'Regen-Saisonalität',
        'elevation': 'Höhe (m)', 'pop_density': 'Bevölkerungsdichte'
    }

    for i, feat in enumerate(features_train):
        ax = axes[i]
        x_vals, y_vals = results[feat]
        
        # Plot
        ax.plot(x_vals, y_vals, color='#003366', linewidth=2.5)
        ax.fill_between(x_vals, y_vals, color='#e6f2ff', alpha=0.5)
        
        ax.set_ylim(0, global_max_y * 1.05)
        
        # Titel
        title = nice_names.get(feat, feat)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if i % cols == 0:
            ax.set_ylabel("Vorhergesagte Inzidenz")

    # Leere Slots ausblenden
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.suptitle("Ökologische Nischen: Einfluss der Faktoren (Ceteris Paribus)", fontsize=16)
    
    out_path = os.path.join(OUTPUT_DIR, "final_response_curves_uniform.png")
    plt.savefig(out_path, dpi=300)
    print(f"✅ Grafik gespeichert: {out_path}")

if __name__ == "__main__":
    generate_curves()