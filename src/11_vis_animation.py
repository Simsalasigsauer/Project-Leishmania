import pandas as pd
import numpy as np
import rasterio
import pickle
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# KONFIGURATION
# =============================================================================
BASE_DIR = r"C:\Users\Thomas\Forschungsprojekt- Leishmaniose"
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(PROCESSED_DATA_DIR, "rf_model_final.pkl")
YEARLY_BIO_DIR = os.path.join(PROCESSED_DATA_DIR, "yearly_biovars")
YEARLY_TIFF_DIR = os.path.join(PROCESSED_DATA_DIR, "yearly_risk_maps")
os.makedirs(YEARLY_TIFF_DIR, exist_ok=True)

# HIER DEN GEWÜNSCHTEN ZEITRAUM FESTLEGEN
years_to_process = range(1951, 2024) 

print("=" * 80)
print("ERSTELLUNG JAHRESWEISER RISIKO-TIFF-DATEIEN (FINALE VERSION - 1:1 LOGIK)")
print("=" * 80)

# =============================================================================
# 1. MODELL LADEN
# =============================================================================
print("\n--- PHASE 1: Lade Modell ---")
with open(MODEL_PATH, 'rb') as f:
    model_package = pickle.load(f)
rf_model = model_package['model']
scaler = model_package['scaler']
bio_columns = model_package['bio_columns']
temp_variables_to_scale = model_package.get('temp_variables_to_scale', [])
print("Modell geladen.")

# =============================================================================
# 2. DEFINIERE DIE KARTENERSTELLUNGS-FUNKTION (EXAKTE KOPIE)
# =============================================================================
def create_risk_map_final(model, climate_files, output_path, scaler, bio_cols):
    """Dies ist eine exakte Kopie der Logik aus der funktionierenden train.py"""
    
    print(f"  Erstelle: {os.path.basename(output_path)}...")
    template_file = climate_files[0]
    
    with rasterio.open(template_file) as src:
        meta = src.meta.copy()
        meta.update(count=1, dtype='float32', compress='deflate', nodata=-9999.0)
        height, width = src.height, src.width
    
    climate_data = np.zeros((len(bio_cols), height, width), dtype=np.float32)
    
    for i, bio_var in enumerate(bio_cols):
        matching_file = [f for f in climate_files if os.path.basename(f).startswith(bio_var)][0]
        with rasterio.open(matching_file) as src:
            climate_data[i, :, :] = src.read(1).astype(np.float32)

    pixels = climate_data.reshape(len(bio_cols), -1).T
    valid_mask = ~np.any((pixels < -9000) | np.isnan(pixels), axis=1)
    
    predictions = np.full(pixels.shape[0], -9999.0, dtype=np.float32)
    
    if np.any(valid_mask):
        pixel_data_to_predict = pixels[valid_mask].copy()
        columns_to_scale_indices = [i for i, col in enumerate(bio_cols) if col in temp_variables_to_scale]
        if columns_to_scale_indices:
            pixel_data_to_predict[:, columns_to_scale_indices] *= 10
        
        pixels_scaled = scaler.transform(pixel_data_to_predict)
        proba = model.predict_proba(pixels_scaled)
        predictions[valid_mask] = proba[:, 1]
    
    risk_map = predictions.reshape(height, width)
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(risk_map.astype(np.float32), 1)
    
    print(f"    ✓ Gespeichert: {output_path}")

# =============================================================================
# 3. HAUPTSCHLEIFE ZUR KARTENERSTELLUNG
# =============================================================================
print("\n--- PHASE 3: Erstelle Risiko-Raster für jedes Jahr ---")
for year in years_to_process:
    yearly_climate_files = glob.glob(os.path.join(YEARLY_BIO_DIR, f"*_{year}.tif"))
    
    if len(yearly_climate_files) < len(bio_columns):
        print(f"WARNUNG: Nicht alle Klimadaten für das Jahr {year} gefunden. Überspringe.")
        continue
    
    output_filepath = os.path.join(YEARLY_TIFF_DIR, f"risk_map_{year}.tif")
    create_risk_map_final(rf_model, yearly_climate_files, output_filepath, scaler, bio_columns)

print("\n" + "="*80)
print("ERSTELLUNG ALLER JAHRESKARTEN ABGESCHLOSSEN")
print("="*80)

# =============================================================================
# 4. ERSTELLE ANIMIERTE ZEITREIHE (GIF)
# =============================================================================
print("\n--- PHASE 4: Erstelle animierte Zeitreihe (GIF) ---")
try:
    import imageio
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    images = []
    
    # Referenz für Plot-Grenzen holen
    try:
        any_tiff = glob.glob(os.path.join(YEARLY_TIFF_DIR, "*.tif"))[0]
        with rasterio.open(any_tiff) as src:
            bounds = src.bounds
    except IndexError:
        print("    Keine TIFF-Dateien zum Erstellen des GIFs gefunden.")
        exit() # Beenden, wenn keine Karten da sind

    for year in years_to_process:
        tiff_path = os.path.join(YEARLY_TIFF_DIR, f"risk_map_{year}.tif")
        if os.path.exists(tiff_path):
            print(f"  Verarbeite Frame für Jahr {year}...")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_facecolor('white') # Setzt Ozean/Hintergrund auf weiß
            
            with rasterio.open(tiff_path) as src:
                data = src.read(1)
                data_masked = np.ma.masked_where(data == -9999.0, data)
                
                cmap = LinearSegmentedColormap.from_list('risk', ['blue', 'yellow', 'red'], N=100)
                im = ax.imshow(data_masked, cmap=cmap, vmin=0, vmax=1, 
                               extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
                
                ax.set_title(f'Leishmaniose-Risiko {year}', fontsize=16, fontweight='bold')
                ax.set_xlabel('Längengrad'); ax.set_ylabel('Breitengrad')
                
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Risiko-Wahrscheinlichkeit', rotation=270, labelpad=15)
                
                # Speichere Frame temporär ab
                temp_path = os.path.join(YEARLY_TIFF_DIR, f'temp_{year}.png')
                plt.savefig(temp_path, dpi=100, bbox_inches='tight')
                plt.close(fig) # Schließe die Figur, um Speicher freizugeben
                
                # Lade das gespeicherte Bild und füge es zur Liste hinzu
                images.append(imageio.imread(temp_path))
                os.remove(temp_path) # Lösche die temporäre Datei
    
    if images:
        gif_path = os.path.join(PROCESSED_DATA_DIR, 'risk_evolution_total.gif')
        print(f"\n  Erstelle finale GIF-Datei... (Dies kann dauern)")
        imageio.mimsave(gif_path, images, duration=1.0)  # 1 Sekunde pro Frame
        print(f"  ✓ Animierte GIF erstellt: {gif_path}")
    else:
        print("  Keine Bilder zum Erstellen eines GIFs gefunden.")
    
except ImportError:
    print("\n  HINWEIS: Für die GIF-Erstellung, installieren Sie bitte 'imageio' und 'matplotlib'.")
    print("  (pip install imageio imageio-ffmpeg matplotlib)")

print("\n" + "="*80)
print("GESAMTER PROZESS ABGESCHLOSSEN")
print("="*80)