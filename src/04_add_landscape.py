import os
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
from tqdm import tqdm
import fiona

# PFADE 
BASE_DIR = r"C:\Users\Thomas\Forschungsprojekt- Leishmaniose"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

MASTER_FILE = os.path.join(DATA_DIR, "master_table_with_population.csv")
SHAPEFILE = os.path.join(RAW_DIR, "map_brazil.gpkg") 
RASTER_FILE = os.path.join(RAW_DIR, "wc2.1_2.5m_elev.tif")

OUTPUT_FILE = os.path.join(DATA_DIR, "master_table_with_landscape.csv")

def get_zonal_stats(shape_geometry, raster_path):
    with rasterio.open(raster_path) as src:
        try:
            out_image, out_transform = mask(src, [shape_geometry], crop=True)
            valid_data = out_image[out_image > -9999]
            if valid_data.size == 0: return np.nan
            return np.mean(valid_data)
        except:
            return np.nan

def run_extraction():
    print("--- Starte Landschaftsanalyse (Version Level 2 Fix) ---")
    
    # 1. Layer Auswahl
    print(f"Prüfe Datei: {SHAPEFILE}")
    try:
        layers = fiona.listlayers(SHAPEFILE)
        print(f"Verfügbare Layer: {layers}")
        
        
        target_layer = None
        for l in layers:
            if "ADM_2" in l or "adm_2" in l.lower() or l.endswith("_2"):
                target_layer = l
                break
        
        if target_layer is None:
            print("WARNUNG: Kein Layer mit '_2' gefunden. Nehme den letzten (oft der detaillierteste).")
            target_layer = layers[-1]

        print(f"--> Wähle Layer: '{target_layer}' (Gemeinde-Ebene)")
        
        gdf = gpd.read_file(SHAPEFILE, layer=target_layer)
        print(f"Landkarte geladen: {len(gdf)} Gemeinden.")

    except Exception as e:
        print(f"Kritischer Fehler: {e}")
        return

    # 2. Spalte mit Code finden
    code_col = None
    
    possible_cols = [c for c in gdf.columns if 'CC_2' in c or 'CD_MUN' in c or 'code' in c.lower()]
    
    if possible_cols:
        code_col = possible_cols[0]
    else:
        
        for c in gdf.columns:
            if 'ID' in c and '2' in c:
                code_col = c
                break
        if not code_col: 
            # Notfall: Nimm Spalte 0
            code_col = gdf.columns[0]

    print(f"Nutze Spalte '{code_col}' als Gemeinde-Code.")

    # 3. Master laden
    print("Lade Master-Tabelle...")
    df = pd.read_csv(MASTER_FILE)
    
    # Harmonisierung
    gdf['code_short'] = gdf[code_col].astype(str).str.replace(r'\.0', '', regex=True).str.slice(0, 6)
    df['code_short'] = df['code_gemeinde'].astype(str).str.replace(r'\.0', '', regex=True).str.slice(0, 6)
    
    # 4. Filtern
    relevant_codes = set(df['code_short'].unique())
    gdf_filtered = gdf[gdf['code_short'].isin(relevant_codes)].copy()
    
    print(f"Berechne Höhe für {len(gdf_filtered)} gematchte Gemeinden...")
    
    if len(gdf_filtered) == 0:
        print("FEHLER: Immer noch keine Übereinstimmung.")
        print(f"Karte Codes (Bsp): {gdf['code_short'].head(5).tolist()}")
        print(f"CSV Codes (Bsp):   {df['code_short'].head(5).tolist()}")
        return

    # 5. Berechnung
    results = []
    print("Starte Berechnung (kann 1-3 Min dauern)...")
    for idx, row in tqdm(gdf_filtered.iterrows(), total=len(gdf_filtered)):
        elev = get_zonal_stats(row.geometry, RASTER_FILE)
        results.append({'code_short': row['code_short'], 'elevation': elev})
        
    df_landscape = pd.DataFrame(results)
    
    # 6. Merge & Speichern
    df_final = pd.merge(df, df_landscape, on='code_short', how='left')
    df_final['elevation'] = df_final['elevation'].fillna(df_final['elevation'].mean())
    df_final.drop(columns=['code_short'], inplace=True)
    
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ FERTIG! Datei gespeichert: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_extraction()