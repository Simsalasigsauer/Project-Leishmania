import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# KONFIGURATION
# =============================================================================
BASE_DIR = r"C:\Users\Thomas\Forschungsprojekt- Leishmaniose"
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

MAP_PATH = os.path.join(RAW_DATA_DIR, "map_brazil.gpkg")
FUTURE_TIF = os.path.join(RAW_DATA_DIR, "worldclim_future", "wc2.1_2.5m_bioc_MPI-ESM1-2-HR_ssp245_2041-2060.tif")
OUTPUT_CSV = os.path.join(PROCESSED_DATA_DIR, "future_climate_2041-2060_FIXED.csv")

print("=" * 80)
print("VERBESSERTE EXTRAKTION VON FUTURE-KLIMADATEN")
print("=" * 80)

# =============================================================================
# 1. LADE GEODATEN
# =============================================================================
print("\n--- PHASE 1: Lade Geodaten ---")

import fiona
available_layers = fiona.listlayers(MAP_PATH)
layer = [l for l in available_layers if l.endswith('_2')][0] if any(l.endswith('_2') for l in available_layers) else available_layers[0]

brazil_map = gpd.read_file(MAP_PATH, layer=layer)
print(f"Gemeinden geladen: {len(brazil_map)}")
print(f"Original CRS: {brazil_map.crs}")

# Code-Spalte
if 'CC_2' in brazil_map.columns:
    brazil_map['code_gemeinde'] = brazil_map['CC_2'].astype(str).str[:-1]
elif 'GID_2' in brazil_map.columns:
    brazil_map['code_gemeinde'] = brazil_map['GID_2'].apply(lambda x: x.split('.')[-1] if '.' in str(x) else str(x))
else:
    brazil_map['code_gemeinde'] = brazil_map.index.astype(str)

name_col = None
for col in ['NAME_2', 'NM_MUN', 'NOME_2']:
    if col in brazil_map.columns:
        name_col = col
        break

# =============================================================================
# 2. ÖFFNE TIF UND PRÜFE
# =============================================================================
print("\n--- PHASE 2: Öffne TIF und analysiere ---")

with rasterio.open(FUTURE_TIF) as src:
    print(f"TIF Eigenschaften:")
    print(f"  Bands: {src.count}")
    print(f"  CRS: {src.crs}")
    print(f"  NoData: {src.nodata}")
    print(f"  Bounds: {src.bounds}")
    
    # Prüfe CRS
    if str(src.crs) != str(brazil_map.crs):
        print(f"\n CRS-Mismatch! Reprojiziere Gemeinden...")
        print(f"   Von: {brazil_map.crs} → Nach: {src.crs}")
        brazil_map = brazil_map.to_crs(src.crs)
    
    # Lese Band 1 komplett für Brasilien
    from rasterio.windows import from_bounds
    brazil_bounds = (-73, -33, -34, 5)
    window = from_bounds(*brazil_bounds, src.transform)
    band1_sample = src.read(1, window=window)
    
    print(f"\n Band 1 Statistik für Brasilien-Region:")
    print(f"  Array Shape: {band1_sample.shape}")
    print(f"  Min: {np.nanmin(band1_sample):.2f}")
    print(f"  Max: {np.nanmax(band1_sample):.2f}")
    print(f"  Mean: {np.nanmean(band1_sample):.2f}")
    print(f"  NaN count: {np.isnan(band1_sample).sum()} ({np.isnan(band1_sample).sum()/band1_sample.size*100:.1f}%)")
    
    # Prüfe ob Werte sinnvoll aussehen
    valid_vals = band1_sample[~np.isnan(band1_sample)]
    if len(valid_vals) > 0:
        if valid_vals.mean() > 100:
            print(f"  → Sieht aus wie ABSOLUTE Werte × 10")
            print(f"  → Als °C: {valid_vals.min()/10:.1f} bis {valid_vals.max()/10:.1f}°C")
        elif valid_vals.mean() < 50:
            print(f"  → Sieht aus wie ANOMALIEN oder ABSOLUTE Werte")
            print(f"  → Als °C: {valid_vals.min():.1f} bis {valid_vals.max():.1f}°C")

# =============================================================================
# 3. EXTRAHIERE MIT VERBESSERTER METHODE
# =============================================================================
print("\n--- PHASE 3: Extrahiere mit verbesserter Methode ---")
print("Methode: Sampelt zentrale Pixel statt Maske für bessere Abdeckung")

results = []
problems = []

with rasterio.open(FUTURE_TIF) as src:
    
    for idx, row in tqdm(brazil_map.iterrows(), total=len(brazil_map), desc="Gemeinden"):
        code = row['code_gemeinde']
        name = row[name_col] if name_col else f"Muni_{idx}"
        geometry = row['geometry']
        
        try:
            # METHODE 1: Maskiere Geometrie (Original)
            geom = [mapping(geometry)]
            out_image, out_transform = mask(src, geom, crop=True, all_touched=True)
            
            bio_values = {}
            valid_pixels = 0
            
            for band_idx in range(1, src.count + 1):
                band_data = out_image[band_idx - 1]
                
                # Filtere NaN UND unrealistische Werte
                valid_mask = ~np.isnan(band_data)
                
                # Zusätzlich: Filter für realistische Werte
                # Temperatur (Band 1-11): -50 bis 50°C oder 0 bis 500 (× 10)
                # Niederschlag (Band 12-19): 0 bis 10000mm
                if band_idx <= 11:
                    valid_mask &= (band_data >= -50) & (band_data <= 500)
                else:
                    valid_mask &= (band_data >= 0) & (band_data <= 10000)
                
                valid_data = band_data[valid_mask]
                
                if len(valid_data) > 0:
                    bio_values[f'bio_{band_idx}_future'] = np.mean(valid_data)
                    valid_pixels += len(valid_data)
                else:
                    bio_values[f'bio_{band_idx}_future'] = np.nan
            
            # Wenn keine Daten mit Maske, versuche Centroid
            if valid_pixels == 0:
                centroid = geometry.centroid
                py, px = src.index(centroid.x, centroid.y)
                
                try:
                    for band_idx in range(1, src.count + 1):
                        val = src.read(band_idx, window=((py, py+1), (px, px+1)))[0, 0]
                        
                        # Prüfe ob gültig
                        if not np.isnan(val):
                            if band_idx <= 11:
                                if -50 <= val <= 500:
                                    bio_values[f'bio_{band_idx}_future'] = val
                            else:
                                if 0 <= val <= 10000:
                                    bio_values[f'bio_{band_idx}_future'] = val
                except:
                    pass
            
            result = {
                'code_gemeinde': code,
                'NAME_2': name,
                **bio_values
            }
            results.append(result)
            
            # Track problems
            if all(np.isnan(v) for k, v in bio_values.items()):
                problems.append((name, code, "Alle NaN"))
            
        except Exception as e:
            result = {
                'code_gemeinde': code,
                'NAME_2': name,
                **{f'bio_{i}_future': np.nan for i in range(1, src.count + 1)}
            }
            results.append(result)
            problems.append((name, code, str(e)))

# =============================================================================
# 4. ERSTELLE DATAFRAME
# =============================================================================
print("\n--- PHASE 4: Analyse Ergebnisse ---")

df_future = pd.DataFrame(results)

missing = df_future['bio_1_future'].isna().sum()
print(f"\n EXTRAKTIONS-ERFOLG:")
print(f"  Gesamt: {len(df_future)} Gemeinden")
print(f"  Mit Daten: {len(df_future) - missing} ({(len(df_future) - missing)/len(df_future)*100:.1f}%)")
print(f"  Ohne Daten: {missing} ({missing/len(df_future)*100:.1f}%)")

if len(problems) > 0:
    print(f"\n PROBLEME bei {len(problems)} Gemeinden:")
    for i, (name, code, reason) in enumerate(problems[:5]):
        print(f"  {i+1}. {name} ({code}): {reason}")
    if len(problems) > 5:
        print(f"  ... und {len(problems)-5} weitere")

# Statistik für erfolgreiche Extraktionen
has_data = df_future['bio_1_future'].notna()
if has_data.sum() > 0:
    print(f"\n STATISTIK (nur Gemeinden mit Daten):")
    
    temp_vals = df_future.loc[has_data, 'bio_1_future']
    print(f"  bio_1_future:")
    print(f"    Min: {temp_vals.min():.2f}")
    print(f"    Max: {temp_vals.max():.2f}")
    print(f"    Mean: {temp_vals.mean():.2f}")
    print(f"    Median: {temp_vals.median():.2f}")
    
    # Diagnose: Absolute oder Anomalie?
    if temp_vals.mean() > 100:
        print(f"\n Interpretation: ABSOLUTE Werte × 10")
        print(f"   Als °C: {temp_vals.min()/10:.1f} bis {temp_vals.max()/10:.1f}°C")
    elif 10 < temp_vals.mean() < 100:
        print(f"\n Interpretation: ABSOLUTE Werte (schon in °C)")
        print(f"   Temperaturen: {temp_vals.min():.1f} bis {temp_vals.max():.1f}°C")
    else:
        print(f"\n Interpretation: ANOMALIEN (Änderungen in °C)")
        print(f"   Änderungen: {temp_vals.min():+.1f} bis {temp_vals.max():+.1f}°C")

# =============================================================================
# 5. SPEICHERE
# =============================================================================
print(f"\n--- PHASE 5: Speichere ---")

df_future.to_csv(OUTPUT_CSV, index=False)

print(f"\n Gespeichert: {OUTPUT_CSV}")
print(f"   {len(df_future) - missing} Gemeinden mit Daten")

# =============================================================================
# 6. VERGLEICH MIT VERGANGENHEIT
# =============================================================================
print("\n--- PHASE 6: Vergleich mit Vergangenheit ---")

try:
    TRAINING_TABLE = os.path.join(PROCESSED_DATA_DIR, "master_table_dynamic_regression.csv")
    df_past = pd.read_csv(TRAINING_TABLE)
    
    # Aggregiere Vergangenheit
    df_past_agg = df_past.groupby('code_gemeinde').agg({
        'bio_1': 'mean'
    }).reset_index()
    
    # Bereinige Codes
    def clean_code(x):
        try:
            return str(int(float(str(x).strip())))
        except:
            return str(x).strip()
    
    df_past_agg['code_gemeinde'] = df_past_agg['code_gemeinde'].apply(clean_code)
    df_future['code_gemeinde'] = df_future['code_gemeinde'].apply(clean_code)
    
    # Merge
    df_compare = df_past_agg.merge(
        df_future[['code_gemeinde', 'bio_1_future']],
        on='code_gemeinde',
        how='inner'
    )
    
    # Nur Gemeinden mit Future-Daten
    df_compare = df_compare[df_compare['bio_1_future'].notna()]
    
    if len(df_compare) > 0:
        print(f"\n Vergleich möglich für {len(df_compare)} Gemeinden:")
        print(f"\n VERGLEICH:")
        print(f"  Vergangenheit (bio_1):")
        print(f"    Mean: {df_compare['bio_1'].mean():.2f}°C")
        print(f"    Range: {df_compare['bio_1'].min():.1f} - {df_compare['bio_1'].max():.1f}°C")
        
        print(f"\n  Future (bio_1_future):")
        print(f"    Mean: {df_compare['bio_1_future'].mean():.2f}")
        print(f"    Range: {df_compare['bio_1_future'].min():.1f} - {df_compare['bio_1_future'].max():.1f}")
        
        # Diagnose
        if df_compare['bio_1_future'].mean() > 100:
            print(f"\n DIAGNOSE: Future sind ABSOLUTE Werte × 10")
            df_compare['temp_future_celsius'] = df_compare['bio_1_future'] / 10
            df_compare['temp_change'] = df_compare['temp_future_celsius'] - df_compare['bio_1']
            print(f"  Future in °C: {df_compare['temp_future_celsius'].mean():.2f}°C")
            print(f"  Änderung: {df_compare['temp_change'].mean():+.2f}°C")
        elif df_compare['bio_1_future'].mean() > 10:
            print(f"\n DIAGNOSE: Future sind ABSOLUTE Werte (schon in °C)")
            df_compare['temp_change'] = df_compare['bio_1_future'] - df_compare['bio_1']
            print(f"  Änderung: {df_compare['temp_change'].mean():+.2f}°C")
        else:
            print(f"\n DIAGNOSE: Future sind ANOMALIEN (Änderungen)")
            df_compare['temp_future'] = df_compare['bio_1'] + df_compare['bio_1_future']
            print(f"  Future: {df_compare['temp_future'].mean():.2f}°C")
            print(f"  Änderung: {df_compare['bio_1_future'].mean():+.2f}°C")
        
        print(f"\n INTERPRETATION:")
        print(f"  Diese Diagnose zeigt wie die TIF-Werte zu interpretieren sind!")
    else:
        print(" Keine Gemeinden für Vergleich gefunden")
        
except Exception as e:
    print(f" Vergleich nicht möglich: {e}")

print("\n" + "="*80)
print("FERTIG!")
print("="*80)