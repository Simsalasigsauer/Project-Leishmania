import xarray as xr
import rioxarray as rio
import glob
import os
import numpy as np

print("--- START: Skript zur Berechnung der jährlichen BIO-Variablen (Minimale Korrektur) ---")
print("Dieser Prozess kann je nach Systemleistung einige Zeit dauern.")

# =============================================================================
# === 1. PFADE DEFINIEREN
# =============================================================================
HISTORICAL_CLIMATE_DIR = r"C:\Users\Thomas\Forschungsprojekt- Leishmaniose\data\raw\worldclim_monthly"
OUTPUT_DIR = r"C:\Users\Thomas\Forschungsprojekt- Leishmaniose\data\processed\yearly_biovars"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# === 2. JAHRE DEFINIEREN
# =============================================================================
YEARS = range(1951, 2024)

# =============================================================================
# === 3. DATEN LADEN UND BIO-VARIABLEN BERECHNEN
# =============================================================================
all_tmin_files = sorted(glob.glob(os.path.join(HISTORICAL_CLIMATE_DIR, "*tmin*.tif")))
all_tmax_files = sorted(glob.glob(os.path.join(HISTORICAL_CLIMATE_DIR, "*tmax*.tif")))
all_prec_files = sorted(glob.glob(os.path.join(HISTORICAL_CLIMATE_DIR, "*prec*.tif")))

for year in YEARS:
    print(f"\n--- Verarbeite Jahr: {year} ---")
    
    year_tmin_paths = [f for f in all_tmin_files if f"_{year}-" in os.path.basename(f)]
    year_tmax_paths = [f for f in all_tmax_files if f"_{year}-" in os.path.basename(f)]
    year_prec_paths = [f for f in all_prec_files if f"_{year}-" in os.path.basename(f)]

    if len(year_tmin_paths) != 12 or len(year_tmax_paths) != 12 or len(year_prec_paths) != 12:
        print(f"    WARNUNG: Nicht alle 12 Monatsdateien für das Jahr {year} gefunden. Überspringe.")
        continue

    # Lade die Karten und behalte die "NoData"-Information
    
    tmin_ds = xr.concat([rio.open_rasterio(p, masked=True) for p in year_tmin_paths], dim='month')
    tmax_ds = xr.concat([rio.open_rasterio(p, masked=True) for p in year_tmax_paths], dim='month')
    prec_ds = xr.concat([rio.open_rasterio(p, masked=True) for p in year_prec_paths], dim='month')
    
    tavg_ds = (tmin_ds + tmax_ds) / 2.0
    print(f"    Monatsdaten für {year} geladen. Beginne Berechnung...")

    #  Berechne die BIO-Variablen 
    bio1 = tavg_ds.mean(dim='month')
    bio2 = (tmax_ds - tmin_ds).mean(dim='month')
    bio4 = tavg_ds.std(dim='month') * 100
    bio7 = tmax_ds.max(dim='month') - tmin_ds.min(dim='month')
    tavg_rolled = tavg_ds.roll(month=-1, roll_coords=False)
    quarterly_means = (tavg_ds + tavg_rolled + tavg_rolled.roll(month=-1, roll_coords=False)) / 3
    bio10 = quarterly_means.max(dim='month')
    bio11 = quarterly_means.min(dim='month')
    bio12 = prec_ds.sum(dim='month')
    prec_mean = prec_ds.mean(dim='month')
    bio15 = (prec_ds.std(dim='month') / (prec_mean + 1e-9)) * 100
    bio15 = bio15.where(np.isfinite(bio15), 0)
    
    print(f"    BIO-Variablen für {year} berechnet. Speichere Rasterdateien...")

    biovars = {'bio_1': bio1, 'bio_2': bio2, 'bio_4': bio4, 'bio_7': bio7, 
               'bio_10': bio10, 'bio_11': bio11, 'bio_12': bio12, 'bio_15': bio15}

    for name, data_array in biovars.items():
        output_path = os.path.join(OUTPUT_DIR, f"{name}_{year}.tif")
        
        data_array = data_array.rio.write_nodata(tmin_ds.rio.nodata, encoded=True)
        
        data_array.rio.write_crs(tmin_ds.rio.crs, inplace=True)
        data_array.rio.to_raster(output_path, compress='lzw')
        
print("\n--- Alle jährlichen BIO-Variablen wurden erfolgreich erstellt! ---")
print(f"Die Dateien befinden sich in: {OUTPUT_DIR}")