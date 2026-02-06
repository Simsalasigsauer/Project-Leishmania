import pandas as pd
import geopandas as gpd
import glob
import os
import fiona
from rasterstats import zonal_stats
import warnings


warnings.filterwarnings("ignore", category=UserWarning, module='rasterstats')

print("--- START: Skript zur Erstellung der dynamischen Master-Tabelle ---")

# =============================================================================
# === 1. PFADE DEFINIEREN
# =============================================================================
BASE_DIR = r"C:\Users\Thomas\Forschungsprojekt- Leishmaniose"
MAP_PATH = os.path.join(BASE_DIR, "data", "raw", "map_brazil.gpkg")
CASES_PATH = os.path.join(BASE_DIR, "data", "raw", "leishmaniacases_brazil_2007-2022_communes.csv")
YEARLY_BIO_DIR = os.path.join(BASE_DIR, "data", "processed", "yearly_biovars")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
OUTPUT_TABLE_PATH = os.path.join(PROCESSED_DATA_DIR, "master_table_dynamic_regression.csv")

# =============================================================================
# === 2. BASIS-TABELLE ERSTELLEN (Fälle + Geometrie im "Long-Format")
# =============================================================================
print("\n--- PHASE 1: Erstelle Basis-Tabelle aus Rohdaten ---")

try:
    df_cases = pd.read_csv(CASES_PATH, encoding="latin-1", sep=";", skiprows=3, skipfooter=18, engine="python")
    available_layers = fiona.listlayers(MAP_PATH)
    commune_layer = [layer for layer in available_layers if layer.endswith('_2')][0]
    df_map = gpd.read_file(MAP_PATH, layer=commune_layer)
except Exception as e:
    print(f"FEHLER: Konnte Rohdaten nicht laden: {e}")
    exit()

df_map['code_gemeinde'] = df_map['CC_2'].astype(str).str[:-1]
df_cases['code_gemeinde'] = df_cases['Município de notificação'].str[:6]
df_merged = pd.merge(df_map, df_cases, on="code_gemeinde")

years_to_process = range(2007, 2019) 
years_str = [str(year) for year in years_to_process]
columns_to_keep = ["code_gemeinde", "NAME_2", "geometry"] + years_str
df_wide_clean = df_merged[columns_to_keep]

id_vars = ["code_gemeinde", "NAME_2", "geometry"]
df_long_base = df_wide_clean.melt(id_vars=id_vars, value_vars=years_str, var_name='Jahr', value_name='Faelle')
df_long_base['Jahr'] = pd.to_numeric(df_long_base['Jahr'])
df_long_base['Faelle'] = pd.to_numeric(df_long_base['Faelle'], errors='coerce').fillna(0).astype(int)

print("✓ Basis-Tabelle erfolgreich erstellt.")

# =============================================================================
# === 3. DYNAMISCHE KLIMADATEN HINZUFÜGEN
# =============================================================================
print("\n--- PHASE 2: Füge dynamische, jährliche Klimadaten hinzu ---")
print("(Dies kann einige Minuten dauern)")

all_data_rows = []
for year in sorted(df_long_base['Jahr'].unique()):
    yearly_bio_files = glob.glob(os.path.join(YEARLY_BIO_DIR, f"*_{year}.tif"))
    
    if len(yearly_bio_files) < 8:
        print(f"    WARNUNG: Nicht alle 8 BIO-Dateien für das Jahr {year} gefunden. Überspringe.")
        continue
        
    print(f"  Verarbeite Zonal Statistics für das Jahr {year}...")
    df_year = df_long_base[df_long_base['Jahr'] == year].copy()
    
    for bio_path in yearly_bio_files:
        column_name = os.path.basename(bio_path).replace(f"_{year}.tif", "")
        stats = zonal_stats(df_year, bio_path, stats="mean", all_touched=True)
        mean_values = [s['mean'] if s and s.get('mean') is not None else 0 for s in stats]
        df_year[column_name] = mean_values
        
    all_data_rows.append(df_year)

# Kombiniere alle Jahres-Daten zu einer finalen Master-Tabelle
master_table = pd.concat(all_data_rows, ignore_index=True)
print("✓ Finale Master-Tabelle mit dynamischen Klimadaten erfolgreich erstellt.")

# =============================================================================
# === 4. ERGEBNIS ANZEIGEN UND SPEICHERN
# =============================================================================
print("\n--- ERGEBNIS: FINALE MASTER-TABELLE ---")

# Entferne die Geometrie-Spalte für eine übersichtlichere Anzeige und Speicherung
final_table_to_save = master_table.drop(columns='geometry')

print(f"Anzahl der Zeilen: {len(final_table_to_save)}")
print("Spaltennamen:", final_table_to_save.columns.tolist())
print("\n--- Erste 10 Zeilen der Tabelle: ---")
print(final_table_to_save.head(10))

# Speichern der finalen Tabelle
final_table_to_save.to_csv(OUTPUT_TABLE_PATH, index=False)
print(f"\n✓ Master-Tabelle erfolgreich gespeichert unter: {OUTPUT_TABLE_PATH}")
print("\n--- Skript abgeschlossen. Die Tabelle ist jetzt bereit für das Machine Learning. ---")