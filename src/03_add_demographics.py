import pandas as pd
import os
import re

# --- PFADE ---
BASE_DIR = r"C:\Users\Thomas\Forschungsprojekt- Leishmaniose"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

POP_FILE = os.path.join(RAW_DIR, "estimativa_dou_2021.xls")
MASTER_FILE = os.path.join(DATA_DIR, "master_table_dynamic_regression.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "master_table_with_population.csv")

def clean_population_string(val):
    """Bereinigt Zahlensalat wie '12.345(1)' -> 12345"""
    if pd.isna(val): return 0
    val_str = str(val)
    
    val_str = re.sub(r'\([^)]*\)', '', val_str) 
    val_str = val_str.replace('.', '').strip()
    try:
        return int(val_str)
    except:
        return 0

def to_clean_str(val):
    """Macht '12.0' zu '12'"""
    return str(val).split('.')[0].strip()

def run_merge():
    print("--- Starte intelligenten Daten-Merge (Version 4 - Anti-Crash) ---")
    
    # 1. Excel laden
    print(f"Lade Excel: {POP_FILE}")
    xls = pd.ExcelFile(POP_FILE)
    
    # Blatt finden
    target_sheet = [s for s in xls.sheet_names if "MUNIC" in s.upper()]
    target_sheet = target_sheet[0] if target_sheet else xls.sheet_names[1]
    print(f"Nutze Tabellenblatt: {target_sheet}")
    
    
    df_raw = pd.read_excel(POP_FILE, sheet_name=target_sheet, header=None)
    
    
    header_row_idx = None
    for idx, row in df_raw.iterrows():
        row_str = row.astype(str).str.upper().values
        if any("COD" in s and "MUNIC" in s for s in row_str):
            header_row_idx = idx
            break
            
    if header_row_idx is None:
        print("FEHLER: Konnte Header-Zeile nicht finden. Breche ab.")
        return

    print(f"Header gefunden in Zeile: {header_row_idx}")
    
    
    df_pop = pd.read_excel(POP_FILE, sheet_name=target_sheet, header=header_row_idx, dtype=str)
    
    # Spalten identifizieren
    col_uf = [c for c in df_pop.columns if 'COD' in str(c).upper() and 'UF' in str(c).upper()][0]
    col_mun = [c for c in df_pop.columns if 'COD' in str(c).upper() and 'MUNIC' in str(c).upper()][0]
    col_pop = [c for c in df_pop.columns if 'POPULA' in str(c).upper()][0]
    
    print(f"Spalten Mapping: UF='{col_uf}' | MUN='{col_mun}' | POP='{col_pop}'")

    # 2. Daten reinigen 
    df_clean = df_pop.copy()
    

    df_clean = df_clean[df_clean[col_mun].str.isnumeric().fillna(False)]
    
    # Keys bauen
    df_clean['uf_clean'] = df_clean[col_uf].apply(to_clean_str)
    df_clean['mun_clean'] = df_clean[col_mun].apply(to_clean_str).str.zfill(5) 
    
    df_clean['join_key'] = df_clean['uf_clean'] + df_clean['mun_clean']
    
    df_clean['join_key'] = df_clean['join_key'].str.slice(0, 6)
    
    df_clean['population'] = df_clean[col_pop].apply(clean_population_string)
    
    # Duplikate weg
    df_final_pop = df_clean[['join_key', 'population']].drop_duplicates(subset='join_key')
    
    print(f"Excel Beispiele (validiert): {df_final_pop['join_key'].head(3).tolist()}")

    # 3. Master laden
    print("Lade Master Tabelle...")
    try:
        df_master = pd.read_csv(MASTER_FILE, sep=',')
        if len(df_master.columns) < 5: df_master = pd.read_csv(MASTER_FILE, sep=';')
    except:
        print("Fehler beim Laden der Master CSV.")
        return

    df_master['join_key'] = df_master['code_gemeinde'].apply(to_clean_str).str.slice(0, 6)
    print(f"Master Beispiele: {df_master['join_key'].head(3).tolist()}")

    # 4. Merge
    print("Merge läuft...")
    df_merged = pd.merge(df_master, df_final_pop, on='join_key', how='left')
    
    # Stats
    found = df_merged[df_merged['population'] > 0].shape[0]
    total = df_merged.shape[0]
    print(f"MATCH QUOTE: {found} von {total} Einträgen haben jetzt Bevölkerungsdaten ({(found/total)*100:.1f}%)")

    # 5. Inzidenz berechnen
    df_merged['population'] = df_merged['population'].fillna(0)
    df_merged['incidence'] = df_merged.apply(
        lambda row: (row['Faelle'] / row['population'] * 100000) if row['population'] > 0 else 0, 
        axis=1
    )

    df_merged.drop(columns=['join_key', 'uf_clean', 'mun_clean'], inplace=True, errors='ignore')
    df_merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Gespeichert: {OUTPUT_FILE}")

    # Check
    sample = df_merged[df_merged['incidence'] > 0].head(5)
    if not sample.empty:
        print("\n--- ERFOLG! Hier sind echte Inzidenzen: ---")
        print(sample[['code_gemeinde', 'Faelle', 'population', 'incidence']])
    else:
        print("\nWARNUNG: Keine positiven Inzidenzen berechnet.")

if __name__ == "__main__":
    run_merge()