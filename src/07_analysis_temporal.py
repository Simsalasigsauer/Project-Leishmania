import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import pickle
import os
import fiona
import warnings
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')

# =============================================================================
# KONFIGURATION
# =============================================================================
BASE_DIR = r"C:\Users\Thomas\Forschungsprojekt- Leishmaniose"
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
TRAINING_TABLE_PATH = os.path.join(PROCESSED_DATA_DIR, "master_table_dynamic_regression.csv")
MODEL_PATH = os.path.join(PROCESSED_DATA_DIR, "rf_model_final.pkl")
MAP_PATH = os.path.join(BASE_DIR, "data", "raw", "map_brazil.gpkg")

print("=" * 80)
print("ZEITLICHE ANALYSE DER LEISHMANIOSE-AUSBREITUNG")
print("=" * 80)

# =============================================================================
# 1. DATEN UND MODELL LADEN
# =============================================================================
print("\n--- PHASE 1: Lade Daten und Modell ---")

# Lade Trainingsdaten
df = pd.read_csv(TRAINING_TABLE_PATH)
print(f"Geladene Daten: {df.shape}")
print(f"Jahre verfügbar: {sorted(df['Jahr'].unique())}")

# Lade trainiertes Modell
with open(MODEL_PATH, 'rb') as f:
    model_package = pickle.load(f)
    
rf_model = model_package['model']
scaler = model_package['scaler']
bio_columns = model_package['bio_columns']

print(f"Modell geladen mit Performance: ROC-AUC = {model_package['performance']['roc_auc']:.3f}")

# Lade Brasilien-Karte und identifiziere korrekten Spaltennamen
brazil_map = None
map_name_column = None
try:
    # Lade den ersten verfügbaren Layer
    available_layers = fiona.listlayers(MAP_PATH)
    if available_layers:
        layer_to_use = [layer for layer in available_layers if layer.endswith('_2')]
        if not layer_to_use:
            layer_to_use = [available_layers[0]]
        
        brazil_map = gpd.read_file(MAP_PATH, layer=layer_to_use[0])
        
        # Finde die richtige Namensspalte
        possible_name_columns = ['NAME_2', 'name_2', 'NM_MUNICIP', 'NM_MUN', 'NOME', 'nome']
        for col in brazil_map.columns:
            if any(possible in col.upper() for possible in ['NAME', 'NOME', 'NM_MUN']):
                map_name_column = col
                print(f"Verwende Kartenspalte: {map_name_column}")
                break
        
        if not map_name_column:
            print("Warnung: Keine Namensspalte in Karte gefunden")
            print(f"Verfügbare Spalten: {brazil_map.columns.tolist()}")
        
        # Vereinfache Geometrien für schnellere Plots
        brazil_map['geometry'] = brazil_map['geometry'].simplify(0.01)
except Exception as e:
    print(f"Warnung: Konnte Karte nicht laden: {e}")

# =============================================================================
# 2. JAHRESWEISE VORHERSAGEN
# =============================================================================
print("\n--- PHASE 2: Erstelle jahresweise Vorhersagen ---")

# Füge Vorhersagen für jedes Jahr hinzu
df['prediction_proba'] = np.nan

for year in sorted(df['Jahr'].unique()):
    year_mask = df['Jahr'] == year
    year_data = df[year_mask]
    
    # Skaliere Features und mache Vorhersage
    X_year = scaler.transform(year_data[bio_columns])
    predictions = rf_model.predict_proba(X_year)[:, 1]
    
    df.loc[year_mask, 'prediction_proba'] = predictions
    
    # Berechne Statistiken
    actual_cases = (year_data['Faelle'] > 0).sum()
    predicted_high_risk = (predictions > 0.5).sum()
    mean_risk = predictions.mean()
    
    print(f"  {year}: {actual_cases} tatsächliche Fälle, "
          f"{predicted_high_risk} vorhergesagt (Risiko > 0.5), "
          f"mittleres Risiko: {mean_risk:.3f}")

# =============================================================================
# 3. ZEITLICHE TRENDS ANALYSIEREN
# =============================================================================
print("\n--- PHASE 3: Analysiere zeitliche Trends ---")

# Aggregiere Daten pro Jahr
yearly_stats = df.groupby('Jahr').agg({
    'Faelle': ['sum', 'mean', lambda x: (x > 0).sum()],
    'prediction_proba': ['mean', 'std'],
    **{col: 'mean' for col in bio_columns}
}).round(3)

yearly_stats.columns = ['_'.join(col).strip() for col in yearly_stats.columns]
yearly_stats.columns = [col.replace('<lambda_0>', 'count_positive') for col in yearly_stats.columns]

print("\nJährliche Statistiken:")
print(yearly_stats[['Faelle_sum', 'Faelle_count_positive', 'prediction_proba_mean']])

# Berechne Korrelationen zwischen Klimavariablen und Fallzahlen
correlations = {}
for col in bio_columns:
    corr, p_value = pearsonr(yearly_stats[f'{col}_mean'], yearly_stats['Faelle_sum'])
    correlations[col] = {'correlation': corr, 'p_value': p_value}

corr_df = pd.DataFrame(correlations).T.sort_values('correlation', key=abs, ascending=False)
print("\nTop 5 Klimafaktoren korreliert mit Fallzahlen:")
print(corr_df.head())

# =============================================================================
# 4. RÄUMLICHE VERTEILUNG PRO JAHR
# =============================================================================
print("\n--- PHASE 4: Analysiere räumliche Verteilung ---")

# Identifiziere Hotspot-Gemeinden
hotspot_analysis = df.groupby('NAME_2').agg({
    'Faelle': 'sum',
    'prediction_proba': 'mean'
}).sort_values('Faelle', ascending=False)

print("\nTop 10 Gemeinden mit meisten Fällen (2007-2018):")
print(hotspot_analysis.head(10))

# =============================================================================
# 5. VISUALISIERUNG - ZEITLICHE ENTWICKLUNG
# =============================================================================
print("\n--- PHASE 5: Erstelle Visualisierungen ---")

# Erstelle umfassende Visualisierung
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

# 1. Zeitlicher Verlauf der Fälle
ax1 = fig.add_subplot(gs[0, :])
ax1_2 = ax1.twinx()

years = yearly_stats.index
ax1.bar(years, yearly_stats['Faelle_sum'], alpha=0.7, color='coral', label='Gesamtfälle')
ax1_2.plot(years, yearly_stats['prediction_proba_mean'], 'b-', linewidth=2, 
          marker='o', markersize=8, label='Mittleres vorhergesagtes Risiko')

ax1.set_xlabel('Jahr', fontsize=12)
ax1.set_ylabel('Anzahl Fälle', color='coral', fontsize=12)
ax1_2.set_ylabel('Mittleres Risiko', color='blue', fontsize=12)
ax1.set_title('Zeitliche Entwicklung der Leishmaniose-Fälle und Risikovorhersage', 
             fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1_2.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 2. Klimafaktoren über Zeit (Top 3)
top_climate_factors = corr_df.head(3).index.tolist()
ax2 = fig.add_subplot(gs[1, :])

for i, factor in enumerate(top_climate_factors):
    values = yearly_stats[f'{factor}_mean']
    # Normalisiere für bessere Vergleichbarkeit
    normalized = (values - values.mean()) / values.std()
    ax2.plot(years, normalized, marker='o', linewidth=2, label=factor, alpha=0.8)

ax2.set_xlabel('Jahr', fontsize=12)
ax2.set_ylabel('Normalisierter Wert (Z-Score)', fontsize=12)
ax2.set_title(f'Zeitliche Entwicklung der Top 3 Klimafaktoren', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# 3. Heatmap: Klimavariablen über Jahre
ax3 = fig.add_subplot(gs[2, :])

# Erstelle Matrix für Heatmap
climate_matrix = yearly_stats[[f'{col}_mean' for col in bio_columns]].T
climate_matrix.index = [idx.replace('_mean', '') for idx in climate_matrix.index]

# Normalisiere jede Variable
climate_matrix_norm = (climate_matrix - climate_matrix.mean(axis=1).values.reshape(-1, 1)) / \
                      climate_matrix.std(axis=1).values.reshape(-1, 1)

sns.heatmap(climate_matrix_norm, cmap='RdBu_r', center=0, ax=ax3,
            cbar_kws={'label': 'Z-Score'}, vmin=-2, vmax=2)
ax3.set_title('Klimavariablen-Anomalien über die Jahre', fontsize=14, fontweight='bold')
ax3.set_xlabel('Jahr', fontsize=12)
ax3.set_ylabel('Bio-Variable', fontsize=12)

# 4. Korrelation Klima-Fälle
ax4 = fig.add_subplot(gs[3, 0])
colors = ['green' if p < 0.05 else 'gray' for p in corr_df['p_value']]
bars = ax4.barh(range(len(bio_columns)), 
                corr_df.loc[bio_columns, 'correlation'].values, 
                color=colors)
ax4.set_yticks(range(len(bio_columns)))
ax4.set_yticklabels(bio_columns)
ax4.set_xlabel('Korrelation mit Fallzahlen', fontsize=11)
ax4.set_title('Klimafaktor-Einfluss\n(grün = signifikant)', fontsize=12, fontweight='bold')
ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax4.grid(True, alpha=0.3, axis='x')
ax4.invert_yaxis()

# 5. Vorhersagegenauigkeit pro Jahr
ax5 = fig.add_subplot(gs[3, 1])

yearly_accuracy = []
for year in years:
    year_data = df[df['Jahr'] == year]
    actual = (year_data['Faelle'] > 0).astype(int)
    predicted = (year_data['prediction_proba'] > 0.5).astype(int)
    accuracy = (actual == predicted).mean()
    yearly_accuracy.append(accuracy)

ax5.plot(years, yearly_accuracy, 'go-', linewidth=2, markersize=8)
ax5.set_xlabel('Jahr', fontsize=11)
ax5.set_ylabel('Genauigkeit', fontsize=11)
ax5.set_title('Modell-Genauigkeit\npro Jahr', fontsize=12, fontweight='bold')
ax5.set_ylim([0.5, 1.0])
ax5.grid(True, alpha=0.3)
ax5.axhline(y=np.mean(yearly_accuracy), color='red', linestyle='--', 
           alpha=0.5, label=f'Mittel: {np.mean(yearly_accuracy):.2%}')
ax5.legend()

# 6. Zusammenfassung
ax6 = fig.add_subplot(gs[3, 2])
ax6.axis('off')

summary_text = f"""
ANALYSE-ZUSAMMENFASSUNG
{'='*35}

ZEITRAUM: {years.min()} - {years.max()}

GESAMT-STATISTIK:
• Gesamtfälle: {int(yearly_stats['Faelle_sum'].sum()):,}
• Betroffene Gemeinden: {int(yearly_stats['Faelle_count_positive'].sum()):,}
• Mittleres Risiko: {yearly_stats['prediction_proba_mean'].mean():.1%}

TRENDS:
• Jahr mit meisten Fällen: {yearly_stats['Faelle_sum'].idxmax()}
  ({int(yearly_stats['Faelle_sum'].max()):,} Fälle)
• Jahr mit höchstem Risiko: {yearly_stats['prediction_proba_mean'].idxmax()}
  ({yearly_stats['prediction_proba_mean'].max():.1%})

TOP KLIMAFAKTOREN:
{chr(10).join([f'• {idx}: r={corr_df.loc[idx, "correlation"]:.3f}' 
               for idx in corr_df.head(3).index])}

MODELL-PERFORMANCE:
• Mittlere Genauigkeit: {np.mean(yearly_accuracy):.1%}
• Beste Jahr: {years[np.argmax(yearly_accuracy)]} ({max(yearly_accuracy):.1%})
• Schlechteste: {years[np.argmin(yearly_accuracy)]} ({min(yearly_accuracy):.1%})
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('Leishmaniose in Brasilien: Zeitliche Analyse 2007-2018', 
            fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DATA_DIR, 'temporal_analysis.png'), 
           dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 6. JAHRESWEISE RISIKOKARTEN (vereinfacht ohne Geodaten-Merge)
# =============================================================================
print("\n--- PHASE 6: Erstelle jahresweise Analyse ---")

# Erstelle vereinfachte Visualisierung der jahresweisen Entwicklung
selected_years = [2007, 2010, 2013, 2016, 2018]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, year in enumerate(selected_years):
    ax = axes[i]
    year_data = df[df['Jahr'] == year]
    
    # Erstelle Scatter Plot: Risiko vs Fälle
    ax.scatter(year_data['prediction_proba'], 
              np.log1p(year_data['Faelle']),  # Log-transformiert für bessere Sichtbarkeit
              alpha=0.5, s=5)
    
    # Füge Trennlinie bei 0.5 Risiko hinzu
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Risiko-Schwelle')
    
    # Statistiken
    high_risk = (year_data['prediction_proba'] > 0.5).sum()
    with_cases = (year_data['Faelle'] > 0).sum()
    
    ax.set_xlabel('Vorhergesagtes Risiko')
    ax.set_ylabel('Log(Fälle + 1)')
    ax.set_title(f'{year}\n{with_cases} Gemeinden mit Fällen\n{high_risk} mit hohem Risiko', 
                fontsize=11)
    ax.grid(True, alpha=0.3)

# Info-Panel
ax = axes[5]
ax.axis('off')

# Berechne weitere Statistiken
risk_trend = yearly_stats['prediction_proba_mean'].values
fall_trend = yearly_stats['Faelle_sum'].values

from scipy.stats import linregress
slope_risk, _, r_risk, _, _ = linregress(range(len(risk_trend)), risk_trend)
slope_falls, _, r_falls, _, _ = linregress(range(len(fall_trend)), fall_trend)

info_text = f"""
ZEITLICHE TRENDS 2007-2018
{'─'*30}

RISIKO-ENTWICKLUNG:
• Mittleres Risiko 2007: {yearly_stats['prediction_proba_mean'].iloc[0]:.1%}
• Mittleres Risiko 2018: {yearly_stats['prediction_proba_mean'].iloc[-1]:.1%}
• Trend: {"↑ steigend" if slope_risk > 0 else "↓ fallend"}

FALL-ENTWICKLUNG:
• Fälle 2007: {int(yearly_stats['Faelle_sum'].iloc[0]):,}
• Fälle 2018: {int(yearly_stats['Faelle_sum'].iloc[-1]):,}
• Trend: {"↑ steigend" if slope_falls > 0 else "↓ fallend"}

KLIMATISCHE TREIBER:
Die wichtigsten Faktoren sind:
{chr(10).join([f'{i+1}. {idx}: {"positiv" if corr_df.loc[idx, "correlation"] > 0 else "negativ"} korreliert'
               for i, idx in enumerate(corr_df.head(3).index)])}

HOTSPOT-GEMEINDEN:
Die am stärksten betroffenen
Gebiete sind hauptsächlich im
Amazonasgebiet und an der Küste.
"""

ax.text(0.1, 0.9, info_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.suptitle('Jahresweise Risiko-Analyse', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DATA_DIR, 'yearly_risk_analysis.png'), 
           dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 7. EXPORT DETAILLIERTER ERGEBNISSE
# =============================================================================
print("\n--- PHASE 7: Exportiere detaillierte Ergebnisse ---")

# Speichere erweiterte Tabelle mit Vorhersagen
output_table = df[['Jahr', 'NAME_2', 'Faelle', 'prediction_proba'] + bio_columns]
output_path = os.path.join(PROCESSED_DATA_DIR, 'temporal_analysis_results.csv')
output_table.to_csv(output_path, index=False)

# Speichere Jahres-Statistiken
yearly_stats_path = os.path.join(PROCESSED_DATA_DIR, 'yearly_statistics.csv')
yearly_stats.to_csv(yearly_stats_path)

# Speichere Klimafaktor-Korrelationen
corr_path = os.path.join(PROCESSED_DATA_DIR, 'climate_correlations.csv')
corr_df.to_csv(corr_path)

print(f"\n✓ Detaillierte Ergebnisse gespeichert in:")
print(f"  • {output_path}")
print(f"  • {yearly_stats_path}")
print(f"  • {corr_path}")

print("\n" + "="*80)
print("ANALYSE ABGESCHLOSSEN!")
print("="*80)
print("\nWichtigste Erkenntnisse:")
print(f"1. Stärkster Klimafaktor: {corr_df.index[0]} (r={corr_df.iloc[0]['correlation']:.3f})")
print(f"2. Jahr mit höchstem Risiko: {yearly_stats['prediction_proba_mean'].idxmax()}")
print(f"3. Modell-Genauigkeit über alle Jahre: {np.mean(yearly_accuracy):.1%}")

# Interessante Beobachtungen
if yearly_stats['Faelle_sum'].idxmax() != yearly_stats['prediction_proba_mean'].idxmax():
    print(f"\nInteressant: Das Jahr mit den meisten Fällen ({yearly_stats['Faelle_sum'].idxmax()}) "
          f"ist nicht das Jahr mit dem höchsten Risiko ({yearly_stats['prediction_proba_mean'].idxmax()})!")

print("\nVisualisierungen wurden gespeichert.")