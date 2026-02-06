import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

# Konfiguration
BASE_DIR = r"C:\Users\Thomas\Forschungsprojekt- Leishmaniose"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

INPUT_FILE = os.path.join(DATA_DIR, "master_table_with_landscape.csv")
SHAPEFILE = os.path.join(RAW_DIR, "map_brazil.gpkg")

def load_and_engineer_features():
    print("Loading and preparing data...")
    
    df = pd.read_csv(INPUT_FILE)
    print(f"Rows loaded: {len(df)}")

    # Bevoelkerungsdichte berechnen
    try:
        print("Calculating population density...")
        gdf = gpd.read_file(SHAPEFILE, layer='ADM_ADM_2')
        # Area in km2 (projizierte Schaetzung)
        gdf['area_km2'] = gdf.geometry.to_crs({'proj':'cea'}).area / 10**6
        
        # Match keys
        gdf['code_short'] = gdf['CC_2'].astype(str).str.replace(r'\.0', '', regex=True).str.slice(0, 6)
        df['code_short'] = df['code_gemeinde'].astype(str).str.replace(r'\.0', '', regex=True).str.slice(0, 6)
        
        df = pd.merge(df, gdf[['code_short', 'area_km2']], on='code_short', how='left')
        df['pop_density'] = df['population'] / df['area_km2']
        df['pop_density'] = df['pop_density'].replace([np.inf, -np.inf], 0).fillna(0)
        
        features_extra = ['elevation', 'pop_density']
    except Exception as e:
        print(f"Warning: Could not calc area ({e}). Using elevation only.")
        features_extra = ['elevation']
        df['pop_density'] = 0

    climate_cols = [c for c in df.columns if c.startswith('bio_')]
    feature_cols = climate_cols + features_extra
    
    df_model = df.dropna(subset=feature_cols + ['Faelle', 'incidence'])
    
    # Targets
    df_model['target_class'] = (df_model['Faelle'] > 0).astype(int)
    df_model['target_inc_log'] = np.log1p(df_model['incidence']) 
    
    return df_model, feature_cols

def train_ensemble_hurdle(df, features):
    print("Training ensemble models...")
    
    X = df[features]
    y_class = df['target_class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.3, random_state=42, stratify=y_class)
    
    # 1. Classification
    print("Step 1: Classification (RF + XGB)...")
    clf1 = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf2 = XGBClassifier(n_estimators=200, eval_metric='logloss', use_label_encoder=False, random_state=42)
    
    eclf = VotingClassifier(estimators=[('rf', clf1), ('xgb', clf2)], voting='soft')
    eclf.fit(X_train, y_train)
    
    y_pred_proba = eclf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Classification AUC: {auc:.4f}")
    
    # 2. Regression
    print("Step 2: Regression (Positive cases only)...")
    mask_pos = df['target_class'] == 1
    X_reg = df[mask_pos][features]
    y_reg = df[mask_pos]['target_inc_log']
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
    
    reg1 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg2 = XGBRegressor(n_estimators=200, random_state=42)
    
    ereg = VotingRegressor(estimators=[('rf', reg1), ('xgb', reg2)])
    ereg.fit(X_train_r, y_train_r)
    
    y_pred_log = ereg.predict(X_test_r)
    r2 = r2_score(y_test_r, y_pred_log)
    mae = mean_absolute_error(np.expm1(y_test_r), np.expm1(y_pred_log))
    
    print(f"Regression R2: {r2:.4f}")
    print(f"MAE: {mae:.2f}")

    return eclf, ereg

def analyze_importance(model, features, name):
    try:
        # Average importance approximation for VotingRegressor
        rf_imp = model.estimators_[0].feature_importances_
        xgb_imp = model.estimators_[1].feature_importances_
        avg_imp = (rf_imp + xgb_imp) / 2
        
        indices = np.argsort(avg_imp)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importance ({name})")
        plt.bar(range(len(features)), avg_imp[indices], align="center", color='darkblue')
        plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "results", f"importance_{name}.png"))
        print(f"Saved plot: importance_{name}.png")
    except:
        pass

if __name__ == "__main__":
    df, cols = load_and_engineer_features()
    if df is not None:
        clf_model, reg_model = train_ensemble_hurdle(df, cols)
        analyze_importance(clf_model, cols, "Classification")
        analyze_importance(reg_model, cols, "Regression")