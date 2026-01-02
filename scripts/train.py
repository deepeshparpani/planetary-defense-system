import os
import requests
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import Pipeline    

# Load environment variables (API Key)
load_dotenv()
API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")

def fetch_data(pages=500):
    all_neos = []
    # Note: Use the /browse endpoint for a large, balanced historical dataset
    url = f"https://api.nasa.gov/neo/rest/v1/neo/browse?api_key={API_KEY}"
    
    print(f"📡 Downloading professional-scale asteroid data...")
    for p in range(pages):
        try:
            response = requests.get(url).json()
            if 'near_earth_objects' not in response: break
                
            for obj in response['near_earth_objects']:
                if not obj.get('close_approach_data'): continue
                
                # We explicitly name the keys to match what engineer_features expects
                all_neos.append({
                    "est_diameter_min": obj['estimated_diameter']['kilometers']['estimated_diameter_min'],
                    "relative_velocity": float(obj['close_approach_data'][0]['relative_velocity']['kilometers_per_hour']),
                    "miss_distance": float(obj['close_approach_data'][0]['miss_distance']['kilometers']),
                    "absolute_magnitude": obj['absolute_magnitude_h'],
                    "is_hazardous": int(obj['is_potentially_hazardous_asteroid'])
                })
            
            if (p + 1) % 50 == 0:
                print(f"✅ Syncing page {p+1}/{pages}...")
                time.sleep(1) 
            
            url = response['links']['next'].replace("http://", "https://")
        except Exception as e:
            print(f"⚠️ Stopped at page {p+1} due to error: {e}")
            break
            
    return pd.DataFrame(all_neos)

def engineer_features(df):
    """Adds interaction features with a safety check for column names."""
    # Safety Check: Print columns to debug if this fails again
    required_cols = ['est_diameter_min', 'relative_velocity', 'miss_distance']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Error: Missing columns. Found: {df.columns.tolist()}")
        return df

    # Ratio of size to distance (Smaller/Closer = Higher Risk)
    df['size_dist_ratio'] = df['est_diameter_min'] / (df['miss_distance'] + 1e-5)
    
    # Kinetic energy proxy (Velocity squared * diameter)
    df['kinetic_proxy'] = (df['relative_velocity']**2) * df['est_diameter_min']
    
    # Velocity to Distance ratio
    df['velocity_dist_ratio'] = df['relative_velocity'] / (df['miss_distance'] + 1e-5)
    
    return df

def train_and_evaluate():
    raw_df = fetch_data()
    
    if raw_df.empty:
        print("❌ No data fetched. Check your API key and connection.")
        return

    df = engineer_features(raw_df) 
    
    X = df.drop('is_hazardous', axis=1)
    y = df['is_hazardous']
    
    neg, pos = (y == 0).sum(), (y == 1).sum()
    imbalance_ratio = neg / pos
    print(f"\n📊 Dataset Stats: Safe={neg}, Hazardous={pos} (Ratio: {imbalance_ratio:.2f})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Setup a Pipeline with SMOTE and XGBoost
    model_pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(
            scale_pos_weight=imbalance_ratio * 1.5, 
            tree_method='hist',
            n_jobs=-1,
            eval_metric='logloss'
        ))
    ])

    param_grid = {
        'xgb__n_estimators': [500], # Keep it simple for the first run after fix
        'xgb__max_depth': [4, 6],
        'xgb__learning_rate': [0.01, 0.05],
        'xgb__subsample': [0.9]
    }

    print("🧠 Starting High-Recall Grid Search (SMOTE + XGBoost)...")
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring='recall', 
        cv=3,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"\n🏆 Best Settings Found: {grid_search.best_params_}")

    # EVALUATION
    y_pred = best_model.predict(X_test)
    print("\n--- PERFORMANCE REPORT ---")
    print(classification_report(y_test, y_pred))
    
    # FEATURE IMPORTANCE
    final_xgb = best_model.named_steps['xgb']
    plot_importance(final_xgb)
    plt.title("Advanced Feature Importance (SMOTE + Engineering)")
    plt.show()
    
    # SAVE MODEL
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/neo_classifier.joblib')
    print(f"✅ Advanced model saved in 'models/'")

if __name__ == "__main__":
    train_and_evaluate()