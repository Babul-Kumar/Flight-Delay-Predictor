import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- DIRECTORY SETUP ---
OUTPUT_DIRS = [
    'output/plots',
    'output/models',
    'output/logs',
    'output/metrics',
    'output/reports'
]
for d in OUTPUT_DIRS:
    os.makedirs(d, exist_ok=True)

# --- LOGGING SETUP ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = logging.FileHandler('output/logs/pipeline.log', mode='w')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

class FlightFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.origin_traffic_ = {}
        self.dest_traffic_ = {}
        self.route_freq_ = {}
        self.major_cities = ['New York', 'Los Angeles', 'Chicago', 'Atlanta', 'Dallas-Fort Worth', 'Dallas', 'Houston', 'Denver']
        
    def fit(self, X, y=None):
        self.origin_traffic_ = X['ORIGIN_AIRPORT'].value_counts().to_dict()
        self.dest_traffic_ = X['DESTINATION_AIRPORT'].value_counts().to_dict()
        
        route_str = X['ORIGIN_AIRPORT'].astype(str) + '_' + X['DESTINATION_AIRPORT'].astype(str)
        self.route_freq_ = route_str.value_counts().to_dict()
        return self

    def transform(self, X, y=None):
        X_out = X.copy()
        
        try:
            dep_str = X_out['SCHEDULED_DEPARTURE'].astype(float).astype(int).astype(str).str.zfill(4)
            X_out['HOUR'] = dep_str.str[:2].astype(float)
        except Exception:
            X_out['HOUR'] = 0.0
        X_out['HOUR'] = X_out['HOUR'].fillna(0)
        
        # Cyclic encoding
        X_out['MONTH'] = X_out['MONTH'].fillna(1)
        X_out['DAY'] = X_out['DAY'].fillna(1)
        X_out['DAY_OF_WEEK'] = X_out['DAY_OF_WEEK'].fillna(1)

        X_out['MONTH_sin'] = np.sin(2 * np.pi * X_out['MONTH'] / 12.0)
        X_out['MONTH_cos'] = np.cos(2 * np.pi * X_out['MONTH'] / 12.0)
        X_out['DAY_sin'] = np.sin(2 * np.pi * X_out['DAY'] / 31.0)
        X_out['DAY_cos'] = np.cos(2 * np.pi * X_out['DAY'] / 31.0)
        X_out['DAY_OF_WEEK_sin'] = np.sin(2 * np.pi * X_out['DAY_OF_WEEK'] / 7.0)
        X_out['DAY_OF_WEEK_cos'] = np.cos(2 * np.pi * X_out['DAY_OF_WEEK'] / 7.0)
        X_out['HOUR_sin'] = np.sin(2 * np.pi * X_out['HOUR'] / 24.0)
        X_out['HOUR_cos'] = np.cos(2 * np.pi * X_out['HOUR'] / 24.0)
        
        # Flags
        X_out['is_weekend'] = (X_out['DAY_OF_WEEK'] > 5).astype(int)
        X_out['is_peak_hour'] = ((X_out['HOUR'] >= 7) & (X_out['HOUR'] <= 9)) | ((X_out['HOUR'] >= 16) & (X_out['HOUR'] <= 19))
        X_out['is_peak_hour'] = X_out['is_peak_hour'].astype(int)
        X_out['is_night_flight'] = ((X_out['HOUR'] >= 22) | (X_out['HOUR'] <= 5)).astype(int)
        
        # Speed
        time_valid = X_out['SCHEDULED_TIME'].replace(0, np.nan)
        X_out['SPEED'] = X_out['DISTANCE'] / time_valid
        
        # Advanced Features
        X_out['ROUTE'] = X_out['ORIGIN_AIRPORT'].astype(str) + '_' + X_out['DESTINATION_AIRPORT'].astype(str)
        
        X_out['GEO_DISTANCE'] = haversine(
            X_out['ORIGIN_LATITUDE'].fillna(0), 
            X_out['ORIGIN_LONGITUDE'].fillna(0), 
            X_out['DEST_LATITUDE'].fillna(0), 
            X_out['DEST_LONGITUDE'].fillna(0)
        )
        
        X_out['SAME_STATE'] = (X_out['ORIGIN_STATE'] == X_out['DEST_STATE']).astype(int)
        
        orig_major = X_out['ORIGIN_CITY'].isin(self.major_cities)
        dest_major = X_out['DEST_CITY'].isin(self.major_cities)
        X_out['MAJOR_CITY'] = (orig_major | dest_major).astype(int)
        
        orig_t = X_out['ORIGIN_AIRPORT'].map(self.origin_traffic_).fillna(1)
        dest_t = X_out['DESTINATION_AIRPORT'].map(self.dest_traffic_).fillna(1)
        X_out['AIRPORT_TRAFFIC'] = orig_t + dest_t
        X_out['AIRPORT_IMPORTANCE'] = (orig_t * dest_t) / 1000.0  
        
        X_out['ROUTE_FREQUENCY'] = X_out['ROUTE'].map(self.route_freq_).fillna(1)
        X_out['DISTANCE_CATEGORY'] = pd.cut(X_out['DISTANCE'], bins=[0, 500, 1500, 10000], labels=[1, 2, 3]).astype(float).fillna(2)
        X_out['LONG_HAUL'] = (X_out['GEO_DISTANCE'] > 2000).astype(int)
        
        # --- NEW INTERACTION FEATURES ---
        X_out['TIME_DISTANCE'] = X_out['HOUR'] * X_out['DISTANCE']
        X_out['TRAFFIC_PRESSURE'] = X_out['AIRPORT_TRAFFIC'] * X_out['is_peak_hour']
        X_out['ROUTE_COMPLEXITY'] = X_out['ROUTE_FREQUENCY'] / (X_out['DISTANCE'] + 1)
        
        cols_to_drop = [
            'MONTH', 'DAY', 'DAY_OF_WEEK', 'HOUR', 'SCHEDULED_DEPARTURE', 
            'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_ARRIVAL',
            'ORIGIN_LATITUDE', 'ORIGIN_LONGITUDE', 'DEST_LATITUDE', 'DEST_LONGITUDE',
            'ORIGIN_CITY', 'ORIGIN_STATE', 'DEST_CITY', 'DEST_STATE', 'AIRLINE_NAME', 'AIRLINE'
        ]
        X_out = X_out.drop(columns=[c for c in cols_to_drop if c in X_out.columns])
        
        return X_out

class OOFTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, cv=5, alpha=10):
        self.cols = cols
        self.cv = cv
        self.alpha = alpha
        self.global_means_ = {}
        self.category_means_ = {}

    def fit(self, X, y):
        y_series = pd.Series(y, index=X.index)
        self.global_means_ = {}
        self.category_means_ = {}
        
        for col in self.cols:
            self.global_means_[col] = y_series.mean()
            stats = y_series.groupby(X[col]).agg(['mean', 'count'])
            smooth_mean = (stats['count'] * stats['mean'] + self.alpha * self.global_means_[col]) / (stats['count'] + self.alpha)
            self.category_means_[col] = smooth_mean.to_dict()
            
        return self

    def transform(self, X, y=None):
        X_out = X.copy()
        
        if y is not None:
            y_series = pd.Series(y, index=X.index)
            kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
            
            for col in self.cols:
                oof_encoded = pd.Series(np.nan, index=X.index)
                for train_idx, val_idx in kf.split(X, y):
                    train_x, val_x = X.iloc[train_idx], X.iloc[val_idx]
                    train_y = y_series.iloc[train_idx]
                    
                    stats = train_y.groupby(train_x[col]).agg(['mean', 'count'])
                    smooth_mean = (stats['count'] * stats['mean'] + self.alpha * self.global_means_[col]) / (stats['count'] + self.alpha)
                    
                    mapped = val_x[col].map(smooth_mean).fillna(self.global_means_[col])
                    oof_encoded.iloc[val_idx] = mapped
                    
                X_out[col + '_TE'] = oof_encoded
                X_out.drop(columns=[col], inplace=True)
        else:
            for col in self.cols:
                mapped = X[col].map(self.category_means_[col]).fillna(self.global_means_[col])
                X_out[col + '_TE'] = mapped
                X_out.drop(columns=[col], inplace=True)
                
        return X_out

class FullPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.engineer = FlightFeatureEngineer()
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='constant', fill_value='Unknown')
        self.te = OOFTargetEncoder(cols=['AIRLINE', 'ROUTE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'], alpha=10)
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        X_temp = X.copy()
        X_temp['ROUTE'] = X_temp['ORIGIN_AIRPORT'].astype(str) + '_' + X_temp['DESTINATION_AIRPORT'].astype(str)
        
        cat_cols = ['AIRLINE', 'ROUTE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
        self.imputer_cat.fit(X_temp[cat_cols])
        
        cat_imputed = pd.DataFrame(self.imputer_cat.transform(X_temp[cat_cols]), columns=cat_cols, index=X.index)
        self.te.fit(cat_imputed, y)
        
        X_eng = self.engineer.fit_transform(X)
        num_cols = [c for c in X_eng.columns if c not in cat_cols and c != 'ROUTE']
        self.imputer_num.fit(X_eng[num_cols])
        
        num_imputed = pd.DataFrame(self.imputer_num.transform(X_eng[num_cols]), columns=num_cols, index=X.index)
        te_transformed = self.te.transform(cat_imputed, y)
        
        X_combined = pd.concat([num_imputed, te_transformed], axis=1)
        self.scaler.fit(X_combined)
        self.final_cols = X_combined.columns
        return self
        
    def transform(self, X, y=None):
        X_temp = X.copy()
        X_temp['ROUTE'] = X_temp['ORIGIN_AIRPORT'].astype(str) + '_' + X_temp['DESTINATION_AIRPORT'].astype(str)
        
        cat_cols = ['AIRLINE', 'ROUTE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
        cat_imputed = pd.DataFrame(self.imputer_cat.transform(X_temp[cat_cols]), columns=cat_cols, index=X.index)
        
        X_eng = self.engineer.transform(X)
        num_cols = [c for c in X_eng.columns if c not in cat_cols and c != 'ROUTE']
        
        num_imputed = pd.DataFrame(self.imputer_num.transform(X_eng[num_cols]), columns=num_cols, index=X.index)
        te_transformed = self.te.transform(cat_imputed, y)
        
        X_combined = pd.concat([num_imputed, te_transformed], axis=1)
        X_scaled = pd.DataFrame(self.scaler.transform(X_combined), columns=X_combined.columns, index=X.index)
        return X_scaled

class WeightedAdaBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None, n_estimators=400, learning_rate=0.03, random_state=42):
        if estimator is None:
            estimator = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20)
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = AdaBoostClassifier(
            estimator=self.estimator, 
            n_estimators=self.n_estimators, 
            learning_rate=self.learning_rate, 
            random_state=self.random_state
        )

    def fit(self, X, y):
        # Dynamically apply sample weights: 2.0 for delayed (1), 1.0 for on-time (0)
        sw = np.where(y == 1, 2.0, 1.0)
        self.model.fit(X, y, sample_weight=sw)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    @property
    def feature_importances_(self):
        return self.model.feature_importances_

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    logger.info("STEP 1 & 2: DATA LOADING AND SAFE MERGING")
    try:
        flights = pd.read_csv('flights.csv', usecols=[
            'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 
            'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL', 
            'SCHEDULED_TIME', 'DISTANCE', 'ARRIVAL_DELAY'
        ])
        airlines = pd.read_csv('airlines.csv')
        airports = pd.read_csv('airports.csv')
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return

    flights = flights.dropna(subset=['ARRIVAL_DELAY'])
    flights['DELAYED'] = (flights['ARRIVAL_DELAY'] >= 15).astype(int)

    # Merge Airlines
    flights = flights.merge(airlines, left_on='AIRLINE', right_on='IATA_CODE', how='left')
    flights.rename(columns={'AIRLINE_y': 'AIRLINE_NAME', 'AIRLINE_x': 'AIRLINE'}, inplace=True)
    if 'IATA_CODE' in flights.columns:
        flights.drop(columns=['IATA_CODE'], inplace=True)

    # Merge Airports (Origin)
    flights = flights.merge(airports[['IATA_CODE', 'CITY', 'STATE', 'LATITUDE', 'LONGITUDE']], 
                            left_on='ORIGIN_AIRPORT', right_on='IATA_CODE', how='left')
    flights.rename(columns={'CITY': 'ORIGIN_CITY', 'STATE': 'ORIGIN_STATE', 'LATITUDE': 'ORIGIN_LATITUDE', 'LONGITUDE': 'ORIGIN_LONGITUDE'}, inplace=True)
    if 'IATA_CODE' in flights.columns:
        flights.drop(columns=['IATA_CODE'], inplace=True)

    # Merge Airports (Destination)
    flights = flights.merge(airports[['IATA_CODE', 'CITY', 'STATE', 'LATITUDE', 'LONGITUDE']], 
                            left_on='DESTINATION_AIRPORT', right_on='IATA_CODE', how='left')
    flights.rename(columns={'CITY': 'DEST_CITY', 'STATE': 'DEST_STATE', 'LATITUDE': 'DEST_LATITUDE', 'LONGITUDE': 'DEST_LONGITUDE'}, inplace=True)
    if 'IATA_CODE' in flights.columns:
        flights.drop(columns=['IATA_CODE'], inplace=True)

    X_full = flights.drop(columns=['ARRIVAL_DELAY', 'DELAYED'])
    y_full = flights['DELAYED']

    SAMPLE_SIZE = 50000 
    if len(flights) > SAMPLE_SIZE:
        logger.info(f"Subsampling dataset to {SAMPLE_SIZE} records for feasibility.")
        _, X_full, _, y_full = train_test_split(X_full, y_full, test_size=SAMPLE_SIZE, stratify=y_full, random_state=42)

    logger.info("STEP 5: TRAIN-TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, stratify=y_full, random_state=42
    )
    X_test_original = X_test.copy() # Saving to use in error analysis later
    
    # 1. CLASS DISTRIBUTION
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y_train)
    plt.title("Class Distribution")
    plt.savefig('output/plots/class_distribution.png')
    plt.close()
    
    logger.info("STEP 3, 4, 6: PREPROCESSING PIPELINE (IMPUTE, ENG, OOF TE)")
    preprocessor = FullPreprocessor()
    X_train_preprocessed = preprocessor.fit_transform(X_train, y_train)
    
    # 5. FEATURE CORRELATION HEATMAP
    plt.figure(figsize=(12, 10))
    sns.heatmap(X_train_preprocessed.corr(), cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig('output/plots/correlation_heatmap.png')
    plt.close()
    
    logger.info("STEP 7: FORWARD FEATURE SELECTION")
    lr_evaluator = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    sfs = SequentialFeatureSelector(lr_evaluator, n_features_to_select=22, direction='forward', cv=5, n_jobs=-1)
    
    sfs.fit(X_train_preprocessed, y_train)
    selected_features = X_train_preprocessed.columns[sfs.get_support()].tolist()
    logger.info(f"Selected Features ({len(selected_features)}): {selected_features}")
    
    X_train_sel = X_train_preprocessed[selected_features]
    
    logger.info("STEP 8 & 9: MODELS AND MODEL TRAINING")
    models = {
        'Logistic Regression': Pipeline([
            ('model', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
        ]),
        'Polynomial Regression': Pipeline([
            ('poly', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
            ('model', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
        ]),
        'AdaBoost': Pipeline([
            ('model', WeightedAdaBoost(
                estimator=DecisionTreeClassifier(max_depth=4, min_samples_leaf=20), 
                n_estimators=400, 
                learning_rate=0.03,
                random_state=42
            ))
        ])
    }
    
    model_results = {}
    best_f1 = 0
    best_model_name = ""
    
    # STEP 3 & 7: Use StratifiedKFold and reduce splits to 3 for stability under severe imbalance
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for name, pipeline in models.items():
        logger.info(f"--- Cross-Validating {name} ---")
        fold_f1s = []
        fold_aucs = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_sel, y_train)):
            X_fold_train, X_fold_val = X_train_sel.iloc[train_idx], X_train_sel.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # STEP 1: Debug Logging
            unique_classes, counts = np.unique(y_fold_val, return_counts=True)
            logger.info(f"[{name}] Fold {fold+1} - Class Distribution: {dict(zip(unique_classes, counts))}")
            
            # STEP 2: Verify Root Cause
            if len(unique_classes) < 2:
                logger.warning(f"[{name}] Fold {fold+1} contains only one class -> AUC undefined!")
            
            # Fit model on training fold
            pipeline.fit(X_fold_train, y_fold_train)
            
            try:
                y_pred = pipeline.predict(X_fold_val)
                y_proba = pipeline.predict_proba(X_fold_val)[:, 1]
                
                # Check predictions
                logger.info(f"[{name}] Fold {fold+1} - Unique Preds: {np.unique(y_pred)}, Proba Range: [{np.min(y_proba):.4f}, {np.max(y_proba):.4f}]")
                
                # STEP 6: Safe AUC Computation
                if len(unique_classes) < 2:
                    auc = np.nan
                else:
                    auc = roc_auc_score(y_fold_val, y_proba)
                    
                f1 = f1_score(y_fold_val, y_pred)
                
                fold_f1s.append(f1)
                fold_aucs.append(auc)
                logger.info(f"[{name}] Fold {fold+1} - F1: {f1:.4f}, AUC: {auc:.4f}")
                
            except Exception as e:
                # Force error visibility for debugging
                logger.error(f"[{name}] Fold {fold+1} prediction failed: {str(e)}")
                raise e # STEP 5: Force error visibility
        
        # STEP 4: Handle NaN Safely
        mean_f1 = np.nanmean(fold_f1s)
        mean_auc = np.nanmean(fold_aucs)
        
        # STEP 8: Warning if NaN
        if np.isnan(mean_auc):
            logger.warning(f"Warning: Mean AUC for {name} resulted in NaN!")
            
        model_results[name] = {'F1 (CV)': mean_f1, 'ROC-AUC (CV)': mean_auc}
        logger.info(f"{name} -> Mean CV F1: {mean_f1:.4f}, Mean CV AUC: {mean_auc:.4f}")
        
        # Fit model on full training data
        pipeline.fit(X_train_sel, y_train)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_model_name = name

    logger.info("MODEL COMPARISON & SAVING METRICS")
    df_results = pd.DataFrame(model_results).T
    df_results.to_csv('output/metrics/model_comparison.csv')
    
    plt.figure(figsize=(10, 6))
    df_results.plot(kind='bar', figsize=(10,6), colormap='viridis')
    plt.title('Model Comparison (CV)')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('output/plots/model_comparison.png')
    plt.close()
    
    # We calibrate the best model over a 3-fold CV of the selected features training set.
    logger.info(f"STEP 6 & 10: CALIBRATION & THRESHOLD OPTIMIZATION on {best_model_name}")
    best_model_base = models[best_model_name]
    calibrated_model = CalibratedClassifierCV(best_model_base, method='sigmoid', cv=3)
    calibrated_model.fit(X_train_sel, y_train)
    
    # Get CV probabilities for threshold tuning (using uncalibrated base model predictions over CV folds)
    y_probs_cv = cross_val_predict(best_model_base, X_train_sel, y_train, cv=3, method='predict_proba', n_jobs=-1)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_probs_cv)
    
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    optim_scores = (0.7 * f1_scores) + (0.5 * recalls[:-1])
    
    # 3. THRESHOLD VS F1 CURVE
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_scores)
    plt.title("Threshold vs F1 Score")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.savefig('output/plots/threshold_vs_f1.png')
    plt.close()
    
    # Restrict to threshold bounds [0.1, 0.9]
    valid_idx = np.where((thresholds >= 0.1) & (thresholds <= 0.9))[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmax(optim_scores[valid_idx])]
        best_threshold = thresholds[best_idx]
        logger.info(f"Optimal Threshold: {best_threshold:.4f} (Objective Score: {optim_scores[best_idx]:.4f})")
    else:
        best_threshold = 0.5
        logger.warning("No valid threshold found between 0.1 and 0.9. Defaulting to 0.5")
    
    logger.info("STEP 11: FINAL EVALUATION (TEST SET)")
    X_test_preprocessed = preprocessor.transform(X_test, y=None)
    X_test_sel = X_test_preprocessed[selected_features]
    
    test_probs = calibrated_model.predict_proba(X_test_sel)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)
    
    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds)
    rec = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    auc = roc_auc_score(y_test, test_probs)
    cm = confusion_matrix(y_test, test_preds)
    
    logger.info(f"Test Accuracy: {acc:.4f}, F1: {f1:.4f}, Recall: {rec:.4f}, AUC: {auc:.4f}")
    
    logger.info("STEP 12: VISUALIZATION")
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Calibrated)')
    plt.legend()
    plt.savefig('output/plots/roc_curve.png')
    plt.close()
    
    pr, rc, _ = precision_recall_curve(y_test, test_probs)
    plt.figure(figsize=(8,6))
    plt.plot(rc, pr, color='green')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('output/plots/pr_curve.png')
    plt.close()
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig('output/plots/confusion_matrix.png')
    plt.close()
    
    if best_model_name == 'Logistic Regression':
        try:
            coef = best_model_base.named_steps['model'].coef_[0]
            idx = np.argsort(np.abs(coef))[::-1][:20]
            plt.figure(figsize=(10,8))
            sns.barplot(x=coef[idx], y=np.array(selected_features)[idx], palette='viridis')
            plt.title('Top 20 Feature Importances (Logistic Regression)')
            plt.tight_layout()
            plt.savefig('output/plots/feature_importance.png')
            plt.close()
        except AttributeError:
            logger.warning("Could not extract feature importances from Logistic Regression.")
    elif best_model_name == 'AdaBoost':
        try:
            importances = best_model_base.named_steps['model'].feature_importances_
            idx = np.argsort(importances)[::-1][:20]
            plt.figure(figsize=(10,8))
            sns.barplot(x=importances[idx], y=np.array(selected_features)[idx], palette='viridis')
            plt.title('Top 20 Feature Importances (AdaBoost)')
            plt.tight_layout()
            plt.savefig('output/plots/feature_importance.png')
            plt.close()
        except AttributeError:
            logger.warning("Could not extract feature importances from wrapper.")
        
    logger.info("STEP 8 & 13: DELAY CLUSTER PROFILING ON SELECTED FEATURES")
    kmeans = KMeans(n_clusters=6, random_state=42)
    train_clusters = kmeans.fit_predict(X_train_sel)
    
    X_train_clustered = X_train_sel.copy()
    X_train_clustered['CLUSTER'] = train_clusters
    X_train_clustered['DELAYED'] = y_train.values
    
    cluster_analysis = X_train_clustered.groupby('CLUSTER').mean()
    cluster_counts = X_train_clustered['CLUSTER'].value_counts(normalize=True) * 100
    cluster_analysis['CLUSTER_SIZE_%'] = cluster_counts
    
    cluster_analysis.to_csv('output/reports/cluster_summary.csv')
    
    plt.figure(figsize=(8,5))
    sns.countplot(x=train_clusters, palette='Set2')
    plt.title('Cluster Distribution')
    plt.savefig('output/plots/cluster_distribution.png')
    plt.close()
    
    plt.figure(figsize=(8,5))
    cluster_analysis['DELAYED'].plot(kind='bar', color='salmon')
    plt.title('Delay Rate per Cluster')
    plt.savefig('output/plots/cluster_delay_rate.png')
    plt.close()
    
    logger.info("STEP 9: ERROR ANALYSIS")
    errors_df = X_test_original.copy()
    errors_df['True_Label'] = y_test.values
    errors_df['Predicted_Label'] = test_preds
    errors_df['Probability'] = test_probs
    
    fp_mask = (errors_df['True_Label'] == 0) & (errors_df['Predicted_Label'] == 1)
    fn_mask = (errors_df['True_Label'] == 1) & (errors_df['Predicted_Label'] == 0)
    
    fp_samples = errors_df[fp_mask].sample(n=min(50, fp_mask.sum()), random_state=42)
    fn_samples = errors_df[fn_mask].sample(n=min(50, fn_mask.sum()), random_state=42)
    
    error_analysis_df = pd.concat([fp_samples, fn_samples])
    error_analysis_df.to_csv('output/reports/error_analysis.csv', index=False)
    logger.info(f"Saved {len(fp_samples)} FP and {len(fn_samples)} FN samples to error_analysis.csv")
    
    # 4. ERROR ANALYSIS VISUALIZATION
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(fp_samples['DISTANCE'], kde=True, color='red')
    plt.title('False Positives (DISTANCE)')
    plt.subplot(1, 2, 2)
    sns.histplot(fn_samples['DISTANCE'], kde=True, color='blue')
    plt.title('False Negatives (DISTANCE)')
    plt.tight_layout()
    plt.savefig('output/plots/error_analysis_distance.png')
    plt.close()
    
    logger.info("STEP 14: SAVE PIPELINE")
    final_pipeline_obj = {
        'preprocessor': preprocessor,
        'feature_selector': sfs,
        'selected_features': selected_features,
        'base_model': best_model_base,
        'calibrated_model': calibrated_model,
        'best_threshold': best_threshold,
        'cluster_model': kmeans
    }
    joblib.dump(final_pipeline_obj, 'output/models/model.joblib')
    logger.info("Pipeline successfully saved to output/models/model.joblib")

if __name__ == '__main__':
    main()
