"""
Machine Learning algorithms for currency detection
Includes traditional ML models with engineered features
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVM
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from config import MODELS_DIR, ML_CONFIG

logger = logging.getLogger(__name__)

class MLCurrencyDetector:
    """Machine Learning based currency authentication"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.is_trained = False
        self.feature_names = []
        
    def prepare_data(self, features_dict: Dict[str, List[float]], labels: List[int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature data for ML training"""
        # Convert features dictionary to DataFrame
        df = pd.DataFrame(features_dict)
        
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Get feature names
        feature_names = df.columns.tolist()
        
        # Convert to numpy arrays
        X = df.values
        y = np.array(labels)
        
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, feature_names
    
    def preprocess_features(self, X: np.ndarray, y: Optional[np.ndarray] = None, fit: bool = True) -> np.ndarray:
        """Preprocess features with scaling and selection"""
        # Scale features
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Feature selection
        if fit and y is not None:
            # Select top K features
            k = min(ML_CONFIG['feature_selection_k'], X.shape[1])
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
            
            # Get selected feature names
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_feature_names = [self.feature_names[i] for i in selected_indices]
            
            logger.info(f"Selected {k} best features out of {X.shape[1]}")
        else:
            X_selected = self.feature_selector.transform(X_scaled)
        
        return X_selected
    
    def create_models(self) -> Dict[str, Any]:
        """Create various ML models for ensemble"""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=ML_CONFIG['random_state'],
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=ML_CONFIG['random_state']
            ),
            'svm': SVM(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=ML_CONFIG['random_state']
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=ML_CONFIG['random_state']
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            'naive_bayes': GaussianNB()
        }
        
        return models
    
    def hyperparameter_tuning(self, model, X: np.ndarray, y: np.ndarray, param_grid: Dict) -> Any:
        """Perform hyperparameter tuning using GridSearchCV"""
        cv = StratifiedKFold(n_splits=ML_CONFIG['cross_validation_folds'], 
                           shuffle=True, 
                           random_state=ML_CONFIG['random_state'])
        
        grid_search = GridSearchCV(
            model, param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_individual_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train individual ML models"""
        models = self.create_models()
        trained_models = {}
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=ML_CONFIG['cross_validation_folds'], 
                           shuffle=True, 
                           random_state=ML_CONFIG['random_state'])
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
            logger.info(f"{name} CV F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Train on full dataset
            model.fit(X, y)
            trained_models[name] = model
        
        return trained_models
    
    def create_ensemble_model(self, individual_models: Dict[str, Any]) -> VotingClassifier:
        """Create ensemble model from individual models"""
        # Select best performing models for ensemble
        estimators = [(name, model) for name, model in individual_models.items()]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability voting
        )
        
        return ensemble
    
    def train(self, features_dict: Dict[str, List[float]], labels: List[int], currency: str = 'USD'):
        """Train all ML models"""
        logger.info(f"Training ML models for {currency} currency detection...")
        
        # Prepare data
        X, y, self.feature_names = self.prepare_data(features_dict, labels)
        
        # Preprocess features
        X_processed = self.preprocess_features(X, y, fit=True)
        
        # Train individual models
        individual_models = self.train_individual_models(X_processed, y)
        
        # Create ensemble model
        ensemble_model = self.create_ensemble_model(individual_models)
        ensemble_model.fit(X_processed, y)
        
        # Store models
        self.models[currency] = {
            'individual': individual_models,
            'ensemble': ensemble_model
        }
        
        self.is_trained = True
        logger.info(f"ML models trained successfully for {currency}")
    
    def predict(self, features_dict: Dict[str, List[float]], currency: str = 'USD') -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using trained models"""
        if not self.is_trained or currency not in self.models:
            raise ValueError(f"Models not trained for {currency}")
        
        # Prepare data
        X, _, _ = self.prepare_data(features_dict, [0] * len(list(features_dict.values())[0]))
        
        # Preprocess features
        X_processed = self.preprocess_features(X, fit=False)
        
        # Get ensemble predictions
        ensemble_model = self.models[currency]['ensemble']
        predictions = ensemble_model.predict(X_processed)
        probabilities = ensemble_model.predict_proba(X_processed)
        
        return predictions, probabilities
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray, currency: str = 'USD') -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models"""
        if not self.is_trained or currency not in self.models:
            raise ValueError(f"Models not trained for {currency}")
        
        results = {}
        
        # Evaluate individual models
        for name, model in self.models[currency]['individual'].items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
        
        # Evaluate ensemble model
        ensemble_model = self.models[currency]['ensemble']
        y_pred_ensemble = ensemble_model.predict(X_test)
        
        results['ensemble'] = {
            'accuracy': accuracy_score(y_test, y_pred_ensemble),
            'precision': precision_score(y_test, y_pred_ensemble),
            'recall': recall_score(y_test, y_pred_ensemble),
            'f1': f1_score(y_test, y_pred_ensemble)
        }
        
        return results
    
    def get_feature_importance(self, currency: str = 'USD') -> Dict[str, float]:
        """Get feature importance from tree-based models"""
        if not self.is_trained or currency not in self.models:
            raise ValueError(f"Models not trained for {currency}")
        
        importance_dict = {}
        
        # Get importance from Random Forest
        rf_model = self.models[currency]['individual']['random_forest']
        rf_importance = rf_model.feature_importances_
        
        # Get importance from Gradient Boosting
        gb_model = self.models[currency]['individual']['gradient_boosting']
        gb_importance = gb_model.feature_importances_
        
        # Average importance
        avg_importance = (rf_importance + gb_importance) / 2
        
        # Map to feature names
        for i, importance in enumerate(avg_importance):
            if i < len(self.selected_feature_names):
                importance_dict[self.selected_feature_names[i]] = importance
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def save_models(self, currency: str = 'USD'):
        """Save trained models to disk"""
        if not self.is_trained or currency not in self.models:
            raise ValueError(f"Models not trained for {currency}")
        
        # Save individual models
        for name, model in self.models[currency]['individual'].items():
            model_path = f"{MODELS_DIR}/ml_{currency}_{name}.joblib"
            joblib.dump(model, model_path)
        
        # Save ensemble model
        ensemble_path = f"{MODELS_DIR}/ml_{currency}_ensemble.joblib"
        joblib.dump(self.models[currency]['ensemble'], ensemble_path)
        
        # Save preprocessors
        scaler_path = f"{MODELS_DIR}/ml_{currency}_scaler.joblib"
        selector_path = f"{MODELS_DIR}/ml_{currency}_selector.joblib"
        
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_selector, selector_path)
        
        logger.info(f"Models saved for {currency}")
    
    def load_models(self, currency: str = 'USD'):
        """Load trained models from disk"""
        try:
            # Load individual models
            individual_models = {}
            model_names = ['random_forest', 'gradient_boosting', 'svm', 
                          'logistic_regression', 'knn', 'naive_bayes']
            
            for name in model_names:
                model_path = f"{MODELS_DIR}/ml_{currency}_{name}.joblib"
                individual_models[name] = joblib.load(model_path)
            
            # Load ensemble model
            ensemble_path = f"{MODELS_DIR}/ml_{currency}_ensemble.joblib"
            ensemble_model = joblib.load(ensemble_path)
            
            # Load preprocessors
            scaler_path = f"{MODELS_DIR}/ml_{currency}_scaler.joblib"
            selector_path = f"{MODELS_DIR}/ml_{currency}_selector.joblib"
            
            self.scaler = joblib.load(scaler_path)
            self.feature_selector = joblib.load(selector_path)
            
            self.models[currency] = {
                'individual': individual_models,
                'ensemble': ensemble_model
            }
            
            self.is_trained = True
            logger.info(f"Models loaded for {currency}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def plot_feature_importance(self, currency: str = 'USD', top_n: int = 20):
        """Plot feature importance"""
        importance = self.get_feature_importance(currency)
        
        # Get top N features
        top_features = dict(list(importance.items())[:top_n])
        
        plt.figure(figsize=(12, 8))
        features = list(top_features.keys())
        values = list(top_features.values())
        
        plt.barh(range(len(features)), values)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance for {currency} Currency Detection')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, currency: str = 'USD'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        plt.title(f'Confusion Matrix - {currency} Currency Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    n_samples = 1000
    n_features = 50
    
    # Generate synthetic features
    np.random.seed(42)
    X_fake = np.random.randn(n_samples//2, n_features)
    X_real = np.random.randn(n_samples//2, n_features) + 1  # Real currency has different distribution
    
    X = np.vstack([X_fake, X_real])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    # Convert to features dictionary format
    features_dict = {f'feature_{i}': X[:, i].tolist() for i in range(n_features)}
    labels = y.tolist()
    
    # Create and train detector
    detector = MLCurrencyDetector()
    detector.train(features_dict, labels, 'USD')
    
    # Make predictions
    test_features = {f'feature_{i}': X[:100, i].tolist() for i in range(n_features)}
    predictions, probabilities = detector.predict(test_features, 'USD')
    
    print(f"Sample predictions: {predictions[:10]}")
    print(f"Sample probabilities: {probabilities[:5, 1]}")  # Probability of being real