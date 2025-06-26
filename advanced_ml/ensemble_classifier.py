"""
Advanced Ensemble Machine Learning System for Currency Detection
Professional-grade ML pipeline with state-of-the-art algorithms
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, learning_curve
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional
import time
from pathlib import Path

# Advanced ensemble methods
try:
    from sklearn.ensemble import StackingClassifier
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingClassifier
    ADVANCED_ENSEMBLE = True
except ImportError:
    ADVANCED_ENSEMBLE = False
    logging.warning("Advanced ensemble methods not available")

# XGBoost and LightGBM if available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedEnsembleClassifier:
    """Professional ensemble classifier for currency authentication"""
    
    def __init__(self, currency: str = 'USD'):
        self.currency = currency
        self.models = {}
        self.preprocessors = {}
        self.feature_selectors = {}
        self.is_trained = False
        self.feature_importance_ = None
        self.training_scores = {}
        self.best_params = {}
        
        # Initialize scalers
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        
        # Model configurations
        self.model_configs = self._get_model_configurations()
        
    def _get_model_configurations(self) -> Dict:
        """Get optimized model configurations"""
        configs = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'extra_trees': {
                'model': ExtraTreesClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'svm': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01],
                    'kernel': ['rbf', 'poly', 'sigmoid']
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            },
            'naive_bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                }
            },
            'mlp': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            },
            'ada_boost': {
                'model': AdaBoostClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'algorithm': ['SAMME', 'SAMME.R']
                }
            },
            'lda': {
                'model': LinearDiscriminantAnalysis(),
                'params': {
                    'solver': ['svd', 'lsqr', 'eigen'],
                    'shrinkage': [None, 'auto', 0.1, 0.5]
                }
            }
        }
        
        # Add advanced models if available
        if ADVANCED_ENSEMBLE:
            configs['hist_gradient_boosting'] = {
                'model': HistGradientBoostingClassifier(random_state=42),
                'params': {
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_iter': [100, 200],
                    'max_depth': [3, 5, 7],
                    'min_samples_leaf': [5, 10, 20]
                }
            }
        
        if XGBOOST_AVAILABLE:
            configs['xgboost'] = {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            }
        
        if LIGHTGBM_AVAILABLE:
            configs['lightgbm'] = {
                'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        
        return configs
    
    def preprocess_features(self, X: np.ndarray, y: np.ndarray = None, 
                          scaler_type: str = 'standard', 
                          feature_selection: str = 'selectkbest',
                          n_features: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Advanced feature preprocessing pipeline"""
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Feature scaling
        if scaler_type not in self.scalers:
            scaler_type = 'standard'
        
        scaler = self.scalers[scaler_type]
        
        if y is not None:  # Training phase
            X_scaled = scaler.fit_transform(X)
            self.preprocessors['scaler'] = scaler
        else:  # Prediction phase
            X_scaled = self.preprocessors['scaler'].transform(X)
        
        # Feature selection
        if y is not None:  # Training phase
            X_selected, selector = self._select_features(X_scaled, y, feature_selection, n_features)
            self.preprocessors['feature_selector'] = selector
        else:  # Prediction phase
            if 'feature_selector' in self.preprocessors:
                X_selected = self.preprocessors['feature_selector'].transform(X_scaled)
            else:
                X_selected = X_scaled
        
        return X_selected, y
    
    def _handle_missing_values(self, X: np.ndarray) -> np.ndarray:
        """Handle missing values in features"""
        # Replace inf with nan
        X = np.where(np.isinf(X), np.nan, X)
        
        # Fill nan with median
        if np.any(np.isnan(X)):
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        return X
    
    def _select_features(self, X: np.ndarray, y: np.ndarray, 
                        method: str, n_features: int) -> Tuple[np.ndarray, Any]:
        """Advanced feature selection"""
        
        if method == 'selectkbest':
            selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
        elif method == 'rfe':
            base_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator=base_estimator, n_features_to_select=min(n_features, X.shape[1]))
        elif method == 'model_based':
            base_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = SelectFromModel(estimator=base_estimator, max_features=min(n_features, X.shape[1]))
        else:
            # No feature selection
            return X, None
        
        X_selected = selector.fit_transform(X, y)
        logger.info(f"Feature selection: {X.shape[1]} -> {X_selected.shape[1]} features")
        
        return X_selected, selector
    
    def train_individual_models(self, X: np.ndarray, y: np.ndarray, 
                              optimize_hyperparameters: bool = True) -> Dict[str, Any]:
        """Train individual models with hyperparameter optimization"""
        
        logger.info(f"Training individual models for {self.currency}...")
        
        results = {}
        
        for model_name, config in self.model_configs.items():
            logger.info(f"Training {model_name}...")
            
            try:
                if optimize_hyperparameters:
                    # Hyperparameter optimization
                    model = self._optimize_hyperparameters(
                        config['model'], config['params'], X, y, model_name
                    )
                else:
                    model = config['model']
                    model.fit(X, y)
                
                # Cross-validation evaluation
                cv_scores = self._evaluate_model_cv(model, X, y)
                
                # Store model and results
                self.models[model_name] = model
                results[model_name] = {
                    'model': model,
                    'cv_scores': cv_scores,
                    'mean_cv_score': np.mean(cv_scores),
                    'std_cv_score': np.std(cv_scores)
                }
                
                logger.info(f"{model_name} - CV Score: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return results
    
    def _optimize_hyperparameters(self, model: Any, param_grid: Dict, 
                                 X: np.ndarray, y: np.ndarray, 
                                 model_name: str) -> Any:
        """Optimize hyperparameters using GridSearchCV"""
        
        # Use StratifiedKFold for balanced splits
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        # Store best parameters
        self.best_params[model_name] = grid_search.best_params_
        
        logger.debug(f"{model_name} best params: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def _evaluate_model_cv(self, model: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate model using cross-validation"""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        return scores
    
    def create_ensemble_models(self, individual_results: Dict) -> Dict[str, Any]:
        """Create ensemble models from individual models"""
        
        logger.info("Creating ensemble models...")
        
        ensemble_models = {}
        
        # Get top performing models
        sorted_models = sorted(individual_results.items(), 
                             key=lambda x: x[1]['mean_cv_score'], reverse=True)
        
        top_models = [(name, results['model']) for name, results in sorted_models[:7]]
        
        # Voting Classifier (Hard voting)
        voting_hard = VotingClassifier(
            estimators=top_models,
            voting='hard'
        )
        ensemble_models['voting_hard'] = voting_hard
        
        # Voting Classifier (Soft voting)
        voting_soft = VotingClassifier(
            estimators=top_models,
            voting='soft'
        )
        ensemble_models['voting_soft'] = voting_soft
        
        # Stacking Classifier if available
        if ADVANCED_ENSEMBLE:
            try:
                stacking_clf = StackingClassifier(
                    estimators=top_models[:5],  # Use top 5 models
                    final_estimator=LogisticRegression(random_state=42),
                    cv=3
                )
                ensemble_models['stacking'] = stacking_clf
            except Exception as e:
                logger.warning(f"Stacking classifier creation failed: {e}")
        
        return ensemble_models
    
    def train_ensemble_models(self, ensemble_models: Dict, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train ensemble models"""
        
        logger.info("Training ensemble models...")
        
        ensemble_results = {}
        
        for ensemble_name, ensemble_model in ensemble_models.items():
            logger.info(f"Training {ensemble_name}...")
            
            try:
                # Train ensemble
                ensemble_model.fit(X, y)
                
                # Evaluate
                cv_scores = self._evaluate_model_cv(ensemble_model, X, y)
                
                ensemble_results[ensemble_name] = {
                    'model': ensemble_model,
                    'cv_scores': cv_scores,
                    'mean_cv_score': np.mean(cv_scores),
                    'std_cv_score': np.std(cv_scores)
                }
                
                # Store in models dict
                self.models[ensemble_name] = ensemble_model
                
                logger.info(f"{ensemble_name} - CV Score: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
                
            except Exception as e:
                logger.error(f"Error training {ensemble_name}: {e}")
                continue
        
        return ensemble_results
    
    def train_complete_system(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the complete ensemble system"""
        
        logger.info(f"Training complete ensemble system for {self.currency}...")
        start_time = time.time()
        
        # Preprocess features
        X_processed, y_processed = self.preprocess_features(X, y)
        
        # Train individual models
        individual_results = self.train_individual_models(X_processed, y_processed)
        
        # Create and train ensemble models
        ensemble_models = self.create_ensemble_models(individual_results)
        ensemble_results = self.train_ensemble_models(ensemble_models, X_processed, y_processed)
        
        # Combine all results
        all_results = {**individual_results, **ensemble_results}
        
        # Find best model
        best_model_name = max(all_results.keys(), key=lambda x: all_results[x]['mean_cv_score'])
        best_model = all_results[best_model_name]['model']
        
        self.best_model = best_model
        self.best_model_name = best_model_name
        self.is_trained = True
        
        # Calculate feature importance if available
        self._calculate_feature_importance()
        
        # Store training scores
        self.training_scores = all_results
        
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best model: {best_model_name} (CV Score: {all_results[best_model_name]['mean_cv_score']:.4f})")
        
        return all_results
    
    def _calculate_feature_importance(self):
        """Calculate feature importance from best model"""
        
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance_ = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            self.feature_importance_ = np.abs(self.best_model.coef_[0])
        else:
            # Try to get from individual models in ensemble
            if hasattr(self.best_model, 'estimators_'):
                importances = []
                for estimator in self.best_model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances.append(estimator.feature_importances_)
                if importances:
                    self.feature_importance_ = np.mean(importances, axis=0)
    
    def predict(self, X: np.ndarray, return_probabilities: bool = False) -> np.ndarray:
        """Make predictions using the best model"""
        
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_complete_system() first.")
        
        # Preprocess features
        X_processed, _ = self.preprocess_features(X)
        
        if return_probabilities:
            if hasattr(self.best_model, 'predict_proba'):
                return self.best_model.predict_proba(X_processed)
            else:
                # Return binary predictions as probabilities
                predictions = self.best_model.predict(X_processed)
                probabilities = np.zeros((len(predictions), 2))
                probabilities[predictions == 0, 0] = 1.0
                probabilities[predictions == 1, 1] = 1.0
                return probabilities
        else:
            return self.best_model.predict(X_processed)
    
    def evaluate_on_test_set(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Comprehensive evaluation on test set"""
        
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_complete_system() first.")
        
        logger.info("Evaluating on test set...")
        
        # Preprocess test features
        X_processed, _ = self.preprocess_features(X_test)
        
        # Get predictions for all models
        results = {}
        
        for model_name, model in self.models.items():
            try:
                # Predictions
                y_pred = model.predict(X_processed)
                
                # Probabilities if available
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_processed)[:, 1]
                else:
                    y_prob = y_pred.astype(float)
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                    'auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0
                }
                
                results[model_name] = {
                    'predictions': y_pred,
                    'probabilities': y_prob,
                    'metrics': metrics
                }
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        return results
    
    def plot_model_comparison(self, results: Dict) -> None:
        """Plot model comparison"""
        
        # Extract metrics for plotting
        model_names = []
        accuracies = []
        f1_scores = []
        
        for model_name, result in results.items():
            if 'metrics' in result:
                model_names.append(model_name)
                accuracies.append(result['metrics']['accuracy'])
                f1_scores.append(result['metrics']['f1'])
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        ax1.barh(model_names, accuracies)
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_xlim(0, 1)
        
        # F1-score comparison
        ax2.barh(model_names, f1_scores)
        ax2.set_xlabel('F1-Score')
        ax2.set_title('Model F1-Score Comparison')
        ax2.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str] = None, top_n: int = 20) -> None:
        """Plot feature importance"""
        
        if self.feature_importance_ is None:
            logger.warning("Feature importance not available")
            return
        
        # Get top features
        indices = np.argsort(self.feature_importance_)[-top_n:]
        
        if feature_names:
            features = [feature_names[i] for i in indices]
        else:
            features = [f'Feature_{i}' for i in indices]
        
        importances = self.feature_importance_[indices]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importances)), importances)
        plt.yticks(range(len(importances)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances - {self.best_model_name}')
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curves(self, X: np.ndarray, y: np.ndarray) -> None:
        """Plot learning curves for the best model"""
        
        if not self.is_trained:
            logger.warning("Model not trained")
            return
        
        train_sizes, train_scores, val_scores = learning_curve(
            self.best_model, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        
        plt.plot(train_sizes, val_mean, 'o-', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title(f'Learning Curves - {self.best_model_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'models': self.models,
            'preprocessors': self.preprocessors,
            'best_model_name': self.best_model_name,
            'feature_importance': self.feature_importance_,
            'training_scores': self.training_scores,
            'currency': self.currency,
            'best_params': self.best_params
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model"""
        
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.preprocessors = model_data['preprocessors']
        self.best_model_name = model_data['best_model_name']
        self.best_model = self.models[self.best_model_name]
        self.feature_importance_ = model_data.get('feature_importance')
        self.training_scores = model_data.get('training_scores', {})
        self.currency = model_data.get('currency', self.currency)
        self.best_params = model_data.get('best_params', {})
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        
        if not self.is_trained:
            return {"status": "Model not trained"}
        
        summary = {
            'currency': self.currency,
            'best_model': self.best_model_name,
            'total_models': len(self.models),
            'training_scores': {}
        }
        
        # Add training scores
        for model_name, scores in self.training_scores.items():
            summary['training_scores'][model_name] = {
                'mean_cv_score': scores['mean_cv_score'],
                'std_cv_score': scores['std_cv_score']
            }
        
        # Add feature information
        if self.feature_importance_ is not None:
            summary['n_features'] = len(self.feature_importance_)
            summary['top_5_features'] = [
                f"Feature_{i}" for i in np.argsort(self.feature_importance_)[-5:][::-1]
            ]
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Test the ensemble classifier
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000, n_features=100, n_informative=20,
        n_redundant=10, n_clusters_per_class=1, random_state=42
    )
    
    # Create classifier
    classifier = AdvancedEnsembleClassifier('USD')
    
    # Train system
    results = classifier.train_complete_system(X, y)
    
    # Print results
    print("🎯 ADVANCED ENSEMBLE CLASSIFIER RESULTS")
    print("="*50)
    
    for model_name, result in results.items():
        print(f"{model_name:20s}: {result['mean_cv_score']:.4f} (±{result['std_cv_score']:.4f})")
    
    print(f"\n🏆 Best Model: {classifier.best_model_name}")
    print(f"📊 Total Models: {len(classifier.models)}")
    print("\n✅ Professional ensemble system ready for deployment!")