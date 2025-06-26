"""
Deep Learning algorithms for currency detection
Includes CNN models, transfer learning, and ensemble deep learning
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Tuple, List, Dict, Any, Optional
import os
from config import MODELS_DIR, DL_CONFIG, IMAGE_SIZE

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class DeepLearningDetector:
    """Deep Learning based currency authentication"""
    
    def __init__(self):
        self.models = {}
        self.histories = {}
        self.is_trained = False
        
        # Set GPU memory growth to avoid allocation issues
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger.warning(f"GPU configuration warning: {e}")
    
    def create_custom_cnn(self, input_shape: Tuple[int, int, int] = (*IMAGE_SIZE, 3)) -> keras.Model:
        """Create a custom CNN architecture for currency detection"""
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            # Classification layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        return model
    
    def create_transfer_learning_model(self, base_model_name: str = 'EfficientNetB0', 
                                     input_shape: Tuple[int, int, int] = (*IMAGE_SIZE, 3),
                                     trainable_layers: int = 20) -> keras.Model:
        """Create transfer learning model using pre-trained networks"""
        
        # Load pre-trained base model
        if base_model_name == 'VGG16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'ResNet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'EfficientNetB0':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'MobileNetV2':
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Fine-tuning: unfreeze top layers
        if trainable_layers > 0:
            base_model.trainable = True
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
        
        return model
    
    def create_attention_model(self, input_shape: Tuple[int, int, int] = (*IMAGE_SIZE, 3)) -> keras.Model:
        """Create CNN with attention mechanism for better feature focus"""
        
        inputs = layers.Input(shape=input_shape)
        
        # Feature extraction layers
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        
        # Attention mechanism
        attention = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
        x = layers.Multiply()([x, attention])
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def create_ensemble_model(self, models_list: List[keras.Model]) -> keras.Model:
        """Create ensemble model from multiple trained models"""
        
        # Get input shape from first model
        input_shape = models_list[0].input_shape[1:]
        inputs = layers.Input(shape=input_shape)
        
        # Get predictions from all models
        predictions = []
        for i, model in enumerate(models_list):
            # Make base models non-trainable in ensemble
            model.trainable = False
            pred = model(inputs)
            predictions.append(pred)
        
        # Average predictions
        if len(predictions) > 1:
            averaged = layers.Average()(predictions)
        else:
            averaged = predictions[0]
        
        ensemble_model = keras.Model(inputs=inputs, outputs=averaged)
        return ensemble_model
    
    def prepare_data_generators(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               batch_size: int = DL_CONFIG['batch_size']) -> Tuple[Any, Any]:
        """Prepare data generators with augmentation"""
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # Currency shouldn't be flipped
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
        
        return train_generator, val_generator
    
    def compile_model(self, model: keras.Model, learning_rate: float = DL_CONFIG['learning_rate']) -> keras.Model:
        """Compile model with optimizer, loss, and metrics"""
        
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def get_callbacks(self, model_name: str, currency: str = 'USD') -> List[callbacks.Callback]:
        """Get callbacks for training"""
        
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=DL_CONFIG['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=os.path.join(MODELS_DIR, f'dl_{currency}_{model_name}_best.h5'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        return callback_list
    
    def train_model(self, model: keras.Model, model_name: str,
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   currency: str = 'USD') -> keras.callbacks.History:
        """Train a single model"""
        
        logger.info(f"Training {model_name} for {currency} currency detection...")
        
        # Prepare data generators
        train_gen, val_gen = self.prepare_data_generators(X_train, y_train, X_val, y_val)
        
        # Get callbacks
        callback_list = self.get_callbacks(model_name, currency)
        
        # Train model
        history = model.fit(
            train_gen,
            epochs=DL_CONFIG['epochs'],
            validation_data=val_gen,
            callbacks=callback_list,
            verbose=1
        )
        
        return history
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        currency: str = 'USD'):
        """Train all deep learning models"""
        
        logger.info(f"Training deep learning models for {currency} currency detection...")
        
        # Create and train custom CNN
        custom_cnn = self.create_custom_cnn()
        custom_cnn = self.compile_model(custom_cnn)
        history_cnn = self.train_model(custom_cnn, 'custom_cnn', X_train, y_train, X_val, y_val, currency)
        
        # Create and train transfer learning models
        transfer_models = ['EfficientNetB0', 'ResNet50', 'MobileNetV2']
        trained_models = {'custom_cnn': custom_cnn}
        histories = {'custom_cnn': history_cnn}
        
        for base_model_name in transfer_models:
            try:
                model = self.create_transfer_learning_model(base_model_name)
                model = self.compile_model(model)
                history = self.train_model(model, base_model_name.lower(), X_train, y_train, X_val, y_val, currency)
                
                trained_models[base_model_name.lower()] = model
                histories[base_model_name.lower()] = history
                
            except Exception as e:
                logger.error(f"Error training {base_model_name}: {e}")
        
        # Create and train attention model
        try:
            attention_model = self.create_attention_model()
            attention_model = self.compile_model(attention_model)
            history_attention = self.train_model(attention_model, 'attention_cnn', X_train, y_train, X_val, y_val, currency)
            
            trained_models['attention_cnn'] = attention_model
            histories['attention_cnn'] = history_attention
            
        except Exception as e:
            logger.error(f"Error training attention model: {e}")
        
        # Create ensemble model
        if len(trained_models) > 1:
            ensemble_model = self.create_ensemble_model(list(trained_models.values()))
            ensemble_model = self.compile_model(ensemble_model, learning_rate=DL_CONFIG['learning_rate']/10)
            
            trained_models['ensemble'] = ensemble_model
        
        # Store models and histories
        self.models[currency] = trained_models
        self.histories[currency] = histories
        self.is_trained = True
        
        logger.info(f"Deep learning models trained successfully for {currency}")
    
    def predict(self, X: np.ndarray, currency: str = 'USD', model_name: str = 'ensemble') -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using trained models"""
        
        if not self.is_trained or currency not in self.models:
            raise ValueError(f"Models not trained for {currency}")
        
        if model_name not in self.models[currency]:
            model_name = 'ensemble' if 'ensemble' in self.models[currency] else list(self.models[currency].keys())[0]
        
        model = self.models[currency][model_name]
        
        # Normalize input
        X_normalized = X.astype(np.float32) / 255.0
        
        # Get predictions
        probabilities = model.predict(X_normalized)
        predictions = (probabilities > 0.5).astype(int).flatten()
        
        return predictions, probabilities.flatten()
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray, currency: str = 'USD') -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models"""
        
        if not self.is_trained or currency not in self.models:
            raise ValueError(f"Models not trained for {currency}")
        
        results = {}
        X_test_normalized = X_test.astype(np.float32) / 255.0
        
        for model_name, model in self.models[currency].items():
            try:
                # Get predictions
                probabilities = model.predict(X_test_normalized)
                predictions = (probabilities > 0.5).astype(int).flatten()
                
                # Calculate metrics
                accuracy = np.mean(predictions == y_test)
                
                # Precision, Recall, F1
                tp = np.sum((predictions == 1) & (y_test == 1))
                fp = np.sum((predictions == 1) & (y_test == 0))
                fn = np.sum((predictions == 0) & (y_test == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
        
        return results
    
    def plot_training_history(self, currency: str = 'USD'):
        """Plot training history for all models"""
        
        if currency not in self.histories:
            logger.warning(f"No training history found for {currency}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {currency} Currency Detection', fontsize=16)
        
        metrics = ['loss', 'accuracy', 'precision', 'recall']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            for model_name, history in self.histories[currency].items():
                if metric in history.history:
                    ax.plot(history.history[metric], label=f'{model_name}_train')
                    if f'val_{metric}' in history.history:
                        ax.plot(history.history[f'val_{metric}'], label=f'{model_name}_val', linestyle='--')
            
            ax.set_title(f'{metric.capitalize()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self, currency: str = 'USD'):
        """Save trained models to disk"""
        
        if not self.is_trained or currency not in self.models:
            raise ValueError(f"Models not trained for {currency}")
        
        for model_name, model in self.models[currency].items():
            model_path = os.path.join(MODELS_DIR, f'dl_{currency}_{model_name}.h5')
            model.save(model_path)
        
        logger.info(f"Deep learning models saved for {currency}")
    
    def load_models(self, currency: str = 'USD'):
        """Load trained models from disk"""
        
        try:
            model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith(f'dl_{currency}_') and f.endswith('.h5')]
            
            loaded_models = {}
            for model_file in model_files:
                model_name = model_file.replace(f'dl_{currency}_', '').replace('.h5', '')
                model_path = os.path.join(MODELS_DIR, model_file)
                loaded_models[model_name] = keras.models.load_model(model_path)
            
            self.models[currency] = loaded_models
            self.is_trained = True
            
            logger.info(f"Deep learning models loaded for {currency}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    n_samples = 1000
    X = np.random.randint(0, 255, (n_samples, *IMAGE_SIZE, 3), dtype=np.uint8)
    y = np.random.randint(0, 2, n_samples)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    val_split_idx = int(0.8 * len(X_train))
    X_train, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
    y_train, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
    
    # Create detector
    detector = DeepLearningDetector()
    
    # Train models (this would take time with real data)
    logger.info("This is a demonstration. In practice, training would take significant time.")
    
    # Show model architectures
    custom_model = detector.create_custom_cnn()
    print("Custom CNN Model Summary:")
    custom_model.summary()
    
    print(f"\nCreated {len(detector.create_transfer_learning_model('EfficientNetB0').layers)} layer transfer learning model")