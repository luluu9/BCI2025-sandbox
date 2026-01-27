import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from mne.time_frequency import psd_array_welch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class CNN(BaseEstimator, ClassifierMixin):
    def __init__(self, sfreq=160.0, epochs=50, batch_size=64, verbose=0):
        self.sfreq = sfreq
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_ = None
        self.le_ = None
        self.norm_params_ = None

    def _compute_features(self, X):
        psds, _ = psd_array_welch(X, sfreq=self.sfreq, fmin=8.0, fmax=32.0, n_fft=256, verbose=False)
        # psds: (n_epochs, n_channels, n_freqs)
        
        # Transpose to (n_epochs, n_freqs, n_channels) for the model
        return np.transpose(psds, (0, 2, 1))

    def _build_model(self, input_shape, n_classes):
        inputs = keras.Input(shape=input_shape)
        
        x = layers.Conv1D(16, kernel_size=2, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv1D(32, kernel_size=2, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)

        outputs = layers.Dense(n_classes, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(5e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, X, y):
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)
        
        X_feat = self._compute_features(X)
        
        # Normalization
        mean = X_feat.mean(axis=0, keepdims=True)
        std = X_feat.std(axis=0, keepdims=True) + 1e-8
        self.norm_params_ = {'mean': mean, 'std': std}
        X_norm = (X_feat - mean) / std
        
        self.model_ = self._build_model(X_norm.shape[1:], n_classes)
        
        # Validation split for potential EarlyStopping
        X_tr, X_val, y_tr, y_val = train_test_split(X_norm, y_encoded, test_size=0.1, random_state=42)
        
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=self.verbose),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
        ]
        
        self.model_.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=self.verbose
        )
        return self

    def predict_proba(self, X):
        X_feat = self._compute_features(X)
        mean = self.norm_params_['mean']
        std = self.norm_params_['std']
        X_norm = (X_feat - mean) / std
        return self.model_.predict(X_norm, verbose=0)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.le_.inverse_transform(probs.argmax(axis=1))
