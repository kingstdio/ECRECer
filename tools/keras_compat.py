"""Keras compatibility helpers for legacy ECRECer models."""

try:
    import tf_keras as keras
except ImportError:  # pragma: no cover - fallback for legacy environments
    try:
        from tensorflow import keras
    except ImportError:
        import keras

Model = keras.models.Model
load_model = keras.models.load_model
Adam = keras.optimizers.Adam
Input = keras.layers.Input
Dense = keras.layers.Dense
GRU = keras.layers.GRU
Bidirectional = keras.layers.Bidirectional
Layer = keras.layers.Layer
TensorBoard = keras.callbacks.TensorBoard
ModelCheckpoint = keras.callbacks.ModelCheckpoint
