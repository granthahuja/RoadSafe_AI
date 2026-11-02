from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path
import tensorflow as tf
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "road_accident_mobilenetv2.keras"

print("\n========== ROADSAFE-AI MODEL VERIFICATION ==========\n")

# 1. Check model file
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Place your .keras file beside verify_model.py."
    )

print("Model file found:", MODEL_PATH.name)
print("File size:", round(MODEL_PATH.stat().st_size / (1024 * 1024), 2), "MB")

# 2. Load model
print("\nLoading MobileNetV2...")

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    custom_objects = {
        "preprocess_input": preprocess_input
    },
    compile=False,
    safe_mode=False
)

print("Model loaded successfully!")

# 3. Inspect architecture
print("\n========== MODEL DETAILS ==========")

print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

# 4. Inspect output activation
last_layer = model.layers[-1]
activation = getattr(last_layer, "activation", None)

print(
    "Output activation:",
    getattr(activation, "__name__", "unknown")
)

# 5. Inspect model preprocessing layers
print("\n========== PREPROCESSING INSPECTION ==========")

for layer in model.layers:
    if "lambda" in layer.name.lower() or "preprocess" in layer.name.lower():
        print("Possible preprocessing layer:", layer.name)

# 6. Dummy inference test
print("\n========== INFERENCE TEST ==========")

height = int(model.input_shape[1])
width = int(model.input_shape[2])

dummy_image = np.zeros(
    (1, height, width, 3),
    dtype=np.float32
)

prediction = model.predict(dummy_image, verbose=0)

print("Input dimensions:", dummy_image.shape)
print("Raw prediction:", prediction)

print("\n========== VERIFICATION COMPLETE ==========")