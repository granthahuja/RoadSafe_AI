import cv2
import os
import numpy as np
from PIL import Image
from modules.severity import load_visual_model, analyze_frame
from tkinter import Tk, filedialog

root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select accident image",
    filetypes=[("Images", "*.jpg *.jpeg *.png *.webp")]
)

if not image_path:
    print("No image selected.")
    raise SystemExit

print("\nSelected file:", image_path)
print("File exists:", os.path.exists(image_path))
print("File size:", os.path.getsize(image_path), "bytes")

try:
    pil_image = Image.open(image_path).convert("RGB")
    print("PIL image size:", pil_image.size)

    # Convert RGB to OpenCV BGR
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    print("OpenCV image shape:", image.shape)

    model = load_visual_model()
    result = analyze_frame(model, image)

    print("\nYOLO Analysis:")
    print(result)

except Exception as e:
    print("\nERROR:", repr(e))