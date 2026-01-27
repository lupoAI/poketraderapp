# export_coreml.py
from ultralytics import YOLO

# Load the model
model = YOLO("pokemon-yolo11n-seg-v3.pt")

# Export to CoreML
# nms=True: include Non-Maximum Suppression in the model
# half=True: use FP16 precision for better mobile performance
model.export(format="coreml", nms=True, half=True)

print("Export finished. Look for .mlpackage in the current directory.")
