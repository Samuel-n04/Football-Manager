import cv2
import ultralytics
from ultralytics import YOLO
import SoccerNet

print("--- Vérification des versions ---")
print(f"OpenCV version: {cv2.__version__}")
print(f"Ultralytics (YOLOv8) version: {ultralytics.__version__}")

# Tester le chargement d'un modèle YOLOv8 nano pré-entraîné
try:
    model = YOLO("yolov8n.pt")
    print("Modèle YOLOv8 chargé avec succès !")
except Exception as e:
    print(f"Erreur lors du chargement de YOLO : {e}")
