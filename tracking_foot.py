import cv2
from ultralytics import YOLO

# 1. Charger le modèle YOLOv8 pré-entraîné (le modèle 'medium' ou 'pose' est souvent un bon compromis)
# Le modèle de base yolov8m.pt sait déjà détecter les personnes (class 0) et les ballons (class 32)
model = YOLO("yolov8m.pt")

# 2. Ouvrir la vidéo source
video_path = "match_extrait.mp4"  # Mets le nom exact de ton fichier vidéo
cap = cv2.VideoCapture(video_path)

# Vérifier si la vidéo s'ouvre correctement
if not cap.isOpened():
    print(f"Erreur : Impossible d'ouvrir la vidéo {video_path}")
    exit()

print("Début du traitement vidéo... Appuie sur 'q' pour quitter.")

# 3. Boucle de lecture de la vidéo image par image
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Fin de la vidéo

    # 4. Appliquer YOLOv8 + ByteTrack sur l'image actuelle
    # persist=True indique au tracker (ByteTrack) de garder en mémoire les objets d'une frame à l'autre
    results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    # 5. Récupérer l'image annotée automatiquement par YOLO (boîtes et IDs)
    annotated_frame = results[0].plot()

    # (Optionnel) Si tu veux voir ce qu'il se passe pendant le calcul :
    # Redimensionner l'affichage si la vidéo est trop grande pour ton écran
    display_frame = cv2.resize(annotated_frame, (1280, 720))
    cv2.imshow("IA Football Tracking - Groupe 2", display_frame)

    # Permet de quitter proprement en appuyant sur la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
print("Traitement terminé !")
