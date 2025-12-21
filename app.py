import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import math

# --- Fonctions utilitaires ---
def calculate_distance(pointA, pointB):
    if pointA is None or pointB is None:
        return None
    return math.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)

def calculate_body_metrics(points):
    # Estimations des tours
    tour_de_poitrine = (calculate_distance(points[2], points[5]) +
                        calculate_distance(points[2], points[11]) +
                        calculate_distance(points[5], points[8]))
    tour_de_hanche = calculate_distance(points[8], points[11])
    longueur_bras_droit_epaule_coude = calculate_distance(points[2], points[3])
    longueur_bras_droit_coude_poignet = calculate_distance(points[3], points[4])
    longueur_bras_droit_total = longueur_bras_droit_epaule_coude + longueur_bras_droit_coude_poignet
    longueur_bras_gauche_epaule_coude = calculate_distance(points[5], points[6])
    longueur_bras_gauche_coude_poignet = calculate_distance(points[6], points[7])
    longueur_bras_gauche_total = longueur_bras_gauche_epaule_coude + longueur_bras_gauche_coude_poignet
    longueur_jambe_droite_hanche_genou = calculate_distance(points[8], points[9])
    longueur_jambe_droite_genou_cheville = calculate_distance(points[9], points[10])
    longueur_jambe_droite_total = longueur_jambe_droite_hanche_genou + longueur_jambe_droite_genou_cheville
    longueur_jambe_gauche_hanche_genou = calculate_distance(points[11], points[12])
    longueur_jambe_gauche_genou_cheville = calculate_distance(points[12], points[13])
    longueur_jambe_gauche_total = longueur_jambe_gauche_hanche_genou + longueur_jambe_gauche_genou_cheville
    largeur_epaule = calculate_distance(points[2], points[5])
    longueur_torse = calculate_distance(points[1], points[8]) if points[8] is not None else calculate_distance(points[1], points[11])
    longueur_manche_courte = max(longueur_bras_droit_epaule_coude, longueur_bras_gauche_epaule_coude)
    longueur_manche_longue = max(longueur_bras_droit_total, longueur_bras_gauche_total)
    longueur_culotte = max(longueur_jambe_droite_hanche_genou, longueur_jambe_gauche_hanche_genou)
    longueur_pantalon = max(longueur_jambe_droite_total, longueur_jambe_gauche_total)
    results = {
        "Tour de poitrine": tour_de_poitrine,
        "Tour de hanche": tour_de_hanche,
        "longueur_bras_droit_epaule_coude": longueur_bras_droit_epaule_coude,
        "longueur_bras_droit_coude_poignet": longueur_bras_droit_coude_poignet,
        "Longueur totale du bras droit": longueur_bras_droit_total,
        "longueur_bras_gauche_epaule_coude": longueur_bras_gauche_epaule_coude,
        "longueur_bras_gauche_coude_poignet": longueur_bras_gauche_coude_poignet,
        "Longueur totale du bras gauche": longueur_bras_gauche_total,
        "longueur_jambe_droite_hanche_genou": longueur_jambe_droite_hanche_genou,
        "longueur_jambe_droite_genou_cheville": longueur_jambe_droite_genou_cheville,
        "Longueur totale de la jambe droite": longueur_jambe_droite_total,
        "longueur_jambe_gauche_hanche_genou": longueur_jambe_gauche_hanche_genou,
        "longueur_jambe_gauche_genou_cheville": longueur_jambe_gauche_genou_cheville,
        "Longueur totale de la jambe gauche": longueur_jambe_gauche_total,
        "Largeur des epaules": largeur_epaule,
        "Longueur du torse": longueur_torse,
        "Longueur de la manche courte": longueur_manche_courte,
        "Longueur de la manche longue": longueur_manche_longue,
        "Longueur de la culotte": longueur_culotte,
        "Longueur du pantalon": longueur_pantalon
    }
    return results

def pixels_to_cm(pixels, dpi=96):
    return (pixels / dpi) * 2.54

# --- OpenPose (Caffe) ---
def detect_keypoints(frame, protoFile, weightsFile, threshold=0.1):
    nPoints = 18
    keypoints_names = [
        "Nose", "Neck", "Right Shoulder", "Right Elbow", "Right Wrist",
        "Left Shoulder", "Left Elbow", "Left Wrist", "Right Hip",
        "Right Knee", "Right Ankle", "Left Hip", "Left Knee",
        "Left Ankle", "Chest", "Right Eye", "Left Eye",
        "Right Ear", "Left Ear", "Background"
    ]
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (0, 128, 255), (128, 0, 255),
        (255, 128, 0), (128, 255, 0), (0, 255, 128), (0, 0, 128),
        (128, 0, 0), (0, 128, 0), (128, 128, 0), (164, 192, 156),
        (55, 85, 98), (192, 192, 192), (255, 128, 128)
    ]
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()
    H = output.shape[2]
    W = output.shape[3]
    points = []
    for i in range(nPoints):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
        if prob > threshold:
            cv2.circle(frameCopy, (int(x), int(y)), 8, colors[i], thickness=-1, lineType=cv2.FILLED)
            points.append((int(x), int(y)))
        else:
            points.append(None)
    return frameCopy, points

def draw_skeleton(frame, points):
    POSE_PAIRS = [
        (1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6),
        (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12),
        (12, 13), (1, 14), (14, 15), (0, 16), (0, 17)
    ]
    for pair in POSE_PAIRS:
        partA, partB = pair
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    return frame

# --- Application Streamlit ---
st.title("TailorAI - Mesure corporelle à partir d'une image")

st.write("""
1. Chargez une photo de vous (vue entière, de face, sur fond neutre).
2. Les points clés seront détectés automatiquement.
3. Les mesures seront affichées étape par étape.
""")

# 1. Chargement de l'image
uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    image = cv2.imread(tfile.name)
    if image is None:
        st.error("Erreur lors du chargement de l'image. Veuillez réessayer.")
        st.stop()
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Image chargée", use_column_width=True)

    # 2. Détection des points clés
    st.subheader("Détection des points clés (OpenPose)")
    protoFile = "./openpose/models/pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "./openpose/models/pose/coco/pose_iter_440000.caffemodel"
    try:
        frameCopy, points = detect_keypoints(image, protoFile, weightsFile)
    except Exception as e:
        st.error(f"Erreur lors de la détection des points clés : {e}")
        st.stop()
    st.image(cv2.cvtColor(frameCopy, cv2.COLOR_BGR2RGB), caption="Points clés détectés", use_column_width=True)

    # 3. Affichage du squelette
    st.subheader("Squelette détecté")
    skeleton_img = draw_skeleton(np.copy(image), points)
    st.image(cv2.cvtColor(skeleton_img, cv2.COLOR_BGR2RGB), caption="Squelette sur l'image", use_column_width=True)

    # 4. Calcul des mesures en pixels
    st.subheader("Mesures en pixels")
    body_metrics = calculate_body_metrics(points)
    st.write({k: f"{v:.2f} px" if v is not None else "-" for k, v in body_metrics.items()})

    # 5. Conversion en centimètres
    st.subheader("Mesures en centimètres (sans échelle)")
    dpi = 96
    cm_metrics = {k: pixels_to_cm(v, dpi) if v is not None else None for k, v in body_metrics.items()}
    st.write({k: f"{v:.2f} cm" if v is not None else "-" for k, v in cm_metrics.items()})

    # 6. Saisie de la taille réelle
    st.subheader("Entrez votre taille réelle (en cm)")
    user_real_height = st.number_input("Votre taille réelle (cm)", min_value=100.0, max_value=250.0, value=170.0)

    # 7. Calcul du facteur d'échelle et mesures réelles
    # On prend la plus grande distance jambe (cheville à tête)
    try:
        taille = max(
            calculate_distance(points[15], points[13]) if points[15] and points[13] else 0,
            calculate_distance(points[14], points[10]) if points[14] and points[10] else 0
        )
        image_measured_height = pixels_to_cm(taille, dpi)
        scale_factor = user_real_height / image_measured_height if image_measured_height else 1.0
        scale_factor *= 1.15505
    except Exception:
        scale_factor = 1.0

    st.subheader("Mesures corporelles réelles (cm)")
    real_metrics = {k: (pixels_to_cm(v, dpi) * scale_factor) if v is not None else None for k, v in body_metrics.items()}
    st.write({k: f"{v:.2f} cm" if v is not None else "-" for k, v in real_metrics.items()})

    # 8. Affichage graphique final
    st.subheader("Aperçu graphique des mesures réelles")
    # Palette de couleurs
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (128, 0, 128), (255, 165, 0), (128, 128, 0), (0, 128, 128)
    ]
    height, width, _ = image.shape
    max_text_width = max([
        cv2.getTextSize(f"{metric}: {real_metrics[metric]:.2f} cm", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
        for metric in real_metrics if real_metrics[metric] is not None
    ], default=0)
    black_background_width = max_text_width + 20
    black_background = np.zeros((height, black_background_width, 3), dtype=np.uint8)
    combined_width = width + black_background_width
    combined_image = np.zeros((height, combined_width, 3), dtype=np.uint8)
    combined_image[:height, :width] = skeleton_img
    combined_image[:height, width:] = black_background
    y_offset = 30
    for idx, (metric, value) in enumerate(real_metrics.items()):
        if value is not None:
            color = colors[idx % len(colors)]
            cv2.putText(combined_image, f"{metric}: {value:.2f} cm", (width + 10, y_offset + idx * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, lineType=cv2.LINE_AA)
    st.image(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB), caption="Squelette et mesures réelles", use_column_width=True)