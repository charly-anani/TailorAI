import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import math
from mpl_toolkits.mplot3d import Axes3D

st.title("TailorAI 3D - Mesure corporelle à partir de deux photos")
st.write("""
Chargez deux photos de vous : une de face et une de profil (vue entière, sur fond neutre).
Les points clés seront détectés automatiquement sur chaque image.
Les mesures 3D finales s'afficheront directement, avec possibilité d'ajuster votre taille réelle pour une calibration précise.
""")

# --- Fonctions utilitaires ---
def detect_pose_openpose(image):
    protoFile = "./openpose/models/pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "./openpose/models/pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    frame = image.copy()
    frameHeight, frameWidth = frame.shape[:2]
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0/255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()
    points = []
    threshold = 0.1
    H, W = output.shape[2], output.shape[3]
    for i in range(nPoints):
        probMap = output[0, i, :, :]
        _, prob, _, point = cv2.minMaxLoc(probMap)
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
        if prob > threshold and i not in [14, 15, 16, 17]:
            points.append((int(x), int(y)))
        else:
            points.append(None)
    return frame, points

def detect_a4_calibration(frames):
    a4_heights = []
    a4_real_cm = 29.7
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(cnt) > 30000:
                rect = cv2.minAreaRect(cnt)
                w, h = rect[1]
                if w == 0 or h == 0:
                    continue
                ratio = max(w, h) / min(w, h)
                if 1.3 < ratio < 1.5:
                    box = cv2.boxPoints(rect)
                    height_px = max(
                        np.linalg.norm(box[0] - box[2]),
                        np.linalg.norm(box[1] - box[3])
                    )
                    a4_heights.append(height_px)
                    break
    if a4_heights:
        height_px = np.mean(a4_heights)
        scale = a4_real_cm / height_px
        return scale, True
    return None, False

def simple_3d_triangulation(points_front, points_side):
    points_3d = []
    for pf, ps in zip(points_front, points_side):
        if pf is None or ps is None:
            points_3d.append(None)
        else:
            x = (pf[0] + ps[0]) / 2.0
            y = (pf[1] + ps[1]) / 2.0
            z = abs(pf[0] - ps[0]) * 0.3
            points_3d.append(np.array([x, y, z], dtype=float))
    return points_3d

def calculate_tailor_body_metrics(points_3d, scale_factor):
    def get_point(i):
        if i >= len(points_3d) or points_3d[i] is None:
            return None
        return points_3d[i]
    res = {}
    p_nose      = get_point(0)
    p_neck      = get_point(1)
    p_rshoulder = get_point(2)
    p_relbow    = get_point(3)
    p_rwrist    = get_point(4)
    p_lshoulder = get_point(5)
    p_lelbow    = get_point(6)
    p_lwrist    = get_point(7)
    p_rhip      = get_point(8)
    p_rknee     = get_point(9)
    p_rankle    = get_point(10)
    p_lhip      = get_point(11)
    p_lknee     = get_point(12)
    p_lankle    = get_point(13)
    def ellipse_circumference(a, b):
        return math.pi * (3*(a+b) - math.sqrt((3*a + b) * (a + 3*b)))
    if p_rshoulder is not None and p_lshoulder is not None:
        res["Largeur épaules"] = np.linalg.norm(p_rshoulder - p_lshoulder) * scale_factor
    if p_rshoulder is not None and p_lshoulder is not None:
        a = (np.linalg.norm(p_rshoulder - p_lshoulder) * scale_factor) / 2.0
        if p_rhip is not None and p_lhip is not None:
            depth_z = (abs(p_rshoulder[2] - p_rhip[2]) + abs(p_lshoulder[2] - p_lhip[2])) / 2.0
        else:
            depth_z = 0.3 * a
        b = max(depth_z * scale_factor, 0.3 * a)
        res["Tour de poitrine"] = ellipse_circumference(a, b)
    if p_rhip is not None and p_lhip is not None:
        a_h = (np.linalg.norm(p_rhip - p_lhip) * scale_factor) / 2.0
        if p_rknee is not None and p_lknee is not None:
            depth_z_h = (abs(p_rhip[2] - p_rknee[2]) + abs(p_lhip[2] - p_lknee[2])) / 2.0
        else:
            depth_z_h = 0.35 * a_h
        b_h = max(depth_z_h * scale_factor, 0.35 * a_h)
        res["Tour de hanches"] = ellipse_circumference(a_h, b_h)
    manches_courtes = []
    if p_rshoulder is not None and p_relbow is not None:
        manches_courtes.append(np.linalg.norm(p_rshoulder - p_relbow) * scale_factor)
    if p_lshoulder is not None and p_lelbow is not None:
        manches_courtes.append(np.linalg.norm(p_lshoulder - p_lelbow) * scale_factor)
    if manches_courtes:
        res["Longueur manche courte"] = max(manches_courtes)
    manches_longues = []
    if p_rshoulder is not None and p_rwrist is not None:
        manches_longues.append(np.linalg.norm(p_rshoulder - p_rwrist) * scale_factor)
    if p_lshoulder is not None and p_lwrist is not None:
        manches_longues.append(np.linalg.norm(p_lshoulder - p_lwrist) * scale_factor)
    if manches_longues:
        res["Longueur manche longue"] = max(manches_longues)
    if p_neck is not None and (p_rhip is not None or p_lhip is not None):
        p_hip = p_rhip if p_rhip is not None else p_lhip
        res["Longueur buste"] = np.linalg.norm(p_neck - p_hip) * scale_factor
    culotte = []
    if p_rhip is not None and p_rknee is not None:
        culotte.append(np.linalg.norm(p_rhip - p_rknee) * scale_factor)
    if p_lhip is not None and p_lknee is not None:
        culotte.append(np.linalg.norm(p_lhip - p_lknee) * scale_factor)
    if culotte:
        res["Longueur culotte"] = max(culotte)
    pantalon = []
    if p_rhip is not None and p_rankle is not None:
        pantalon.append(np.linalg.norm(p_rhip - p_rankle) * scale_factor)
    if p_lhip is not None and p_lankle is not None:
        pantalon.append(np.linalg.norm(p_lhip - p_lankle) * scale_factor)
    if pantalon:
        res["Longueur pantalon"] = max(pantalon)
    chevilles = []
    if p_neck is not None and p_rankle is not None:
        chevilles.append(np.linalg.norm(p_neck - p_rankle) * scale_factor)
    if p_neck is not None and p_lankle is not None:
        chevilles.append(np.linalg.norm(p_neck - p_lankle) * scale_factor)
    if chevilles:
        res["Hauteur totale"] = max(chevilles)
    res["Points 3D détectés"] = str(sum(1 for p in points_3d if p is not None))
    return res

def estimate_tshirt_size(chest_circ_cm):
    if chest_circ_cm is None:
        return "Inconnue"
    c = chest_circ_cm
    if c < 94:
        return "S"
    elif c < 102:
        return "M"
    elif c < 110:
        return "L"
    elif c < 118:
        return "XL"
    else:
        return "XXL"

# --- Interface utilisateur ---
st.subheader("1. Uploadez deux photos : face et profil")
col1, col2 = st.columns(2)
with col1:
    uploaded_face = st.file_uploader("Photo de face", type=["jpg", "jpeg", "png"], key="face")
with col2:
    uploaded_profile = st.file_uploader("Photo de profil", type=["jpg", "jpeg", "png"], key="profil")


if uploaded_face and uploaded_profile:
    # Lecture et décodage robustes des images uploadées
    file_bytes_face = np.asarray(bytearray(uploaded_face.read()), dtype=np.uint8)
    image_face = cv2.imdecode(file_bytes_face, cv2.IMREAD_COLOR)
    file_bytes_profile = np.asarray(bytearray(uploaded_profile.read()), dtype=np.uint8)
    image_profile = cv2.imdecode(file_bytes_profile, cv2.IMREAD_COLOR)
    if image_face is None or image_profile is None:
        st.error("Erreur lors du chargement des images. Veuillez réessayer.")
        st.stop()

    # 1. Détection des points clés (sans affichage intermédiaire)
    try:
        _, points_face = detect_pose_openpose(image_face)
        _, points_profile = detect_pose_openpose(image_profile)
    except Exception as e:
        st.error(f"Erreur lors de la détection des points clés : {e}")
        st.stop()

    # 2. Calibration automatique (A4) ou manuelle, possibilité d'ajuster la taille
    scale_factor, a4_detected = detect_a4_calibration([image_face, image_profile])
    method = "A4"
    points_3d_tmp = simple_3d_triangulation(points_face, points_profile)
    p_neck = points_3d_tmp[1] if len(points_3d_tmp) > 1 else None
    p_ankle = points_3d_tmp[10] if len(points_3d_tmp) > 10 else points_3d_tmp[13]
    if not a4_detected:
        # On propose toujours d'ajuster la taille réelle
        st.info("Aucune feuille A4 détectée. Veuillez ajuster votre taille réelle (en cm) pour calibrer.")
        hauteur_px = np.linalg.norm(p_neck - p_ankle) if p_neck is not None and p_ankle is not None else None
        user_height = st.number_input("Votre taille réelle (cm)", min_value=100.0, max_value=250.0, value=170.0)
        scale_factor = user_height / hauteur_px if hauteur_px else 1.0
        method = "Taille utilisateur"
    else:
        # Même si A4 détectée, permettre d'ajuster la taille
        hauteur_px = np.linalg.norm(p_neck - p_ankle) if p_neck is not None and p_ankle is not None else None
        user_height = st.number_input("Ajuster votre taille réelle (cm)", min_value=100.0, max_value=250.0, value=170.0)
        if hauteur_px:
            scale_factor = user_height / hauteur_px
            method = "Taille utilisateur (ajustée)"

    st.subheader("Dimensions finales et squelette 3D")
    st.write(f"Méthode de calibration utilisée : {method} → {scale_factor:.4f} cm/pixel")

    # 3. Reconstruction 3D et affichage des mesures finales
    points_3d = simple_3d_triangulation(points_face, points_profile)
    metrics = calculate_tailor_body_metrics(points_3d, scale_factor)
    metrics["Taille t-shirt"] = estimate_tshirt_size(metrics.get("Tour de poitrine"))
    st.write(metrics)

    # 4. Affichage du squelette 3D uniquement
    valid_pts = [p for p in points_3d if p is not None]
    if valid_pts:
        pts = np.array(valid_pts)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="red", s=60)
        ax.set_title("Squelette 3D")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        st.pyplot(fig)