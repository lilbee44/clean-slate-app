import os
import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import face_recognition
from stability_sdk import client
import stability_sdk.interfaces.sprites_and_models_pb2 as components

# --- CONFIGURATION DE LA PAGE STREAMLIT ---
st.set_page_config(page_title="CleanSlate AI", page_icon="✨", layout="centered")

# Style CSS minimaliste pour l'ambiance haut de gamme
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #f0f6fc; }
    h1, h2, h3 { color: #8892b0; }
    .stButton>button { background-color: #4f46e5; color: white; border-radius: 8px; width: 100%; }
    </style>
""", unsafe_unsafe_with_html=True)

st.title("✨ CleanSlate AI")
st.subheader("L'application de reconstruction et de tri thérapeutique par IA")

# --- ÉTAPE D'ONBOARDING : CHOIX DE LA CIBLE ---
type_rupture = st.radio(
    "De qui souhaitez-vous vous détacher aujourd'hui ?",
    ["💔 Une relation amoureuse (Parcours Nouvelle Page)", "🛑 Une relation amicale (Parcours Tri Sélectif)"],
    index=0
)

# Adaptation dynamique de l'expérience utilisateur (UX)
if "amoureuse" in type_rupture:
    titre_etape_1 = "📸 Qui est votre ex-partenaire ?"
    placeholder_upload = "Importez une photo claire de votre ex..."
    message_succes = "🎉 L'IA a isolé les souvenirs de cette relation. Prenez soin de votre cœur."
    prompt_inpainting = "A beautiful empty background, flawless scenery, realistic photo"
else:
    titre_etape_1 = "📸 Qui est l'ancien(ne) ami(e) ?"
    placeholder_upload = "Importez une photo claire de cette personne..."
    message_succes = "🎉 Vous avez posé vos limites. Place aux relations saines et équilibrées."
    prompt_inpainting = "A beautiful group of friends smiling, group of people hanging out, background continuation, perfect restoration"

# --- CONFIGURATION TECHNIQUE ---
st.write("---")
st.write("### 🔑 Configuration des accès")
api_key = st.text_input("Clé API Stability AI :", type="password", placeholder="sk-...")
dossier_galerie = st.text_input("Chemin de votre dossier Galerie locale :", "./ma_galerie_photos")
dossier_quarantaine = "./quarantaine_clean_slate"

st.write("---")
st.write(f"### {titre_etape_1}")
fichier_ref = st.file_uploader(placeholder_upload, type=["jpg", "jpeg", "png"])

# --- FONCTION DE SEGMENTATION AVANCÉE ---
def generer_masque_silhouette(image_bgr, enc_cible, tolerance=0.6):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(image_rgb)
    encodings = face_recognition.face_encodings(image_rgb, locations)
    
    masque = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cible_trouvee = False
    
    for (top, right, bottom, left), enc_visage in zip(locations, encodings):
        match = face_recognition.compare_faces([enc_cible], enc_visage, tolerance=tolerance)
        
        if match[0]:
            cible_trouvee = True
            hauteur = bottom - top
            largeur = right - left
            
            # Extension de la boîte pour capturer le corps/silhouette de la personne
            top_c = max(0, top - int(hauteur * 0.2))
            bottom_c = min(image_bgr.shape[0], bottom + int(hauteur * 4.5))
            left_c = max(0, left - int(largeur * 0.8))
            right_c = min(image_bgr.shape[1], right + int(largeur * 0.8))
            
            # Application de GrabCut pour isoler la silhouette proprement
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            rect = (left_c, top_c, right_c - left_c, bottom_c - top_c)
            
            masque_local = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
            try:
                cv2.grabCut(image_bgr, masque_local, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                masque_silhouette = np.where((masque_local == 2) | (masque_local == 0), 0, 255).astype('uint8')
                masque = cv2.bitwise_or(masque, masque_silhouette)
            except:
                # Fallback si GrabCut échoue sur les bords
                cv2.rectangle(masque, (left_c, top_c), (right_c, bottom_c), 255, -1)
                
    masque_floute = cv2.GaussianBlur(masque, (21, 21), 0)
    return cible_trouvee, masque_floute

# --- PIPELINE PRINCIPAL DE TRAITEMENT ---
if fichier_ref and api_key:
    # Affichage de la cible
    img_ref_pil = Image.open(fichier_ref)
    st.image(img_ref_pil, caption="Visage cible à effacer de votre vie", width=150)
    
    # Paramètres algorithmiques
    tolerance = st.slider("Sensibilité de l'IA (Ajuster si l'ex ressemble à quelqu'un d'autre)", 0.4, 0.7, 0.6, 0.05)
    action_ia = st.selectbox("Action souhaitée sur les souvenirs :", ["✨ Effacer et recréer par IA (Inpainting)", "🗃️ Mettre en quarantaine (Déplacement simple)"])
    
    if st.button("🚀 Lancer la libération numérique"):
        if not os.path.exists(dossier_galerie):
            st.error(f"Le dossier spécifié '{dossier_galerie}' est introuvable. Veuillez le créer à la racine.")
        else:
            os.makedirs(dossier_quarantaine, exist_ok=True)
            
            # Initialisation du client de génération Stability AI
            stability_api = client.StabilityInference(key=api_key, engine="stable-diffusion-xl-1024-v1-0")
            
            # Étape A : Encodage facial de la cible
            img_ref_rec = face_recognition.load_image_file(fichier_ref)
            encodings_ref = face_recognition.face_encodings(img_ref_rec)
            
            if not encodings_ref:
                st.error("Aucun visage détecté sur la photo de référence. Veuillez utiliser un portrait bien éclairé.")
            else:
                cible_encoding = encodings_ref[0]
                fichiers = [f for f in os.listdir(dossier_galerie) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                compteur_modifie = 0
                
                # Étape B : Itération sur la galerie de photos
                for index, nom_fichier in enumerate(fichiers):
                    chemin_source = os.path.join(dossier_galerie, nom_fichier)
                    status_text.text(f"Analyse clinique de {nom_fichier}...")
                    
                    image_bgr = cv2.imread(chemin_source)
                    trouve, masque_ex = generer_masque_silhouette(image_bgr, cible_encoding, tolerance)
                    
                    if trouve:
                        chemin_destination = os.path.join(dossier_quarantaine, nom_fichier)
                        
                        if action_ia == "🗃️ Mettre en quarantaine (Déplacement simple)":
                            shutil.move(chemin_source, chemin_destination)
                            compteur_modifie += 1
                        else:
                            # Mode Inpainting IA
                            img_init_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                            img_mask_pil = Image.fromarray(masque_ex)
                            
                            try:
                                answers = stability_api.generate(
                                    prompt=prompt_inpainting,
                                    init_image=img_init_pil,
                                    mask_image=img_mask_pil,
                                    start_schedule=1.0,
                                    steps=30,
                                    cfg_scale=7.0
                                )
                                
                                for resp in answers:
                                    for artifact in resp.artifacts:
                                        if artifact.type == components.ARTIFACT_IMAGE:
                                            img_finale = Image.open(io.BytesIO(artifact.binary))
                                            # On remplace l'image d'origine par la version épurée
                                            img_finale.save(chemin_source)
                                            compteur_modifie += 1
                            except Exception as e:
                                st.warning(f"Erreur d'Inpainting sur {nom_fichier} : {e}")
                                
                    progress_bar.progress((index + 1) / len(fichiers))
                
                status_text.empty()
                st.success(f"{message_succes} ({compteur_modifie} fichiers traités avec succès).")
                
st.sidebar.markdown("""
### 🛠️ Mode d'emploi local
1. Créez un dossier nommé `ma_galerie_photos` à côté de ce script.
2. Ajoutez-y vos fichiers images.
3. Obtenez une clé API sur le site de [Stability AI](https://platform.stability.ai).
4. Saisissez les données requises au centre pour démarrer la détection.
""")
