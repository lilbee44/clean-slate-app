import os
import io
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
import face_recognition
from stability_sdk import client

# --- CONFIGURATION DE LA PAGE STREAMLIT ---
st.set_page_config(page_title="CleanSlate AI", page_icon="✨", layout="centered")

# Style CSS minimaliste pour l'ambiance haut de gamme
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #f0f6fc; }
    h1, h2, h3 { color: #8892b0; }
    .stButton>button { background-color: #4f46e5; color: white; border-radius: 8px; width: 100%; }
    </style>
""", unsafe_allow_html=True)

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
    prompt_inpainting = "A beautiful empty background, flawless scenery, realistic photo, high resolution"
else:
    titre_etape_1 = "📸 Qui est l'ancien(ne) ami(e) ?"
    placeholder_upload = "Importez une photo claire de cette personne..."
    message_succes = "🎉 Vous avez posé vos limites. Place aux relations saines et équilibrées."
    prompt_inpainting = "A beautiful group of friends smiling, background continuation, perfect restoration, high resolution"

# --- CONFIGURATION TECHNIQUE ---
st.write("---")
st.write("### 🔑 Configuration des accès")
api_key = st.text_input("Clé API Stability AI :", type="password", placeholder="sk-...")

st.write("### 📂 Vos photos à analyser")
fichiers_galerie = st.file_uploader(
    "Glissez-déposez les photos de votre galerie à trier/nettoyer :", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

st.write("---")
st.write(f"### {titre_etape_1}")
fichier_ref = st.file_uploader(placeholder_upload, type=["jpg", "jpeg", "png"])

# --- FONCTION DE SEGMENTATION FLUIDE AVEC PILLOW ---
def generer_masque_pillow(img_pil, enc_cible, tolerance=0.6):
    image_np = np.array(img_pil)
    locations = face_recognition.face_locations(image_np)
    encodings = face_recognition.face_encodings(image_np, locations)
    
    # Créer un masque noir à la taille de l'image
    masque = Image.new("L", img_pil.size, 0)
    cible_trouvee = False
    
    for (top, right, bottom, left), enc_visage in zip(locations, encodings):
        match = face_recognition.compare_faces([enc_cible], enc_visage, tolerance=tolerance)
        
        if match[0]:
            cible_trouvee = True
            hauteur = bottom - top
            largeur = right - left
            
            # Extension pour englober la silhouette de la personne de haut en bas
            top_c = max(0, top - int(hauteur * 0.3))
            bottom_c = min(img_pil.size[1], bottom + int(hauteur * 5.0))
            left_c = max(0, left - int(largeur * 1.0))
            right_c = min(img_pil.size[0], right + int(largeur * 1.0))
            
            # Dessiner un rectangle blanc sur le masque là où se trouve la silhouette
            from PIL import ImageDraw
            draw = ImageDraw.Draw(masque)
            draw.rectangle([left_c, top_c, right_c, bottom_c], fill=255)
                
    # Flouter les contours du masque pour une transition invisible
    masque_floute = masque.filter(ImageFilter.GaussianBlur(radius=15))
    return cible_trouvee, masque_floute

# --- PIPELINE PRINCIPAL DE TRAITEMENT ---
if fichier_ref and api_key and fichiers_galerie:
    img_ref_pil = Image.open(fichier_ref)
    st.image(img_ref_pil, caption="Visage cible à effacer", width=150)
    
    tolerance = st.slider("Sensibilité de l'IA (Plus bas = plus strict)", 0.4, 0.7, 0.6, 0.05)
    action_ia = st.selectbox("Action souhaitée sur les souvenirs :", ["✨ Effacer et recréer par IA (Inpainting)", "🗃️ Isoler les photos"])
    
    if st.button("🚀 Lancer la libération numérique"):
        stability_api = client.StabilityInference(key=api_key, engine="stable-diffusion-xl-1024-v1-0")
        
        img_ref_rec = face_recognition.load_image_file(fichier_ref)
        encodings_ref = face_recognition.face_encodings(img_ref_rec)
        
        if not encodings_ref:
            st.error("Aucun visage détecté sur la photo de référence. Veuillez utiliser un portrait bien éclairé.")
        else:
            cible_encoding = encodings_ref[0]
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            compteur_modifie = 0
            
            for index, fichier in enumerate(fichiers_galerie):
                status_text.text(f"Analyse de {fichier.name}...")
                
                img_init_pil = Image.open(fichier).convert("RGB")
                trouve, masque_ex = generer_masque_pillow(img_init_pil, cible_encoding, tolerance)
                
                if trouve:
                    if action_ia == "🗃️ Isoler les photos":
                        st.warning(f"⚠️ Cible détectée sur : {fichier.name}")
                        st.image(img_init_pil, caption=f"Identifié : {fichier.name}", width=300)
                        compteur_modifie += 1
                    else:
                        try:
                            answers = stability_api.generate(
                                prompt=prompt_inpainting,
                                init_image=img_init_pil,
                                mask_image=masque_ex,
                                steps=30,
                                cfg_scale=7.0
                            )
                            
                            for resp in answers:
                                for artifact in resp.artifacts:
                                    if artifact.type.name == "ARTIFACT_IMAGE":
                                        img_finale = Image.open(io.BytesIO(artifact.binary))
                                        st.image(img_finale, caption=f"✨ {fichier.name} nettoyé !", width=400)
                                        
                                        buf = io.BytesIO()
                                        img_finale.save(buf, format="JPEG")
                                        byte_im = buf.getvalue()
                                        
                                        st.download_button(
                                            label=f"⬇️ Télécharger {fichier.name} nettoyée",
                                            data=byte_im,
                                            file_name=f"cleanslate_{fichier.name}",
                                            mime="image/jpeg"
                                        )
                                        compteur_modifie += 1
                        except Exception as e:
                            st.warning(f"Erreur d'Inpainting sur {fichier.name} : {e}")
                            
                progress_bar.progress((index + 1) / len(fichiers_galerie))
            
            status_text.empty()
            if compteur_modifie > 0:
                st.success(f"{message_succes} ({compteur_modifie} photos traitées).")
            else:
                st.info("Aucune photo contenant la cible n'a été détectée dans les fichiers fournis.")
