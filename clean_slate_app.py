import os
import io
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
import face_recognition
from stability_sdk import client

# --- CONFIGURATION ET STYLE GRAPHIQUE PREMIUM ---
st.set_page_config(page_title="CleanSlate AI", page_icon="✨", layout="centered")

# Injection de CSS sur-mesure pour transformer l'interface brute en App Premium
st.markdown("""
    <style>
    /* Fond de l'application en dégradé Midnight Blue & Indigo */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1b4b 100%) !important;
        color: #f3f4f6 !important;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Titres stylisés */
    h1 {
        font-weight: 800 !important;
        letter-spacing: -1px !important;
        background: linear-gradient(90deg, #6366f1, #a855f7) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        padding-bottom: 0px !important;
    }
    
    .subtitle-app {
        color: #9ca3af !important;
        font-size: 1.15rem !important;
        margin-bottom: 2rem !important;
    }
    
    /* Cartes au design épuré (Glassmorphism léger) */
    div[data-testid="stVerticalBlock"] > div {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Input Boxes et Sélecteurs */
    input, select, textarea, .stFileUploader {
        background-color: rgba(17, 24, 39, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #f3f4f6 !important;
    }
    
    /* Bouton d'action principal (Couleur violette électrique de la maquette) */
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5, #6366f1) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.6) !important;
    }
    
    /* Cacher les éléments inutiles de Streamlit */
    #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- ÉCRAN : TITRE & ONBOARDING ---
st.markdown("<h1>slate.</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-app'>Prêt(e) à tourner la page ? L'IA s'occupe du tri numérique, à votre rythme.</p>", unsafe_allow_html=True)

st.write("### 🧭 Choisissez votre parcours")
type_rupture = st.radio(
    "Quel type de relation souhaitez-vous archiver ?",
    ["💔 Une relation amoureuse (Parcours Nouvelle Page)", "🛑 Une relation amicale (Parcours Tri Sélectif)"],
    index=0,
    label_visibility="collapsed"
)

# Adaptation dynamique des textes
if "amoureuse" in type_rupture:
    titre_etape_1 = "📸 Qui est votre ex-partenaire ?"
    placeholder_upload = "Déposez une photo claire de votre ex..."
    message_succes = "🎉 L'IA a isolé les souvenirs de cette relation. Prenez soin de votre cœur."
    prompt_inpainting = "A beautiful empty background, flawless scenery, realistic photo, high resolution"
else:
    titre_etape_1 = "📸 Qui est l'ancien(ne) ami(e) ?"
    placeholder_upload = "Déposez une photo claire de cette personne..."
    message_succes = "🎉 Vous avez posé vos limites. Place aux relations saines et équilibrées."
    prompt_inpainting = "A beautiful group of friends smiling, background continuation, perfect restoration, high resolution"

# --- CONFIGURATION TECHNIQUE ---
st.write("### 🔑 Clé d'accès IA")
api_key = st.text_input("Saisissez votre clé API Stability AI :", type="password", placeholder="sk-...", label_visibility="collapsed")

st.write("### 📂 Vos photos à analyser")
fichiers_galerie = st.file_uploader(
    "Glissez-déposez les photos de votre galerie à trier/nettoyer :", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

st.write(f"### {titre_etape_1}")
fichier_ref = st.file_uploader(placeholder_upload, type=["jpg", "jpeg", "png"])

# --- FONCTION DE SEGMENTATION FLUIDE ---
def generer_masque_pillow(img_pil, enc_cible, tolerance=0.6):
    image_np = np.array(img_pil)
    locations = face_recognition.face_locations(image_np)
    encodings = face_recognition.face_encodings(image_np, locations)
    
    masque = Image.new("L", img_pil.size, 0)
    cible_trouvee = False
    
    for (top, right, bottom, left), enc_visage in zip(locations, encodings):
        match = face_recognition.compare_faces([enc_cible], enc_visage, tolerance=tolerance)
        
        if match[0]:
            cible_trouvee = True
            hauteur = bottom - top
            largeur = right - left
            
            top_c = max(0, top - int(hauteur * 0.3))
            bottom_c = min(img_pil.size[1], bottom + int(hauteur * 5.0))
            left_c = max(0, left - int(largeur * 1.0))
            right_c = min(img_pil.size[0], right + int(largeur * 1.0))
            
            from PIL import ImageDraw
            draw = ImageDraw.Draw(masque)
            draw.rectangle([left_c, top_c, right_c, bottom_c], fill=255)
                
    masque_floute = masque.filter(ImageFilter.GaussianBlur(radius=15))
    return cible_trouvee, masque_floute

# --- PIPELINE PRINCIPAL DE TRAITEMENT ---
if fichier_ref and api_key and fichiers_galerie:
    st.write("### 🎛️ Options de traitement")
    tolerance = st.slider("Sensibilité de l'IA (Plus bas = plus strict)", 0.4, 0.7, 0.6, 0.05)
    action_ia = st.selectbox("Action souhaitée :", ["✨ Effacer et recréer par IA (Inpainting)", "🗃️ Isoler les photos"])
    
    if st.button("🚀 Lancer la libération numérique"):
        stability_api = client.StabilityInference(key=api_key, engine="stable-diffusion-xl-1024-v1-0")
        
        img_ref_rec = face_recognition.load_image_file(fichier_ref)
        encodings_ref = face_recognition.face_encodings(img_ref_rec)
        
        if not encodings_ref:
            st.error("Aucun visage détecté sur la photo de référence.")
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
                        st.image(img_init_pil, width=300)
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
                                            label=f"⬇️ Télécharger {fichier.name}",
                                            data=byte_im,
                                            file_name=f"cleanslate_{fichier.name}",
                                            mime="image/jpeg"
                                        )
                                        compteur_modifie += 1
                        except Exception as e:
                            st.warning(f"Erreur d'Inpainting : {e}")
                            
                progress_bar.progress((index + 1) / len(fichiers_galerie))
            
            status_text.empty()
            if compteur_modifie > 0:
                st.success(f"{message_succes} ({compteur_modifie} photos traitées).")
            else:
                st.info("Aucune photo contenant la cible n'a été détectée.")
