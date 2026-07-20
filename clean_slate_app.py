# =============================================================
#  slate. — CleanSlate AI  (v2)
#  Tri & nettoyage de photos après une rupture amoureuse ou amicale
#  Streamlit + face_recognition (100% local) + Stability AI (inpainting cloud, optionnel)
# =============================================================

import io
import zipfile
import numpy as np
import streamlit as st
import requests
from PIL import Image, ImageFilter, ImageDraw

# --- Import protégé : face_recognition (dlib) est la dépendance délicate ---
try:
    import face_recognition
    FACE_OK = True
except ImportError:
    FACE_OK = False

# =============================================================
# CONFIGURATION & STYLE
# =============================================================
st.set_page_config(page_title="slate. — CleanSlate AI", page_icon="✨", layout="centered")

st.markdown("""
<style>
.stApp {
  background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1b4b 100%) !important;
  color: #f3f4f6 !important;
  font-family: 'Inter', -apple-system, sans-serif;
}
h1 {
  font-weight: 800 !important; letter-spacing: -1px !important;
  background: linear-gradient(90deg, #6366f1, #a855f7) !important;
  -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
}
.subtitle-app { color: #9ca3af !important; font-size: 1.1rem !important; margin-bottom: 1.5rem !important; }
input, select, textarea, .stFileUploader {
  background-color: rgba(17,24,39,.7) !important;
  border: 1px solid rgba(255,255,255,.1) !important;
  border-radius: 12px !important; color: #f3f4f6 !important;
}
.stButton>button {
  background: linear-gradient(90deg, #4f46e5, #6366f1) !important;
  color: white !important; font-weight: 600 !important; border: none !important;
  border-radius: 12px !important; padding: .7rem 1.5rem !important;
  box-shadow: 0 4px 14px rgba(79,70,229,.4) !important; width: 100%;
  transition: all .25s ease !important;
}
.stButton>button:hover { transform: translateY(-2px) !important; }
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================
# ÉTAT DE SESSION
# Confidentialité : on ne conserve QUE l'encodage de la cible.
# Les encodages des autres visages ne sont jamais stockés.
# =============================================================
if "scan" not in st.session_state:
    st.session_state.scan = None          # résultats du scan
if "target_encoding" not in st.session_state:
    st.session_state.target_encoding = None
if "corbeille" not in st.session_state:
    st.session_state.corbeille = []       # corbeille temporaire in-app (dernière chance)

# =============================================================
# FONCTIONS
# =============================================================
def detecter_cible(img_pil, enc_cible, tolerance):
    """Retourne la liste des boîtes (top, right, bottom, left) où la cible apparaît."""
    image_np = np.array(img_pil)
    locations = face_recognition.face_locations(image_np)
    if not locations:
        return []
    encodings = face_recognition.face_encodings(image_np, locations)
    boxes = []
    for loc, enc in zip(locations, encodings):
        if face_recognition.compare_faces([enc_cible], enc, tolerance=tolerance)[0]:
            boxes.append(loc)
    # Les encodages des visages tiers sortent de portée ici : rien n'est conservé.
    return boxes


def boite_corps(box, size):
    """Étend la boîte du visage vers le corps (pour masque / flou plein pied)."""
    top, right, bottom, left = box
    h, w = bottom - top, right - left
    W, H = size
    return (max(0, left - w),
            max(0, top - int(h * 0.4)),
            min(W, right + w),
            min(H, bottom + int(h * 5)))


def generer_masque(img_pil, boxes):
    """Masque blanc (zone à effacer) sur fond noir, bords adoucis."""
    masque = Image.new("L", img_pil.size, 0)
    draw = ImageDraw.Draw(masque)
    for box in boxes:
        draw.rectangle(boite_corps(box, img_pil.size), fill=255)
    return masque.filter(ImageFilter.GaussianBlur(radius=15))


def flou_cinematique(img_pil, boxes):
    """Flou bokeh progressif uniquement sur le visage/corps de la cible (local, gratuit)."""
    out = img_pil.copy()
    for box in boxes:
        l, t, r, b = boite_corps(box, img_pil.size)
        region = out.crop((l, t, r, b)).filter(ImageFilter.GaussianBlur(radius=30))
        out.paste(region, (l, t, r, b))
    return out


def sticker_emoji(img_pil, boxes):
    """Remplace le visage par un smiley dessiné (local, gratuit, effet parodique)."""
    out = img_pil.copy()
    draw = ImageDraw.Draw(out)
    for (top, right, bottom, left) in boxes:
        w = right - left
        pad = int(w * 0.25)
        l, t, r, b = left - pad, top - pad, right + pad, bottom + pad
        draw.ellipse([l, t, r, b], fill="#FFD34D", outline="#B8860B", width=3)
        ew = (r - l) // 8
        cy = t + (b - t) // 3
        draw.ellipse([l + 2*ew, cy, l + 3*ew, cy + ew], fill="#1f2937")          # œil gauche
        draw.ellipse([r - 3*ew, cy, r - 2*ew, cy + ew], fill="#1f2937")          # œil droit
        draw.arc([l + 2*ew, t + (b-t)//2, r - 2*ew, b - ew], 20, 160, fill="#1f2937", width=max(3, ew//2))
    return out


def inpainting_stability(img_pil, masque, api_key, prompt):
    """Efface la cible et recrée le décor via l'API REST Stability (v2beta)."""
    # Redimensionner pour rester dans les limites API et maîtriser le coût
    img = img_pil.copy()
    img.thumbnail((1536, 1536))
    msk = masque.resize(img.size)

    buf_img, buf_msk = io.BytesIO(), io.BytesIO()
    img.save(buf_img, format="PNG")
    msk.save(buf_msk, format="PNG")

    r = requests.post(
        "https://api.stability.ai/v2beta/stable-image/edit/inpaint",
        headers={"authorization": f"Bearer {api_key}", "accept": "image/*"},
        files={"image": buf_img.getvalue(), "mask": buf_msk.getvalue()},
        data={"prompt": prompt, "output_format": "jpeg"},
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"API Stability ({r.status_code}) : {r.text[:200]}")
    return Image.open(io.BytesIO(r.content))


# =============================================================
# ÉCRAN 1 — ONBOARDING
# =============================================================
st.markdown("<h1>slate.</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-app'>Prêt(e) à tourner la page ? L'IA s'occupe du tri numérique, à votre rythme.<br>"
            "🔒 <b>La reconnaissance faciale tourne 100% en local.</b> Seul l'inpainting (optionnel) passe par le cloud.</p>",
            unsafe_allow_html=True)

if not FACE_OK:
    st.error("Le module `face_recognition` n'est pas installé. Voir le README pour l'installation de dlib.")
    st.stop()

st.write("### 🧭 De qui souhaitez-vous vous détacher aujourd'hui ?")
type_rupture = st.radio(
    "Parcours",
    ["💔 Une relation amoureuse (Parcours Nouvelle Page)",
     "🛑 Une relation amicale (Parcours Tri Sélectif)"],
    index=0, label_visibility="collapsed",
)

if "amoureuse" in type_rupture:
    titre_casting = "📸 'Le Casting' — Qui est votre ex-partenaire ?"
    message_succes = "🎉 Bravo. Vous venez de tourner la page — prenez soin de votre cœur."
    prompt_defaut = "empty scenic background, natural continuation, realistic photo, high resolution"
else:
    titre_casting = "📸 'Le Casting' — Qui est l'ancien(ne) ami(e) ?"
    message_succes = "🎉 Limites posées. Place aux relations saines et équilibrées."
    prompt_defaut = "group of friends smiling, seamless background continuation, realistic photo, high resolution"

# =============================================================
# ÉCRAN 2 — IDENTIFICATION & GALERIE
# =============================================================
st.write(f"### {titre_casting}")
fichier_ref = st.file_uploader("Une photo claire de la personne, seule de préférence",
                               type=["jpg", "jpeg", "png"])

st.write("### 📂 Votre galerie à analyser")
fichiers_galerie = st.file_uploader("Glissez-déposez les photos à trier / nettoyer",
                                    type=["jpg", "jpeg", "png"], accept_multiple_files=True)

with st.expander("🎛️ Réglages"):
    tolerance = st.slider("Sensibilité de l'IA (plus bas = plus strict)", 0.40, 0.70, 0.60, 0.05)

# =============================================================
# ÉCRAN 3 — SCAN (LA JAUGE DE LIBÉRATION)
# =============================================================
if fichier_ref and fichiers_galerie:
    if st.button("🚀 Lancer le scan"):
        img_ref = face_recognition.load_image_file(fichier_ref)
        encs = face_recognition.face_encodings(img_ref)
        if not encs:
            st.error("Aucun visage détecté sur la photo de référence. Essayez une photo plus nette, de face.")
        else:
            st.session_state.target_encoding = encs[0]
            resultats = []
            barre = st.progress(0, text="Analyse en cours…")
            for i, f in enumerate(fichiers_galerie):
                img = Image.open(f).convert("RGB")
                boxes = detecter_cible(img, st.session_state.target_encoding, tolerance)
                if boxes:
                    resultats.append({"nom": f.name, "image": img, "boxes": boxes, "garder": True})
                pct = int((i + 1) / len(fichiers_galerie) * 100)
                barre.progress((i + 1) / len(fichiers_galerie),
                               text=f"🧠 {pct}% de ton espace mental libéré…")
            barre.empty()
            st.session_state.scan = {"total": len(fichiers_galerie), "resultats": resultats}
            if resultats:
                st.balloons()

# =============================================================
# ÉCRAN 4 — DIAGNOSTIC (TABLEAU DE BORD)
# =============================================================
if st.session_state.scan:
    scan = st.session_state.scan
    n = len(scan["resultats"])

    st.write("## 📊 'Diagnostic'")
    c1, c2 = st.columns(2)
    c1.metric("Photos identifiées", n)
    c2.metric("Photos analysées", scan["total"])

    if n == 0:
        st.info("Aucune photo contenant la cible n'a été détectée. Essayez d'augmenter la sensibilité.")
    else:
        st.write("#### Cochez les photos à traiter (contrôle total) :")
        cols = st.columns(3)
        for i, r in enumerate(scan["resultats"]):
            with cols[i % 3]:
                st.image(r["image"], use_container_width=True)
                r["garder"] = st.checkbox(r["nom"], value=r["garder"], key=f"chk_{i}")

        selection = [r for r in scan["resultats"] if r["garder"]]
        st.write(f"**{len(selection)} photo(s) sélectionnée(s).**")

        # ------------------ CHOIX DE L'ACTION ------------------
        st.write("### ⚡ Action")
        action = st.selectbox("Que fait-on de ces souvenirs ?", [
            "🗃️ Quarantaine — Archiver loin des yeux (ZIP)",
            "🌫️ Flou Cinématique — Bokeh sur la cible (gratuit, local)",
            "🙂 Sticker Emoji — Effet parodique (gratuit, local)",
            "✨ Remplacement Invisible — Effacer & recréer le décor (Stability AI)",
        ])

        api_key, prompt_inpainting = None, prompt_defaut
        if "Stability" in action:
            api_key = st.text_input("🔑 Clé API Stability AI", type="password", placeholder="sk-...")
            prompt_inpainting = st.text_input("Prompt de remplacement (modifiable)", value=prompt_defaut)
            st.caption("💡 Idées virales : remplacez le prompt par *'a cute golden retriever'* ou *'a fluffy llama'*.")

        if selection and st.button("🚀 Lancer la libération numérique"):

            # ---- 1. QUARANTAINE : zip téléchargeable ----
            if "Quarantaine" in action:
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w") as z:
                    for r in selection:
                        b = io.BytesIO(); r["image"].save(b, format="JPEG")
                        z.writestr(r["nom"], b.getvalue())
                st.success(f"{message_succes} ({len(selection)} photos mises en quarantaine)")
                st.download_button("⬇️ Télécharger la quarantaine (.zip)", buf.getvalue(),
                                   "quarantaine_slate.zip", "application/zip")
                st.caption("Conseil : confiez ce zip à un(e) 'Gardien(ne)' de confiance, ou stockez-le hors de vue "
                           "avec un verrou temporel (ex : ne pas rouvrir avant 3 mois).")

            # ---- 2 & 3 & 4. TRANSFORMATIONS avec aperçu avant/après ----
            else:
                traitees = 0
                barre = st.progress(0)
                for i, r in enumerate(selection):
                    try:
                        if "Flou" in action:
                            img_finale = flou_cinematique(r["image"], r["boxes"])
                        elif "Emoji" in action:
                            img_finale = sticker_emoji(r["image"], r["boxes"])
                        else:
                            if not api_key:
                                st.error("Clé API Stability requise pour l'inpainting.")
                                break
                            masque = generer_masque(r["image"], r["boxes"])
                            img_finale = inpainting_stability(r["image"], masque, api_key, prompt_inpainting)

                        # Aperçu Avant / Après (le Choc Visuel)
                        a, b = st.columns(2)
                        a.image(r["image"], caption=f"Avant — {r['nom']}", use_container_width=True)
                        b.image(img_finale, caption="Après ✨", use_container_width=True)

                        buf = io.BytesIO(); img_finale.save(buf, format="JPEG")
                        st.download_button(f"⬇️ Télécharger {r['nom']}", buf.getvalue(),
                                           f"cleanslate_{r['nom']}", "image/jpeg", key=f"dl_{i}")
                        st.session_state.corbeille.append(r["nom"])  # trace 'dernière chance'
                        traitees += 1
                    except Exception as e:
                        st.warning(f"Erreur sur {r['nom']} : {e}")
                    barre.progress((i + 1) / len(selection))

                if traitees:
                    st.balloons()
                    st.success(f"{message_succes} ({traitees} photos traitées)")

# =============================================================
# PIED DE PAGE — CONFIDENTIALITÉ & RESET
# =============================================================
st.divider()
col_a, col_b = st.columns(2)
with col_a:
    if st.button("🧹 Réinitialiser la session"):
        st.session_state.scan = None
        st.session_state.target_encoding = None
        st.session_state.corbeille = []
        st.rerun()
with col_b:
    st.caption("🔒 RGPD : seuls les pixels affichés et l'empreinte de la cible existent le temps de la session. "
               "Aucun visage tiers n'est mémorisé. 'Réinitialiser' efface tout.")
