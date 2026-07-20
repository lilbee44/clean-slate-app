# =============================================================
#  slate. — CleanSlate AI  (v2.2)
#  UI "maquette" : fond clair, mobile-first, cartes, anneau de progression
#  Moteur IA : OpenCV (YuNet + SFace) — 100% local
# =============================================================

import io
import os
import zipfile
import urllib.request
import numpy as np
import streamlit as st
import requests
import cv2
from PIL import Image, ImageFilter, ImageDraw

# =============================================================
# CONFIGURATION & STYLE (design maquette)
# =============================================================
st.set_page_config(page_title="slate. — CleanSlate AI", page_icon="✨", layout="centered")

st.markdown("""
<style>
/* ---- Fond clair dégradé crème -> lavande (comme la maquette) ---- */
.stApp {
  background: linear-gradient(165deg, #fbf7f1 0%, #f7f5fc 45%, #e9e9f8 100%) !important;
  color: #1f2937 !important;
  font-family: -apple-system, 'Inter', 'Segoe UI', sans-serif;
}
.block-container { max-width: 680px !important; padding-top: 2.2rem !important; }

/* ---- Typo ---- */
h1, h2, h3, h4, p, li, label, span { color: #1f2937 !important; }
.logo-slate {
  font-size: 2.1rem; font-weight: 800; letter-spacing: -1px;
  color: #1f2937 !important; text-align: center; margin-bottom: .2rem;
}
.logo-slate .dot { color: #4f6df5 !important; }
.hero-title { font-size: 1.9rem; font-weight: 800; text-align: center; margin: .4rem 0 .6rem; }
.hero-sub { text-align: center; color: #6b7280 !important; font-size: 1.02rem; line-height: 1.5; }
.hero-phoenix { font-size: 4.6rem; text-align: center; margin: 1rem 0 .4rem; }
.secure-note { text-align:center; color:#6b7280 !important; font-size:.85rem; margin-top:.7rem; }
.step-label { text-align:center; letter-spacing:2px; font-size:.8rem; color:#9ca3af !important; font-weight:600; }

/* ---- Cartes blanches ---- */
div[data-testid="stFileUploader"] section {
  background: #ffffff !important; border: 1px solid #e7e7f0 !important;
  border-radius: 16px !important; box-shadow: 0 2px 10px rgba(31,41,55,.05) !important;
}
div[data-testid="stFileUploader"] section * { color:#374151 !important; }
div[data-testid="stExpander"] {
  background:#ffffff !important; border:1px solid #e7e7f0 !important;
  border-radius:16px !important; box-shadow:0 2px 10px rgba(31,41,55,.05) !important;
}

/* ---- Radios en cartes (parcours & actions) ---- */
div[role="radiogroup"] > label {
  background:#ffffff !important; border:1px solid #e7e7f0 !important;
  border-radius:14px !important; padding:.85rem 1rem !important;
  margin-bottom:.55rem !important; width:100%;
  box-shadow:0 2px 8px rgba(31,41,55,.05);
  transition: border-color .15s ease;
}
div[role="radiogroup"] > label:hover { border-color:#4f6df5 !important; }

/* ---- Boutons bleus arrondis (comme "Connecter ma galerie") ---- */
.stButton>button, .stDownloadButton>button {
  background: linear-gradient(90deg, #3b5bfd, #4f6df5) !important;
  color:#ffffff !important; font-weight:600 !important; border:none !important;
  border-radius:999px !important; padding:.72rem 1.5rem !important;
  box-shadow:0 6px 16px rgba(59,91,253,.28) !important; width:100%;
  transition: transform .15s ease !important;
}
.stButton>button:hover { transform: translateY(-2px) !important; }
.stButton>button * , .stDownloadButton>button * { color:#ffffff !important; }

/* ---- Anneau de progression 'Diagnostic' ---- */
.ring { width:220px; height:220px; border-radius:50%; margin:1rem auto;
        display:flex; align-items:center; justify-content:center; }
.ring-inner { width:176px; height:176px; background:#ffffff; border-radius:50%;
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        box-shadow: inset 0 2px 8px rgba(31,41,55,.06); }
.ring-num { font-size:3.2rem; font-weight:800; color:#1f2937 !important; line-height:1; }
.ring-label { font-size:.85rem; color:#6b7280 !important; margin-top:.3rem; }

/* ---- Divers ---- */
div[data-testid="stMetricValue"] { color:#1f2937 !important; }
div[data-testid="stMetricLabel"] * { color:#6b7280 !important; }
.stCaption, small { color:#6b7280 !important; }
#MainMenu, footer, header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================
# MOTEUR IA — OpenCV YuNet (détection) + SFace (reconnaissance)
# =============================================================
MODELES = {
    "yunet": ("face_detection_yunet_2023mar.onnx",
              "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    "sface": ("face_recognition_sface_2021dec.onnx",
              "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
}

@st.cache_resource(show_spinner="🧠 Préparation du moteur IA (première fois uniquement)…")
def charger_moteur():
    os.makedirs("models", exist_ok=True)
    chemins = {}
    for cle, (fichier, url) in MODELES.items():
        chemin = os.path.join("models", fichier)
        if not os.path.exists(chemin):
            urllib.request.urlretrieve(url, chemin)
        chemins[cle] = chemin
    detecteur = cv2.FaceDetectorYN_create(chemins["yunet"], "", (320, 320), 0.7, 0.3, 5000)
    reconnaisseur = cv2.FaceRecognizerSF_create(chemins["sface"], "")
    return detecteur, reconnaisseur


def _visages(img_pil):
    detecteur, _ = charger_moteur()
    bgr = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    detecteur.setInputSize((w, h))
    _, faces = detecteur.detect(bgr)
    if faces is None:
        faces = np.empty((0, 15), dtype=np.float32)
    return bgr, faces


def encodage_reference(img_pil):
    _, reconnaisseur = charger_moteur()
    bgr, faces = _visages(img_pil)
    if len(faces) == 0:
        return None
    visage = max(faces, key=lambda f: f[2] * f[3])
    aligne = reconnaisseur.alignCrop(bgr, visage)
    return reconnaisseur.feature(aligne)


def detecter_cible(img_pil, enc_cible, seuil):
    _, reconnaisseur = charger_moteur()
    bgr, faces = _visages(img_pil)
    H, W = bgr.shape[:2]
    boxes = []
    for f in faces:
        aligne = reconnaisseur.alignCrop(bgr, f)
        empreinte = reconnaisseur.feature(aligne)
        score = reconnaisseur.match(empreinte, enc_cible, cv2.FaceRecognizerSF_FR_COSINE)
        if score >= seuil:
            x, y, w, h = (int(v) for v in f[:4])
            boxes.append((max(0, y), min(W, x + w), min(H, y + h), max(0, x)))
    return boxes


# =============================================================
# TRANSFORMATIONS D'IMAGE
# =============================================================
def boite_corps(box, size):
    top, right, bottom, left = box
    h, w = bottom - top, right - left
    W, H = size
    return (max(0, left - w), max(0, top - int(h * 0.4)),
            min(W, right + w), min(H, bottom + int(h * 5)))


def generer_masque(img_pil, boxes):
    masque = Image.new("L", img_pil.size, 0)
    draw = ImageDraw.Draw(masque)
    for box in boxes:
        draw.rectangle(boite_corps(box, img_pil.size), fill=255)
    return masque.filter(ImageFilter.GaussianBlur(radius=15))


def flou_cinematique(img_pil, boxes):
    out = img_pil.copy()
    for box in boxes:
        l, t, r, b = boite_corps(box, img_pil.size)
        region = out.crop((l, t, r, b)).filter(ImageFilter.GaussianBlur(radius=30))
        out.paste(region, (l, t, r, b))
    return out


def sticker_emoji(img_pil, boxes):
    out = img_pil.copy()
    draw = ImageDraw.Draw(out)
    for (top, right, bottom, left) in boxes:
        w = right - left
        pad = int(w * 0.25)
        l, t, r, b = left - pad, top - pad, right + pad, bottom + pad
        draw.ellipse([l, t, r, b], fill="#FFD34D", outline="#B8860B", width=3)
        ew = max(1, (r - l) // 8)
        cy = t + (b - t) // 3
        draw.ellipse([l + 2*ew, cy, l + 3*ew, cy + ew], fill="#1f2937")
        draw.ellipse([r - 3*ew, cy, r - 2*ew, cy + ew], fill="#1f2937")
        draw.arc([l + 2*ew, t + (b-t)//2, r - 2*ew, b - ew], 20, 160, fill="#1f2937", width=max(3, ew//2))
    return out


def inpainting_stability(img_pil, masque, api_key, prompt):
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
# ÉTAT DE SESSION
# =============================================================
if "etape" not in st.session_state:
    st.session_state.etape = "accueil"
if "scan" not in st.session_state:
    st.session_state.scan = None
if "target_encoding" not in st.session_state:
    st.session_state.target_encoding = None


def logo():
    st.markdown("<div class='logo-slate'>slate<span class='dot'>.</span></div>", unsafe_allow_html=True)


# =============================================================
# ÉCRAN 1 — ONBOARDING (maquette : phénix + Connecter ma galerie)
# =============================================================
if st.session_state.etape == "accueil":
    st.markdown("<div class='step-label'>ONBOARDING</div>", unsafe_allow_html=True)
    logo()
    st.markdown("<div class='hero-phoenix'>🐦‍🔥</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>Project Slate.</div>", unsafe_allow_html=True)
    st.markdown("<p class='hero-sub'>Prêt(e) à tourner la page ?<br>"
                "L'IA s'occupe du tri numérique, à votre rythme.</p>", unsafe_allow_html=True)
    st.write("")
    if st.button("Connecter ma galerie"):
        st.session_state.etape = "casting"
        st.rerun()
    st.markdown("<p class='secure-note'>🔒 <b>Totalement sécurisé.</b> L'IA travaille 100% en local.</p>",
                unsafe_allow_html=True)

# =============================================================
# ÉCRAN 2 — LE CASTING + GALERIE + SCAN
# =============================================================
else:
    st.markdown("<div class='step-label'>IDENTIFICATION</div>", unsafe_allow_html=True)
    logo()

    st.write("### 🧭 De qui souhaitez-vous vous détacher aujourd'hui ?")
    type_rupture = st.radio(
        "Parcours",
        ["💔 Une relation amoureuse (Parcours Nouvelle Page)",
         "🛑 Une relation amicale (Parcours Tri Sélectif)"],
        index=0, label_visibility="collapsed",
    )

    if "amoureuse" in type_rupture:
        titre_casting = "'Le Casting' — Qui est votre ex-partenaire ?"
        message_succes = "🎉 Bravo. Vous venez de tourner la page — prenez soin de votre cœur."
        prompt_defaut = "empty scenic background, natural continuation, realistic photo, high resolution"
    else:
        titre_casting = "'Le Casting' — Qui est l'ancien(ne) ami(e) ?"
        message_succes = "🎉 Limites posées. Place aux relations saines et équilibrées."
        prompt_defaut = "group of friends smiling, seamless background continuation, realistic photo, high resolution"

    st.write(f"### 📸 {titre_casting}")
    fichier_ref = st.file_uploader("Une photo claire de la personne, seule de préférence",
                                   type=["jpg", "jpeg", "png"])

    st.write("### 📂 Votre galerie à analyser")
    fichiers_galerie = st.file_uploader("Glissez-déposez les photos à trier / nettoyer",
                                        type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    with st.expander("🎛️ Réglages"):
        seuil = st.slider("Seuil de correspondance (plus haut = plus strict)", 0.25, 0.50, 0.36, 0.01)
        st.caption("L'IA rate des photos ? Baissez le seuil. Elle détecte d'autres personnes ? Montez-le.")

    if fichier_ref and fichiers_galerie:
        if st.button("🚀 Lancer le scan"):
            img_ref = Image.open(fichier_ref).convert("RGB")
            enc = encodage_reference(img_ref)
            if enc is None:
                st.error("Aucun visage détecté sur la photo de référence. Essayez une photo plus nette, de face.")
            else:
                st.session_state.target_encoding = enc
                resultats = []
                barre = st.progress(0, text="Analyse en cours…")
                for i, f in enumerate(fichiers_galerie):
                    img = Image.open(f).convert("RGB")
                    boxes = detecter_cible(img, enc, seuil)
                    if boxes:
                        resultats.append({"nom": f.name, "image": img, "boxes": boxes, "garder": True})
                    pct = int((i + 1) / len(fichiers_galerie) * 100)
                    barre.progress((i + 1) / len(fichiers_galerie),
                                   text=f"🧠 {pct}% de ton espace mental libéré…")
                barre.empty()
                st.session_state.scan = {"total": len(fichiers_galerie), "resultats": resultats}
                if resultats:
                    st.balloons()

    # =========================================================
    # ÉCRAN 3 — 'DIAGNOSTIC' (anneau + cartes d'action)
    # =========================================================
    if st.session_state.scan:
        scan = st.session_state.scan
        n = len(scan["resultats"])
        total = max(1, scan["total"])
        pct = int(n / total * 100)

        st.divider()
        st.markdown("<div class='step-label'>TABLEAU DE BORD</div>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center'>'Diagnostic'</h3>", unsafe_allow_html=True)

        ring = ("<div class='ring' style='background:conic-gradient(#6366f1 0% " + str(max(6, pct)) +
                "%, #a855f7 " + str(max(6, pct)) + "% " + str(max(8, pct + 2)) +
                "%, #e8e8f3 " + str(max(8, pct + 2)) + "% 100%)'>"
                "<div class='ring-inner'><div class='ring-num'>" + str(n) + "</div>"
                "<div class='ring-label'>Photos identifiées</div></div></div>")
        st.markdown(ring, unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;color:#6b7280 !important'>sur {scan['total']} photos analysées</p>",
                    unsafe_allow_html=True)

        if n == 0:
            st.info("Aucune photo contenant la cible n'a été détectée. Essayez de baisser le seuil dans les Réglages.")
        else:
            with st.expander("🖼️ Voir et sélectionner les photos détectées", expanded=False):
                cols = st.columns(3)
                for i, r in enumerate(scan["resultats"]):
                    with cols[i % 3]:
                        st.image(r["image"], use_container_width=True)
                        r["garder"] = st.checkbox(r["nom"], value=r["garder"], key=f"chk_{i}")
            selection = [r for r in scan["resultats"] if r["garder"]]
            st.markdown(f"<p style='text-align:center'><b>{len(selection)} photo(s) sélectionnée(s)</b></p>",
                        unsafe_allow_html=True)

            action = st.radio("Action", [
                "🗃️ Quarantaine — Archiver loin des yeux",
                "🪄 Remplacer l'Ex par IA — Sticker, flou, ou décor",
                "🗑️ Le Grand Saut — Supprimer définitivement",
            ], label_visibility="collapsed")

            sous_mode, api_key, prompt_inpainting = None, None, prompt_defaut
            if "Remplacer" in action:
                sous_mode = st.radio("Style de remplacement",
                                     ["🌫️ Flou cinématique (gratuit)", "🙂 Sticker emoji (gratuit)",
                                      "✨ Effacer & recréer le décor (Stability AI)"],
                                     horizontal=False)
                if "Stability" in sous_mode:
                    api_key = st.text_input("🔑 Clé API Stability AI", type="password", placeholder="sk-...")
                    prompt_inpainting = st.text_input("Prompt de remplacement", value=prompt_defaut)
                    st.caption("💡 Idées virales : *'a cute golden retriever'* ou *'a fluffy llama'*.")

            confirme = True
            if "Grand Saut" in action:
                st.warning("⚠️ Le Grand Saut retire ces photos de la session, définitivement. "
                           "Pensez à les supprimer ensuite de votre galerie.")
                confirme = st.checkbox("Je comprends que cette action est irréversible.")

            if selection and st.button("🚀 Lancer la libération numérique", disabled=not confirme):

                if "Quarantaine" in action:
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w") as z:
                        for r in selection:
                            b = io.BytesIO(); r["image"].save(b, format="JPEG")
                            z.writestr(r["nom"], b.getvalue())
                    st.success(f"{message_succes} ({len(selection)} photos mises en quarantaine)")
                    st.download_button("⬇️ Télécharger la quarantaine (.zip)", buf.getvalue(),
                                       "quarantaine_slate.zip", "application/zip")
                    st.caption("Conseil : confiez ce zip à un(e) 'Gardien(ne)' de confiance, avec un verrou "
                               "temporel (ex : ne pas rouvrir avant 3 mois).")

                elif "Grand Saut" in action:
                    noms = [r["nom"] for r in selection]
                    scan["resultats"] = [r for r in scan["resultats"] if r["nom"] not in noms]
                    st.balloons()
                    st.success(f"{message_succes} ({len(noms)} photos supprimées de la session)")
                    st.caption("Dans l'app mobile, ces photos seront envoyées dans une corbeille 30 jours "
                               "avant suppression définitive de la galerie.")

                else:
                    traitees = 0
                    barre = st.progress(0)
                    for i, r in enumerate(selection):
                        try:
                            if "Flou" in sous_mode:
                                img_finale = flou_cinematique(r["image"], r["boxes"])
                            elif "emoji" in sous_mode:
                                img_finale = sticker_emoji(r["image"], r["boxes"])
                            else:
                                if not api_key:
                                    st.error("Clé API Stability requise pour l'inpainting.")
                                    break
                                masque = generer_masque(r["image"], r["boxes"])
                                img_finale = inpainting_stability(r["image"], masque, api_key, prompt_inpainting)

                            a, b = st.columns(2)
                            a.image(r["image"], caption=f"Avant — {r['nom']}", use_container_width=True)
                            b.image(img_finale, caption="Après ✨", use_container_width=True)
                            buf = io.BytesIO(); img_finale.save(buf, format="JPEG")
                            st.download_button(f"⬇️ Télécharger {r['nom']}", buf.getvalue(),
                                               f"cleanslate_{r['nom']}", "image/jpeg", key=f"dl_{i}")
                            traitees += 1
                        except Exception as e:
                            st.warning(f"Erreur sur {r['nom']} : {e}")
                        barre.progress((i + 1) / len(selection))

                    if traitees:
                        st.balloons()
                        st.success(f"{message_succes} ({traitees} photos traitées)")

    # ---- Pied de page ----
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 Réinitialiser la session"):
            st.session_state.scan = None
            st.session_state.target_encoding = None
            st.session_state.etape = "accueil"
            st.rerun()
    with col_b:
        st.caption("🔒 RGPD : seuls les pixels affichés et l'empreinte de la cible existent le temps de la session. "
                   "Aucun visage tiers n'est mémorisé.")
