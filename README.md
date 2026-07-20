# slate. — CleanSlate AI (v2)

Tri & nettoyage de photos après une rupture amoureuse ou amicale.

## Installation

```bash
git clone https://github.com/lilbee44/clean-slate-app
cd clean-slate-app
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

### ⚠️ Le point délicat : dlib (face_recognition)

`face-recognition` compile **dlib** à l'installation. Il faut :

- **macOS** : `brew install cmake` puis `pip install face-recognition`
- **Linux (Debian/Ubuntu)** : `sudo apt install cmake build-essential` puis pip
- **Windows** : le plus simple est `pip install dlib-bin` **avant** `pip install face-recognition`,
  ou passer par conda : `conda install -c conda-forge dlib`

Python 3.10 ou 3.11 recommandé (3.13 pose encore des soucis avec dlib).

## Lancement

```bash
streamlit run clean_slate_app.py
```

L'app s'ouvre sur http://localhost:8501.

## Clé API (optionnelle)

Seul le mode **✨ Remplacement Invisible** (inpainting) a besoin d'une clé
Stability AI (https://platform.stability.ai → API Keys, ~0,03–0,08 $/image
via l'endpoint REST v2beta). Les modes **Quarantaine**, **Flou** et **Emoji**
sont 100 % locaux et gratuits.

## Confidentialité

- La reconnaissance faciale tourne entièrement en local.
- Seule l'empreinte du visage cible est gardée, le temps de la session.
- Les visages tiers ne sont jamais mémorisés (conformité RGPD du prototype).
- « Réinitialiser la session » efface tout.
