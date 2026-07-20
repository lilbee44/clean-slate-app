# slate. — CleanSlate AI (v2.1)

Tri & nettoyage de photos après une rupture amoureuse ou amicale.

## Moteur IA

La reconnaissance faciale utilise **OpenCV (YuNet + SFace)** : aucune
compilation, installation en quelques secondes sur macOS, Windows, Linux
et Streamlit Cloud. Les modèles (~40 Mo) sont téléchargés automatiquement
au premier lancement puis mis en cache.

## Installation

```bash
git clone https://github.com/lilbee44/clean-slate-app
cd clean-slate-app
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run clean_slate_app.py
```

L'app s'ouvre sur http://localhost:8501.

## Clé API (optionnelle)

Seul le mode **✨ Remplacement Invisible** (inpainting) a besoin d'une clé
Stability AI (https://platform.stability.ai → API Keys). Les modes
**Quarantaine**, **Flou** et **Emoji** sont 100 % locaux et gratuits.

## Confidentialité

- La reconnaissance faciale tourne entièrement en local.
- Seule l'empreinte du visage cible est gardée, le temps de la session.
- Les visages tiers ne sont jamais mémorisés.
- « Réinitialiser la session » efface tout.
