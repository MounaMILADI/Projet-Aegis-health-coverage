# 🩺 Aegis Health Coverage – Tarification d'assurance santé

[![Ouvrir l'application](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](LIEN_DE_TON_APPLICATION)

**👉 Application en ligne : [https://projet-aegis-health-coverage-simplon.streamlit.app](LIEN_DE_TON_APPLICATION)**

## Contexte

La direction de l'actuariat d'Aegis Health Coverage souhaite automatiser la tarification de ses contrats d'assurance santé. À partir des données de 1 337 assurés, ce projet prédit les frais médicaux annuels d'un souscripteur selon son profil : âge, IMC, statut fumeur, nombre d'enfants, sexe et région.

## Démarche

1. **Analyse exploratoire** : impact de chaque caractéristique sur les frais médicaux.
2. **Prétraitement** : encodage One-Hot des variables catégorielles et standardisation des variables numériques, sans fuite de données.
3. **Modélisation** : comparaison de la régression linéaire, du KNN, de l'arbre de décision et du Random Forest.
4. **Optimisation** du modèle le plus performant (Random Forest).
5. **Dashboard Streamlit** : analyse visuelle des coûts et simulateur de prime.

## Principaux résultats

- Le **statut fumeur** est le premier facteur de coût, suivi de l'IMC et de l'âge.
- L'obésité n'augmente fortement les frais que **chez les fumeurs**.
- Modèle retenu : **Random Forest optimisé**, avec une erreur moyenne (MAE) de **2 175 $** par assuré.

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `Insurance.ipynb` | Notebook d'analyse et de modélisation |
| `app.py` | Application Streamlit |
| `insurance-data.csv` | Données des assurés |
| `requirements.txt` | Bibliothèques nécessaires |

## Lancer l'application en local

```bash
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
pip install -r requirements.txt
streamlit run app.py
```

## Outils

Python · pandas · scikit-learn · Plotly · Streamlit

---
Projet réalisé en binôme dans le cadre de la formation Data Analyst – Simplon.
