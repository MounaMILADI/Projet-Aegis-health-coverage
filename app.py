# =============================================================================
# Aegis Health Coverage - Dashboard de tarification assurance santé
# Lancement : streamlit run app.py
# Fichier nécessaire dans le même dossier que app.py : insurance-data.csv
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# Configuration de la page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Aegis Health Coverage - Tarification",
                   page_icon="🩺", layout="wide")

# Réduction des marges en haut de page, pour que chaque section tienne sur un écran
# et agrandissement du texte de la barre latérale
st.markdown("""<style>
.block-container {padding-top: 2rem; padding-bottom: 1rem;}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label {font-size: 1.1rem;}
</style>""", unsafe_allow_html=True)

# Couleurs fixes pour le statut fumeur, utilisées dans tous les graphiques
COULEURS_FUMEUR = {"Fumeur": "#E76F51", "Non-fumeur": "#2A9D8F"}

# Traductions pour l'affichage (les valeurs d'origine restent en anglais)
TRAD_SEXE = {"female": "Femme", "male": "Homme"}
TRAD_FUMEUR = {"yes": "Fumeur", "no": "Non-fumeur"}
TRAD_REGION = {"northeast": "Nord-Est", "northwest": "Nord-Ouest",
               "southeast": "Sud-Est", "southwest": "Sud-Ouest"}

# Variables utilisées par le modèle (mêmes colonnes et même ordre que dans le notebook)
VARIABLES_NUM = ["age", "bmi", "children"]
COLONNES_MODELE = ["age", "bmi", "children", "sex_male", "smoker_yes",
                   "region_northwest", "region_southeast", "region_southwest"]


def afficher(fig, hauteur=380):
    """Affiche un graphique Plotly avec une hauteur fixe et des marges réduites."""
    fig.update_layout(height=hauteur, margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig)


def format_dollars(valeur):
    """Formate un montant avec un espace comme séparateur de milliers : 12 345 $."""
    return f"{valeur:,.0f} $".replace(",", " ")


# -----------------------------------------------------------------------------
# Chargement des données (mis en cache pour ne pas relire le fichier à chaque clic)
# -----------------------------------------------------------------------------
@st.cache_data
def charger_donnees():
    # Chargement et suppression du doublon, comme dans le notebook
    df = pd.read_csv(Path(__file__).parent / "insurance-data.csv").drop_duplicates()
    # Colonnes traduites, uniquement pour l'affichage et les filtres
    df["Sexe"] = df["sex"].map(TRAD_SEXE)
    df["Statut fumeur"] = df["smoker"].map(TRAD_FUMEUR)
    df["Région"] = df["region"].map(TRAD_REGION)
    return df


# -----------------------------------------------------------------------------
# Entraînement du modèle final, avec exactement le même prétraitement que le notebook.
# Mis en cache : il n'est entraîné qu'une seule fois, au lancement de l'application.
# -----------------------------------------------------------------------------
@st.cache_resource
def entrainer_modele(df):
    # 1. Encodage One-Hot des variables catégorielles
    df_encoded = pd.get_dummies(df[["age", "sex", "bmi", "children", "smoker", "region", "expenses"]],
                                columns=["sex", "smoker", "region"], drop_first=True, dtype=int)

    # 2. Séparation des caractéristiques (X) et de la cible (y)
    X = df_encoded[COLONNES_MODELE]
    y = df_encoded["expenses"]

    # 3. Division 80 % / 20 %, stratifiée sur le statut fumeur
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=X["smoker_yes"])

    # 4. Standardisation des variables numériques uniquement, apprise sur le train
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[VARIABLES_NUM] = scaler.fit_transform(X_train[VARIABLES_NUM])
    X_test_scaled[VARIABLES_NUM] = scaler.transform(X_test[VARIABLES_NUM])

    # 5. Optimisation : réglages testés dans le notebook (étape 9), pour réduire
    #    la complexité des arbres et limiter le surapprentissage
    reglages = {
        "Base (sans limite)": {},
        "max_depth = 4": {"max_depth": 4},
        "max_depth = 6": {"max_depth": 6},
        "min_samples_leaf = 5": {"min_samples_leaf": 5},
        "min_samples_leaf = 10": {"min_samples_leaf": 10},
        "min_samples_leaf = 20": {"min_samples_leaf": 20},
    }
    resultats = []
    for nom, parametres in reglages.items():
        rf = RandomForestRegressor(random_state=42, **parametres).fit(X_train_scaled, y_train)
        resultats.append({
            "Réglage": nom,
            "MAE train": mean_absolute_error(y_train, rf.predict(X_train_scaled)),
            "MAE test": mean_absolute_error(y_test, rf.predict(X_test_scaled))
        })
    optimisation = pd.DataFrame(resultats)

    # 6. Modèle final : Random Forest avec au moins 10 assurés par feuille
    modele = RandomForestRegressor(min_samples_leaf=10, random_state=42)
    modele.fit(X_train_scaled, y_train)

    # 7. Métriques du modèle final sur le jeu de test
    y_pred = modele.predict(X_test_scaled)
    metriques = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "R²": r2_score(y_test, y_pred)
    }
    return modele, scaler, metriques, optimisation


df = charger_donnees()
modele, scaler, metriques, optimisation = entrainer_modele(df)


# -----------------------------------------------------------------------------
# Préparation d'un profil saisi dans le simulateur, avec le même prétraitement
# que pour l'entraînement : encodage 0/1 puis standardisation
# -----------------------------------------------------------------------------
def preparer_profil(age, sex, bmi, children, smoker, region):
    donnees = pd.DataFrame([[0.0] * len(COLONNES_MODELE)], columns=COLONNES_MODELE)

    # Variables numériques
    donnees["age"] = age
    donnees["bmi"] = bmi
    donnees["children"] = children

    # Variables catégorielles : la colonne correspondante passe à 1
    # (female, no et northeast sont les modalités de référence, sans colonne)
    if sex == "male":
        donnees["sex_male"] = 1
    if smoker == "yes":
        donnees["smoker_yes"] = 1
    if region != "northeast":
        donnees[f"region_{region}"] = 1

    # Standardisation avec le scaler appris sur le jeu d'entraînement
    donnees[VARIABLES_NUM] = scaler.transform(donnees[VARIABLES_NUM])
    return donnees


def predire(age, sex, bmi, children, smoker, region):
    return modele.predict(preparer_profil(age, sex, bmi, children, smoker, region))[0]


def categorie_imc(bmi):
    """Catégorie de corpulence selon les seuils de l'OMS."""
    if bmi < 18.5:
        return "insuffisance pondérale"
    if bmi < 25:
        return "corpulence normale"
    if bmi < 30:
        return "surpoids"
    return "obésité"


# -----------------------------------------------------------------------------
# Navigation dans la barre latérale : une section par page
# -----------------------------------------------------------------------------
st.sidebar.title("🩺 Aegis Health Coverage")
st.sidebar.write("")
st.sidebar.markdown("Tarification des contrats d'assurance santé : analyse des frais médicaux "
                    f"de {len(df):,} assurés et estimation de la prime d'un souscripteur.".replace(",", " "))
st.sidebar.divider()
section = st.sidebar.radio("Navigation",
                           ["📊 Analyse des coûts", "🌲 Le modèle", "🧮 Simulateur de prime"])

# =============================================================================
# SECTION 1 : ANALYSE DES COÛTS
# =============================================================================
if section == "📊 Analyse des coûts":
    st.header("📊 Analyse des coûts")

    # --- Filtres, dans la barre latérale ---
    st.sidebar.divider()
    st.sidebar.subheader("Filtres")
    filtre_fumeur = st.sidebar.multiselect("Statut fumeur", ["Fumeur", "Non-fumeur"],
                                           default=["Fumeur", "Non-fumeur"])
    filtre_sexe = st.sidebar.multiselect("Sexe", ["Femme", "Homme"],
                                         default=["Femme", "Homme"])
    filtre_region = st.sidebar.multiselect("Région", list(TRAD_REGION.values()),
                                           default=list(TRAD_REGION.values()))
    filtre_age = st.sidebar.slider("Âge", int(df["age"].min()), int(df["age"].max()),
                                   (int(df["age"].min()), int(df["age"].max())))

    df_f = df[df["Statut fumeur"].isin(filtre_fumeur)
              & df["Sexe"].isin(filtre_sexe)
              & df["Région"].isin(filtre_region)
              & df["age"].between(*filtre_age)]

    if df_f.empty:
        st.warning("Aucun assuré ne correspond à ces filtres. "
                   "Élargissez la sélection pour afficher les graphiques.")
    else:
        # --- Indicateurs clés ---
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Assurés", f"{len(df_f):,}".replace(",", " "))
        k2.metric("Frais médians", format_dollars(df_f["expenses"].median()))
        k3.metric("Frais moyens", format_dollars(df_f["expenses"].mean()))
        k4.metric("Part de fumeurs", f"{(df_f['smoker'] == 'yes').mean() * 100:.1f} %")

        # --- Un thème par onglet ---
        onglet_repartition, onglet_profil, onglet_risques, onglet_corr = st.tabs(
            ["Répartition des frais", "Frais selon le profil",
             "Facteurs de risque combinés", "Matrice de corrélation"])

        # Onglet 1 : nombre d'assurés par tranche de frais de 5 000 $, fumeurs / non-fumeurs
        with onglet_repartition:
            st.markdown("**Répartition des assurés par tranche de frais médicaux**")
            bornes = list(range(0, 70001, 5000))
            libelles = [f"{b // 1000}-{(b + 5000) // 1000} k$" for b in bornes[:-1]]
            df_tranches = df_f.assign(Tranche=pd.cut(df_f["expenses"], bins=bornes,
                                                     labels=libelles, right=False))
            comptage = (df_tranches.groupby(["Tranche", "Statut fumeur"], observed=False)
                        .size().reset_index(name="Assurés"))

            fig_rep = px.bar(comptage, x="Tranche", y="Assurés", color="Statut fumeur",
                             color_discrete_map=COULEURS_FUMEUR, barmode="stack",
                             category_orders={"Tranche": libelles,
                                              "Statut fumeur": ["Non-fumeur", "Fumeur"]},
                             labels={"Tranche": "Frais annuels", "Assurés": "Nombre d'assurés"})
            afficher(fig_rep, 360)

            # Chiffre clé : part des fumeurs parmi les assurés aux frais élevés
            eleves = df_f[df_f["expenses"] >= 30000]
            if not eleves.empty:
                part_fumeurs = (df_f["smoker"] == "yes").mean() * 100
                part_fumeurs_eleves = (eleves["smoker"] == "yes").mean() * 100
                st.info(f"Les fumeurs représentent **{part_fumeurs:.0f} %** des assurés, "
                        f"mais **{part_fumeurs_eleves:.0f} %** des assurés dont les frais "
                        "dépassent 30 000 $.")

        # Onglet 2 : deux graphiques côte à côte
        with onglet_profil:
            col_gauche, col_droite = st.columns(2)

            with col_gauche:
                st.markdown("**Frais selon le statut fumeur, le sexe ou la région**")
                variable_cat = st.selectbox("Critère", ["Statut fumeur", "Sexe", "Région"],
                                            key="variable_cat")
                fig_box = px.box(
                    df_f, x=variable_cat, y="expenses", color=variable_cat,
                    color_discrete_map=COULEURS_FUMEUR if variable_cat == "Statut fumeur" else None,
                    labels={"expenses": "Frais annuels ($)"})
                fig_box.update_layout(showlegend=False)
                afficher(fig_box, 380)

            with col_droite:
                st.markdown("**Frais selon l'âge, l'IMC ou le nombre d'enfants**")
                variable_num = st.selectbox(
                    "Variable", VARIABLES_NUM, key="variable_num",
                    format_func=lambda v: {"age": "Âge", "bmi": "IMC",
                                           "children": "Nombre d'enfants"}[v])
                fig_scatter = px.scatter(
                    df_f, x=variable_num, y="expenses", color="Statut fumeur",
                    color_discrete_map=COULEURS_FUMEUR, opacity=0.7,
                    hover_data=["Sexe", "Région"],
                    labels={"expenses": "Frais annuels ($)", "age": "Âge",
                            "bmi": "IMC", "children": "Nombre d'enfants"})
                afficher(fig_scatter, 380)

        # Onglet 3 : frais médians par tranche d'âge, selon le tabac et l'obésité
        with onglet_risques:
            st.markdown("**Frais médicaux médians selon l'âge, le tabac et l'obésité**")

            # 4 profils, à partir du statut fumeur et de l'IMC (obésité : IMC >= 30, seuil OMS)
            def profil_risque(ligne):
                tabac = "Fumeur" if ligne["smoker"] == "yes" else "Non-fumeur"
                poids = "obèse" if ligne["bmi"] >= 30 else "non obèse"
                return f"{tabac} {poids}"

            ordre_profils = ["Fumeur obèse", "Fumeur non obèse",
                             "Non-fumeur obèse", "Non-fumeur non obèse"]
            ordre_ages = ["18-24", "25-34", "35-44", "45-54", "55-64"]
            df_risques = df_f.assign(
                Profil=df_f.apply(profil_risque, axis=1),
                Âge=pd.cut(df_f["age"], bins=[17, 24, 34, 44, 54, 64], labels=ordre_ages))
            medianes = (df_risques.groupby(["Âge", "Profil"], observed=True)["expenses"]
                        .median().reset_index(name="Frais médians"))

            fig_risques = px.line(
                medianes, x="Âge", y="Frais médians", color="Profil", line_dash="Profil",
                markers=True,
                category_orders={"Âge": ordre_ages, "Profil": ordre_profils},
                color_discrete_map={"Fumeur obèse": "#B23A48", "Fumeur non obèse": "#F4A261",
                                    "Non-fumeur obèse": "#1D6F66", "Non-fumeur non obèse": "#6CC5B8"},
                line_dash_map={"Fumeur obèse": "solid", "Fumeur non obèse": "dash",
                               "Non-fumeur obèse": "solid", "Non-fumeur non obèse": "dash"},
                labels={"Âge": "Tranche d'âge", "Frais médians": "Frais médians ($)"})
            afficher(fig_risques, 380)
            st.caption("L'obésité ne change presque rien aux frais des non-fumeurs (courbes "
                       "confondues), mais elle double ceux des fumeurs : les deux facteurs "
                       "se renforcent.")

        # Onglet 4 : matrice de corrélation
        with onglet_corr:
            st.markdown("**Matrice de corrélation**")
            inclure_cat = st.checkbox("Inclure les variables catégorielles encodées "
                                      "(sexe, statut fumeur, région)")
            if inclure_cat:
                df_corr = pd.get_dummies(
                    df_f[["age", "bmi", "children", "sex", "smoker", "region", "expenses"]],
                    columns=["sex", "smoker", "region"], drop_first=True, dtype=int)
            else:
                df_corr = df_f[["age", "bmi", "children", "expenses"]]

            fig_corr = px.imshow(df_corr.corr(), text_auto=".2f",
                                 color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                 aspect="auto")
            afficher(fig_corr, 380)
            st.caption("Une valeur proche de 1 indique une forte relation positive, "
                       "proche de -1 une forte relation négative, proche de 0 aucune "
                       "relation linéaire.")


# =============================================================================
# SECTION 2 : LE MODÈLE
# =============================================================================
elif section == "🌲 Le modèle":
    st.header("🌲 Le modèle")
    st.caption("Random Forest optimisé (au moins 10 assurés par feuille), retenu après "
               "comparaison avec la régression linéaire, le KNN et l'arbre de décision.")

    onglet_perf, onglet_optim, onglet_imp = st.tabs(
        ["Performances", "Optimisation", "Importance des caractéristiques"])

    with onglet_perf:
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE (erreur moyenne)", format_dollars(metriques["MAE"]))
        m2.metric("RMSE", format_dollars(metriques["RMSE"]))
        m3.metric("R²", f"{metriques['R²']:.3f}")
        st.caption("Performances mesurées sur le jeu de test (20 % des assurés), "
                   "que le modèle n'a pas vus pendant l'entraînement.")

    with onglet_optim:
        base = optimisation.iloc[0]
        final = optimisation[optimisation["Réglage"] == "min_samples_leaf = 10"].iloc[0]
        gain = (1 - final["MAE test"] / base["MAE test"]) * 100

        st.markdown(
            f"**Diagnostic** : le modèle de base se trompait de {format_dollars(base['MAE train'])} "
            f"sur les assurés d'entraînement, mais de {format_dollars(base['MAE test'])} sur des "
            "assurés jamais vus (surapprentissage).  \n"
            "**Méthode** : réduire la complexité des arbres, en limitant leur profondeur "
            "(`max_depth`) ou en imposant un minimum d'assurés par feuille (`min_samples_leaf`).")

        # Graphique : MAE train et MAE test pour chaque réglage testé
        donnees_graph = optimisation.melt(id_vars="Réglage", var_name="Jeu", value_name="MAE")
        donnees_graph["Jeu"] = donnees_graph["Jeu"].map(
            {"MAE train": "Entraînement", "MAE test": "Test"})
        fig_optim = px.bar(donnees_graph, x="Réglage", y="MAE", color="Jeu",
                           barmode="group", text_auto=".0f",
                           color_discrete_map={"Entraînement": "#A8B8CC", "Test": "#1F3A5F"},
                           labels={"MAE": "Erreur moyenne ($)", "Réglage": ""})
        afficher(fig_optim, 300)

        st.markdown(
            f"**Résultat** : `min_samples_leaf = 10` donne l'erreur la plus faible sur le test, "
            f"{format_dollars(final['MAE test'])} contre {format_dollars(base['MAE test'])}, "
            f"soit **{gain:.0f} % d'erreur en moins**, sans écart entre entraînement et test.")

    with onglet_imp:
        noms_lisibles = {"age": "Âge", "bmi": "IMC", "children": "Nombre d'enfants",
                         "sex_male": "Sexe (homme)", "smoker_yes": "Statut fumeur",
                         "region_northwest": "Région Nord-Ouest",
                         "region_southeast": "Région Sud-Est",
                         "region_southwest": "Région Sud-Ouest"}
        importances = pd.DataFrame({
            "Caractéristique": [noms_lisibles[c] for c in COLONNES_MODELE],
            "Importance": modele.feature_importances_
        }).sort_values("Importance")

        fig_imp = px.bar(importances, x="Importance", y="Caractéristique", orientation="h",
                         text_auto=".2f", color_discrete_sequence=["#1F3A5F"],
                         labels={"Caractéristique": ""})
        afficher(fig_imp, 360)
        st.caption("Part de la réduction de l'erreur apportée par chaque caractéristique "
                   "dans le modèle final (la somme vaut 1).")


# =============================================================================
# SECTION 3 : SIMULATEUR DE PRIME
# =============================================================================
else:
    st.header("🧮 Simulateur de prime")
    st.caption("Renseignez le profil : l'estimation se met à jour automatiquement.")

    col_saisie, col_resultat = st.columns(2, gap="large")

    # --- Saisie du profil ---
    with col_saisie:
        age = st.slider("Âge", 18, 64, 40)
        sexe_aff = st.radio("Sexe", ["Femme", "Homme"], horizontal=True)
        bmi = st.slider("IMC (indice de masse corporelle)", 15.0, 55.0, 28.0, step=0.1)
        st.caption(f"Catégorie OMS : {categorie_imc(bmi)}")
        enfants = st.slider("Nombre d'enfants à charge", 0, 5, 0)
        fumeur_aff = st.radio("Statut fumeur", ["Non-fumeur", "Fumeur"], horizontal=True)
        region_aff = st.selectbox("Région", list(TRAD_REGION.values()))

    # Conversion des libellés affichés vers les valeurs attendues par le modèle
    sexe = {v: k for k, v in TRAD_SEXE.items()}[sexe_aff]
    fumeur = {v: k for k, v in TRAD_FUMEUR.items()}[fumeur_aff]
    region = {v: k for k, v in TRAD_REGION.items()}[region_aff]

    prediction = predire(age, sexe, bmi, enfants, fumeur, region)

    # --- Résultat ---
    with col_resultat:
        st.metric("Frais médicaux annuels estimés", format_dollars(prediction))
        st.caption(f"Marge d'erreur moyenne du modèle : ± {format_dollars(metriques['MAE'])}. "
                   "Ce montant correspond à la prime pure, c'est-à-dire le coût "
                   "attendu des soins, hors frais de gestion et marge de l'assureur.")

        mediane = df["expenses"].median()
        ecart = (prediction / mediane - 1) * 100
        sens = "au-dessus" if ecart >= 0 else "en dessous"
        st.write(f"Ce profil se situe **{abs(ecart):.0f} % {sens}** des frais médians "
                 f"du portefeuille ({format_dollars(mediane)}).")

        # Impact du tabac sur l'estimation
        if fumeur == "yes":
            prediction_sans_tabac = predire(age, sexe, bmi, enfants, "no", region)
            st.info(f"Sans tabac, l'estimation serait de "
                    f"**{format_dollars(prediction_sans_tabac)}**, soit "
                    f"{format_dollars(prediction - prediction_sans_tabac)} de moins par an.")

        # Impact de l'obésité sur l'estimation
        if bmi >= 30:
            prediction_imc_29 = predire(age, sexe, 29.9, enfants, fumeur, region)
            st.info(f"Avec un IMC juste sous le seuil de l'obésité (29,9), l'estimation "
                    f"serait de **{format_dollars(prediction_imc_29)}**.")
