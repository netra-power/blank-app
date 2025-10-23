import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.express as px
import os

st.set_page_config(page_title="Simulateur BESS Suisse", layout="wide")

st.title("🔋 Simulateur de revenus pour systèmes de stockage en Suisse")

# -----------------------------------------------------------
# 1️⃣ Lecture sécurisée de la clé ENTSO-E
# -----------------------------------------------------------
if "ENTSOE_API_KEY" in st.secrets:
    ENTSOE_API_KEY = st.secrets["ENTSOE_API_KEY"]
elif os.getenv("ENTSOE_API_KEY"):
    ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")
else:
    ENTSOE_API_KEY = None

if ENTSOE_API_KEY:
    st.success("✅ Clé ENTSO-E détectée avec succès.")
else:
    st.warning("⚠️ Clé ENTSO-E manquante. Les données réelles ne seront pas chargées.")


# -----------------------------------------------------------
# 2️⃣ Fonction : récupération des prix réels ENTSO-E (Swissgrid)
# -----------------------------------------------------------
@st.cache_data(show_spinner=True)
def get_entsoe_prices(api_key, country_code="10YCH-SWISSGRIDZ", days=7):
    """
    Récupère les prix day-ahead Swissgrid depuis ENTSO-E (A44)
    Retourne un DataFrame avec les prix horaires en EUR/kWh.
    """
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        url = (
            f"https://web-api.tp.entsoe.eu/api?securityToken={api_key}"
            f"&documentType=A44"
            f"&in_Domain={country_code}"
            f"&out_Domain={country_code}"
            f"&periodStart={start.strftime('%Y%m%d%H%M')}"
            f"&periodEnd={end.strftime('%Y%m%d%H%M')}"
        )

        r = requests.get(url)
        if r.status_code != 200:
            raise Exception(f"Erreur {r.status_code}: {r.text}")

        df = pd.read_xml(r.text)
        df = df[df["price.amount"].notnull()][["position", "price.amount"]].rename(
            columns={"price.amount": "EUR_MWh"}
        )
        df["EUR_kWh"] = df["EUR_MWh"] / 1000
        df["datetime"] = pd.date_range(start=start, periods=len(df), freq="H")
        return df

    except Exception as e:
        st.warning(f"⚠️ Erreur de récupération ENTSO-E : {e}")
        return None


# -----------------------------------------------------------
# 3️⃣ Interface utilisateur
# -----------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    system_type = st.selectbox(
        "Type de système de stockage",
        [
            "Batterie couplée à une installation PV et raccordée à un bâtiment",
            "Batterie raccordée à un bâtiment (sans PV)",
            "Batterie raccordée directement au réseau (BT ou MT)",
        ],
    )

    bat_puissance = st.number_input("Puissance batterie (kW)", 10, 5000, 100)
    bat_capacite = st.number_input("Capacité batterie (kWh)", 10, 20000, 200)
    rendement = st.slider("Rendement global du système (%)", 70, 98, 90)
    profondeur = st.slider("Profondeur de décharge (%)", 50, 100, 90)
    duree = st.number_input("Durée de simulation (années)", 1, 25, 10)
    taux_actualisation = st.number_input("Taux d'actualisation (%)", 0.0, 15.0, 5.0)

with col2:
    capex = st.number_input("CAPEX (CHF/kWh)", 100, 2000, 600)
    opex = st.number_input("OPEX annuel (% CAPEX)", 0.0, 10.0, 2.0)
    marche_libre = st.radio("Bâtiment sur le marché libre ?", ["Oui", "Non"])
    tarif_achat = st.number_input("Tarif d'achat (CHF/kWh)", 0.0, 1.0, 0.18)
    tarif_revente = st.number_input("Tarif de revente du surplus PV (CHF/kWh)", 0.0, 1.0, 0.08)

    if "PV" in system_type:
        pv_puissance = st.number_input("Puissance PV installée (kWc)", 1, 5000, 100)
    else:
        pv_puissance = 0

st.divider()

# -----------------------------------------------------------
# 4️⃣ Chargement des prix réels
# -----------------------------------------------------------
if ENTSOE_API_KEY:
    st.info("Téléchargement des prix Swissgrid en cours...")
    price_data = get_entsoe_prices(ENTSOE_API_KEY, days=7)
    if price_data is not None:
        st.success("✅ Données Swissgrid chargées avec succès.")
        fig = px.line(price_data, x="datetime", y="EUR_kWh", title="Prix horaires ENTSO-E (Swissgrid)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Données réelles indisponibles, utilisation d’un profil simulé.")
else:
    price_data = None
    st.warning("⚠️ Pas de clé API détectée — utilisation de données simulées.")


# -----------------------------------------------------------
# 5️⃣ Simulation économique simplifiée
# -----------------------------------------------------------
st.header("📈 Résultats de la simulation")

heures = 8760
if price_data is not None:
    prix = price_data["EUR_kWh"].mean() * 0.95  # CHF/kWh approximé
else:
    prix = 0.15  # CHF/kWh simulé

revenu_autoconso = (
    pv_puissance * 1000 * 1100 * 0.3 * tarif_achat / 1000
    if "PV" in system_type
    else 0
)
revenu_arbitrage = bat_puissance * 8760 * 0.05 * (prix * 0.1)
revenu_services = bat_puissance * 100 * 0.2
revenu_peak = bat_puissance * 50 * 0.3

revenu_total = (
    revenu_autoconso + revenu_arbitrage + revenu_services + revenu_peak
)

invest_initial = bat_capacite * capex
opex_total = invest_initial * (opex / 100) * duree
cashflow_net = revenu_total * duree - opex_total

roi = (cashflow_net - invest_initial) / invest_initial * 100

colA, colB = st.columns(2)
colA.metric("💰 Revenu total annuel estimé", f"{revenu_total:,.0f} CHF/an")
colA.metric("📊 ROI sur la durée du projet", f"{roi:.1f} %")
colB.metric("⚡ Rendement système", f"{rendement} %")
colB.metric("💸 CAPEX total", f"{invest_initial:,.0f} CHF")

st.divider()

if ENTSOE_API_KEY and price_data is None:
    st.warning(
        "Simulation réalisée avec succès. Une fois la clé ENTSO-E obtenue, les données réelles remplaceront les profils simulés."
    )
elif price_data is not None:
    st.success("Simulation réalisée avec les prix réels Swissgrid (ENTSO-E).")

st.caption("🧠 Ce simulateur est une version de démonstration — les valeurs réelles peuvent varier selon le site, les tarifs et les conditions Swissgrid.")

