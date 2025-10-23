import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt

# --- Lecture sécurisée de la clé ENTSO-E ---
if "ENTSOE_API_KEY" in st.secrets:
    ENTSOE_API_KEY = st.secrets["ENTSOE_API_KEY"]
elif os.getenv("ENTSOE_API_KEY"):
    ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")
else:
    ENTSOE_API_KEY = None

if not ENTSOE_API_KEY:
    st.warning("⚠️ Clé ENTSO-E manquante. Les données réelles ne seront pas chargées.")
else:
    st.success("✅ Clé ENTSO-E détectée avec succès !")

st.set_page_config(page_title="Simulateur Revenus BESS Suisse", layout="wide")

st.title("🔋 Simulateur de revenus pour systèmes de stockage en Suisse")

st.sidebar.header("⚙️ Paramètres du système")

# --- Configuration générale ---
type_systeme = st.sidebar.selectbox(
    "Type de système", 
    [
        "Batterie couplée à une installation PV sur bâtiment",
        "Batterie sur bâtiment sans PV",
        "Batterie raccordée directement au réseau"
    ]
)

# --- Caractéristiques techniques ---
st.sidebar.subheader("Paramètres techniques de la batterie")
cap_batt = st.sidebar.number_input("Capacité batterie (kWh)", 100, 5000, 1000, 100)
p_batt = st.sidebar.number_input("Puissance batterie (kW)", 50, 2000, 500, 50)
rendement = st.sidebar.slider("Rendement global (%)", 70, 100, 90)
dod = st.sidebar.slider("Profondeur de décharge (DoD %)", 50, 100, 90)

# --- Paramètres financiers ---
st.sidebar.subheader("Paramètres financiers")
capex = st.sidebar.number_input("CAPEX (CHF/kWh)", 200, 1000, 400, 10)
opex = st.sidebar.number_input("OPEX annuel (% du CAPEX)", 0, 10, 2, 1)
taux_actualisation = st.sidebar.slider("Taux d'actualisation (%)", 0.0, 15.0, 5.0, 0.1)
duree_proj = st.sidebar.number_input("Durée du projet (années)", 1, 25, 10)

# --- Simulation simple de profils horaires ---
st.subheader("📊 Simulation annuelle simplifiée")

heures = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="H")
prix = 0.1 + 0.05*np.sin(np.linspace(0, 20*np.pi, len(heures)))  # prix simulé CHF/kWh
charge = 50 + 30*np.sin(np.linspace(0, 10*np.pi, len(heures)))   # consommation simulée
prod_pv = np.zeros(len(heures))
if "PV" in type_systeme:
    prod_pv = np.maximum(0, 80*np.sin(np.linspace(0, 365*np.pi, len(heures))))

df = pd.DataFrame({
    "Heure": heures,
    "Prix (CHF/kWh)": prix,
    "Conso (kWh)": charge,
    "Prod_PV (kWh)": prod_pv
})

# --- Revenus simulés (simplifiés pour test) ---
revenu_autoconso = np.sum(np.minimum(prod_pv, charge) * 0.25) if "PV" in type_systeme else 0
revenu_arbitrage = np.std(prix) * cap_batt * 0.1
revenu_peak = np.max(charge) * 0.01 * 12
revenu_services = p_batt * 10

revenu_total = revenu_autoconso + revenu_arbitrage + revenu_peak + revenu_services
invest_initial = cap_batt * capex
opex_annuel = invest_initial * opex / 100
flux = [-(invest_initial)] + [(revenu_total - opex_annuel)/(1+taux_actualisation/100)**i for i in range(1, duree_proj+1)]
roi = (sum(flux) + invest_initial) / invest_initial * 100

# --- Affichage résultats ---
st.metric("💰 Revenu annuel total (CHF)", f"{revenu_total:,.0f}")
st.metric("📈 ROI global (%)", f"{roi:.1f}")

col1, col2 = st.columns(2)
with col1:
    st.write("### Détails des revenus (CHF/an)")
    st.bar_chart({
        "Optimisation autoconsommation": [revenu_autoconso],
        "Arbitrage": [revenu_arbitrage],
        "Peak shaving": [revenu_peak],
        "Services système": [revenu_services]
    })

with col2:
    st.write("### Profil horaire simulé")
    st.line_chart(df[["Conso (kWh)", "Prod_PV (kWh)"]])

st.success("Simulation réalisée avec succès. Une fois la clé ENTSO-E obtenue, les données réelles remplaceront les profils simulés.")
