import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulateur BESS Suisse", layout="wide")

st.title("🔋 Simulateur d’autoconsommation avec batterie – Suisse 🇨🇭")

# --- Clé API ENTSO-E
api_key = st.secrets.get("ENTSOE_API_KEY", None)

if api_key:
    st.success("✅ Clé ENTSO-E détectée avec succès !")
else:
    st.warning("⚠️ Aucune clé ENTSO-E trouvée dans Streamlit Secrets. Les données seront simulées.")


# --- Fonction de récupération ENTSO-E robuste
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

        st.info("⏳ Téléchargement des prix Swissgrid en cours...")

        r = requests.get(url)
        if r.status_code != 200:
            raise Exception(f"Erreur {r.status_code}: {r.text}")

        # Lecture XML plus souple
        df = pd.read_xml(r.text)
        possible_cols = [c for c in df.columns if "price" in c.lower()]
        if not possible_cols:
            raise KeyError("Aucune colonne de prix trouvée dans la réponse ENTSO-E.")

        price_col = possible_cols[0]
        df = df[[price_col]].rename(columns={price_col: "EUR_MWh"})
        df["EUR_kWh"] = df["EUR_MWh"] / 1000
        df["datetime"] = pd.date_range(start=start, periods=len(df), freq="H")

        if df.empty:
            raise ValueError("Aucune donnée de prix valide reçue.")

        st.success("✅ Données Swissgrid chargées avec succès.")
        return df

    except Exception as e:
        st.warning(f"⚠️ Erreur de récupération ENTSO-E : {e}")
        return None


# --- Génération de profils simulés (fallback)
def generate_simulated_profiles():
    hours = pd.date_range(datetime.now() - timedelta(days=7), datetime.now(), freq="H")
    load = 20 + 5 * np.sin(np.linspace(0, 4 * np.pi, len(hours)))  # consommation
    pv = np.maximum(0, 10 * np.sin(np.linspace(0, 2 * np.pi, len(hours))))  # production PV
    price = 0.20 + 0.05 * np.sin(np.linspace(0, 4 * np.pi, len(hours)))  # prix fictifs
    df = pd.DataFrame({"datetime": hours, "load_kW": load, "pv_kW": pv, "EUR_kWh": price})
    return df


# --- Simulation principale
def simulate_bess_operation(df, capacity_kWh, power_kW, efficiency=0.9):
    soc = 0
    soc_list, grid_import, grid_export = [], [], []

    for i in range(len(df)):
        load = df.loc[i, "load_kW"]
        pv = df.loc[i, "pv_kW"]
        net = pv - load  # positif = surplus

        if net > 0:
            charge = min(net, power_kW, (capacity_kWh - soc))
            soc += charge * efficiency
            grid_export.append(net - charge)
            grid_import.append(0)
        else:
            discharge = min(-net, power_kW, soc)
            soc -= discharge / efficiency
            grid_import.append(-net - discharge)
            grid_export.append(0)
        soc_list.append(soc)

    df["SOC_kWh"] = soc_list
    df["Grid_Import_kW"] = grid_import
    df["Grid_Export_kW"] = grid_export

    return df


# --- Interface utilisateur
st.header("⚙️ Paramètres de simulation")
col1, col2 = st.columns(2)
with col1:
    battery_capacity = st.number_input("Capacité batterie (kWh)", 10, 2000, 200)
    battery_power = st.number_input("Puissance batterie (kW)", 5, 1000, 50)
with col2:
    eff = st.slider("Rendement de conversion (%)", 80, 100, 90) / 100

# --- Données réelles ou simulées
if api_key:
    df = get_entsoe_prices(api_key)
    if df is None:
        df = generate_simulated_profiles()
        data_type = "profil simulé"
    else:
        # On complète avec consommation et PV simulés pour la démonstration
        hours = len(df)
        df["load_kW"] = 20 + 5 * np.sin(np.linspace(0, 4 * np.pi, hours))
        df["pv_kW"] = np.maximum(0, 10 * np.sin(np.linspace(0, 2 * np.pi, hours)))
        data_type = "prix réels Swissgrid"
else:
    df = generate_simulated_profiles()
    data_type = "profil simulé"

# --- Simulation
df = simulate_bess_operation(df, battery_capacity, battery_power, eff)

st.subheader("📈 Résultats")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["datetime"], df["load_kW"], label="Consommation (kW)")
ax.plot(df["datetime"], df["pv_kW"], label="Production PV (kW)")
ax.plot(df["datetime"], df["SOC_kWh"], label="État de charge batterie (kWh)")
ax.legend()
ax.set_xlabel("Temps")
ax.set_ylabel("Puissance / Énergie (kW / kWh)")
st.pyplot(fig)

st.success(f"✅ Simulation réalisée avec {data_type}.")
