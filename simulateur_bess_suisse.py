# simulateur_bess_suisse.py
# Streamlit app — Simulation revenus BESS (Suisse)
# Version finale – configurations PV/bâtiment/réseau, revenus, PV split, sources d'énergie, profils été/hiver,
# Swissgrid (ENTSO-E), peak shaving annuel (si marché libre), cashflow cumulé

import os
from datetime import datetime, timedelta
import math
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulateur revenus BESS Suisse", layout="wide")

# -----------------------------
# Fonctions utilitaires
# -----------------------------
def ensure_len(arr, n=8760):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == n:
        return arr
    if len(arr) == 8784:
        return arr[:n]
    if len(arr) < n:
        tail = np.full(n - len(arr), arr[-1] if len(arr) else 0.0)
        return np.concatenate([arr, tail])
    return arr[:n]

def year_hours(start_year=2024):
    return pd.date_range(datetime(start_year, 1, 1, 0, 0), periods=8760, freq="H")

def build_consumption_profile(kind, annual_kwh, seed=7, start_year=2024):
    rng = np.random.RandomState(seed)
    t = np.arange(8760)
    if kind == "Résidentiel":
        daily = 0.8 + 0.4*np.sin(2*np.pi*((t%24)-7)/24)**2
        seasonal = 1.0 + 0.25*np.sin(2*np.pi*t/(24*365)-0.3)
    elif kind == "Tertiaire (bureaux)":
        daily = 0.6 + 0.7*((t%24)>=7)*((t%24)<=18)
        seasonal = 1.0 + 0.15*np.sin(2*np.pi*t/(24*365)-0.2)
    elif kind == "Industriel léger":
        daily = 0.9 + 0.3*((t%24)>=6)*((t%24)<=22)
        seasonal = 1.0 + 0.10*np.sin(2*np.pi*t/(24*365)-0.1)
    else:
        daily = np.full_like(t, 1.0)
        seasonal = 1.0 + 0.05*np.sin(2*np.pi*t/(24*365))
    noise = rng.normal(1.0, 0.05, len(t))
    prof = daily * seasonal * noise
    prof[prof < 0] = 0
    prof *= annual_kwh / prof.sum()
    return pd.Series(prof, index=year_hours(start_year))

def build_pv_profile(kWc, start_year=2024):
    idx = year_hours(start_year)
    t = np.arange(len(idx))
    day = np.clip(np.sin(2*np.pi*((t%24)-6)/24), 0, None)
    seasonal = (np.sin(2*np.pi*t/(24*365)-0.1)+1)/2 * 0.7 + 0.3
    raw = kWc * day * seasonal
    raw *= (1000.0*kWc)/raw.sum()
    return pd.Series(raw, index=idx)

def parse_entsoe_a44(xml_text):
    root = ET.fromstring(xml_text)
    ns = {"ns": root.tag.split("}")[0].strip("{")}
    prices, times = [], []
    for ts in root.findall(".//ns:TimeSeries", ns):
        for period in ts.findall(".//ns:Period", ns):
            start_s = period.find("./ns:timeInterval/ns:start", ns).text
            start_dt = datetime.fromisoformat(start_s.replace("Z","+00:00")).replace(tzinfo=None)
            for pt in period.findall("./ns:Point", ns):
                pos = int(pt.find("./ns:position", ns).text)
                val_el = pt.find("./ns:price.amount", ns)
                if val_el is None:
                    for ch in pt:
                        if "price" in ch.tag:
                            val_el = ch; break
                if val_el is None: continue
                try:
                    val = float(val_el.text)
                except: continue
                times.append(start_dt + timedelta(hours=pos-1))
                prices.append(val)
    if not prices:
        return None
    df = pd.DataFrame({"datetime": times, "EUR_MWh": prices})
    df["EUR_kWh"] = df["EUR_MWh"]/1000.0
    return df

@st.cache_data(show_spinner=True)
def fetch_entsoe_prices(days=30, in_domain="10YCH-SWISSGRIDZ"):
    token = st.secrets.get("ENTSOE_API_KEY", None)
    if not token:
        return None, "Pas de clé ENTSO-E"
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    url = (
        f"https://web-api.tp.entsoe.eu/api?securityToken={token}"
        f"&documentType=A44&in_Domain={in_domain}&out_Domain={in_domain}"
        f"&periodStart={start.strftime('%Y%m%d%H%M')}&periodEnd={end.strftime('%Y%m%d%H%M')}"
    )
    r = requests.get(url, timeout=40)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    try:
        df = parse_entsoe_a44(r.text)
        if df is None or df.empty:
            return None, "XML sans prix lisibles"
        return df, None
    except Exception as e:
        return None, f"Erreur parsing: {e}"

# -----------------------------
# Interface utilisateur
# -----------------------------
st.title("🔋 Simulateur revenus BESS Suisse")
st.caption("Devise: CHF — Profils horaires sur 1 an (8760 h)")

with st.sidebar:
    st.header("Configuration")
    system_type = st.selectbox("Type de configuration", [
        "Batterie couplée à un bâtiment + PV",
        "Batterie couplée à un bâtiment",
        "Batterie couplée au réseau"
    ])

    # Le marché libre s'applique uniquement s'il y a un bâtiment
    if "bâtiment" in system_type.lower():
        marche_libre = st.radio("Bâtiment sur le marché libre ?", ["Oui", "Non"], index=1)
    else:
        marche_libre = "Non"

    eur_chf = st.number_input("Taux EUR→CHF", min_value=0.5, max_value=2.0, value=1.0, step=0.01)

    st.subheader("📊 Bâtiment — Consommation")
    st.markdown("> ⚙️ **Format CSV attendu :** 1 colonne, 8760 valeurs horaires (kWh/h), sans en-tête (séparateur ',' ou ';').")
    cons_upload = st.file_uploader("Importer profil conso (CSV)", type=["csv"])
    if cons_upload is None:
        building_kind = st.selectbox("Profil type", ["Résidentiel", "Tertiaire (bureaux)", "Industriel léger", "Industriel lourd"])
        annual_kwh = st.number_input("Consommation annuelle (kWh)", min_value=0.0, value=50000.0, step=1000.0, format="%.0f")

    has_pv = "PV" in system_type
    if has_pv:
        st.subheader("☀️ Photovoltaïque")
        pv_kwc = st.number_input("Puissance installée (kWc)", min_value=0.0, value=100.0, step=1.0, format="%.0f")
        pv_kva = st.number_input("Puissance apparente (kVA)", min_value=0.0, value=100.0, step=1.0, format="%.0f")
        st.markdown("> ⚙️ **Format CSV PV attendu :** 1 colonne, 8760 valeurs horaires (kWh/h), sans en-tête.")
        pv_upload = st.file_uploader("Importer profil PV (CSV)", type=["csv"])
        pv_total_kwh = st.number_input("Production annuelle estimée (kWh)", min_value=0.0, value=100000.0, step=1000.0, format="%.0f")
    else:
        pv_kwc = pv_kva = pv_total_kwh = 0.0
        pv_upload = None

    st.subheader("🔋 Batterie")
    batt_kwh = st.number_input("Capacité (kWh)", min_value=0.0, value=200.0, step=10.0, format="%.0f")
    batt_kw = st.number_input("Puissance (kW)", min_value=0.0, value=100.0, step=10.0, format="%.0f")
    eff_rt = st.number_input("Rendement (%)", min_value=50.0, max_value=100.0, value=90.0, step=1.0, format="%.0f") / 100
    dod = st.number_input("Profondeur de décharge (%)", min_value=10.0, max_value=100.0, value=90.0, step=1.0, format="%.0f") / 100

    st.subheader("💰 Paramètres économiques")
    capex_pv_per_kwc = st.number_input("CAPEX PV (CHF/kWc)", min_value=0.0, value=900.0, step=10.0, format="%.0f")
    capex_batt_per_kwh = st.number_input("CAPEX Batterie (CHF/kWh)", min_value=0.0, value=350.0, step=10.0, format="%.0f")
    years = st.number_input("Durée de simulation (ans)", min_value=1.0, max_value=30.0, value=10.0, step=1.0, format="%.0f")
    disc = st.number_input("Taux d'actualisation (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1) / 100

    st.subheader("🔌 Tarification & services")
    if marche_libre == "Non":
        price_buy_fixed = st.number_input("Tarif achat (CHF/kWh)", min_value=0.0, value=0.18, step=0.01)
        price_sell_fixed = st.number_input("Tarif revente PV (CHF/kWh)", min_value=0.0, value=0.08, step=0.01)
    else:
        price_buy_fixed = price_sell_fixed = None
    peak_tariff = st.number_input("Tarif puissance (CHF/kW/an)", min_value=0.0, value=150.0, step=5.0, format="%.0f")
    services_kw = st.number_input("Puissance contractée (kW)", min_value=0.0, value=0.0, step=10.0, format="%.0f")
    services_tariff = st.number_input("Rémunération services (CHF/kW/an)", min_value=0.0, value=40.0, step=1.0, format="%.0f")

# -----------------------------
# Profils (conso & PV)
# -----------------------------
idx = year_hours(2024)
if cons_upload:
    cons_df = pd.read_csv(cons_upload, header=None, sep=None, engine="python")
    load = ensure_len(cons_df.iloc[:, 0].values, 8760)
    load = pd.Series(load, index=idx)
else:
    load = build_consumption_profile(building_kind, annual_kwh, start_year=2024)

if has_pv:
    if pv_upload:
        pv_df = pd.read_csv(pv_upload, header=None, sep=None, engine="python")
        pv = ensure_len(pv_df.iloc[:, 0].values, 8760)
        pv = pd.Series(pv, index=idx)
    else:
        pv = build_pv_profile(pv_kwc)
        if pv_total_kwh > 0:
            pv *= pv_total_kwh / pv.sum()
else:
    pv = pd.Series(np.zeros(8760), index=idx)

# -----------------------------
# Prix de l'électricité
# -----------------------------
if marche_libre == "Oui":
    st.info("⏳ Téléchargement des prix Swissgrid (ENTSO-E)...")
    entsoe_df, err = fetch_entsoe_prices(30)
    if entsoe_df is not None:
        prices = entsoe_df.set_index("datetime")["EUR_kWh"].reindex(idx, method="ffill") * eur_chf
        st.success("✅ Prix ENTSO-E chargés.")
    else:
        st.warning(f"⚠️ Données indisponibles ({err}). Profil synthétique utilisé.")
        t = np.arange(8760)
        prices = pd.Series(0.12 + 0.03*np.sin(2*np.pi*(t%24)/24), index=idx)
else:
    prices = pd.Series(np.full(8760, price_buy_fixed), index=idx)

# -----------------------------
# Simulation + dispatch (sépare charge PV vs réseau)
# -----------------------------
def simulate_dispatch(load, pv, prices, cap_kwh, p_kw, eff_rt, dod, market_free):
    n = len(load)
    soc = 0.5 * cap_kwh
    soc_min, soc_max = (1-dod)*cap_kwh, cap_kwh

    charged = np.zeros(n)
    discharged = np.zeros(n)
    charged_from_pv = np.zeros(n)
    charged_from_grid = np.zeros(n)

    rev_auto = np.zeros(n)  # valeur de décharge pour couvrir le load (évite achat)
    rev_arb = np.zeros(n)   # valeur arbitrage prix

    if market_free:
        low, high = np.quantile(prices, [0.25, 0.75])
    else:
        low = high = None

    for i in range(n):
        pv_h, load_h, price = pv[i], load[i], prices[i]
        net = load_h - pv_h   # >0 déficit, <0 surplus PV

        # 1) Consommation PV directe et charge batterie depuis PV (si surplus)
        if net < -1e-12:
            surplus = -net
            charge_pv = min(surplus, p_kw, soc_max - soc)
            soc += charge_pv * eff_rt
            charged[i] += charge_pv
            charged_from_pv[i] += charge_pv
            # surplus PV restant => export (géré plus tard)
        else:
            # 2) Si déficit, décharger batterie pour remplacer achat réseau
            deficit = net
            if deficit > 1e-12:
                discharge = min(deficit, p_kw, soc - soc_min)
                delivered = discharge * eff_rt
                soc -= discharge
                discharged[i] += delivered
                rev_auto[i] = delivered * price

        # 3) Arbitrage marché libre: charge réseau à bas prix / décharge à haut prix
        if market_free:
            if price <= low and soc < soc_max:
                grid_charge = min(p_kw, soc_max - soc)
                soc += grid_charge * eff_rt
                charged[i] += grid_charge
                charged_from_grid[i] += grid_charge

            if price >= high and soc > soc_min:
                grid_discharge = min(p_kw, soc - soc_min)
                delivered = grid_discharge * eff_rt
                soc -= grid_discharge
                discharged[i] += delivered
                rev_arb[i] = max(0.0, price - low) * delivered

        soc = min(max(soc, soc_min), soc_max)

    # Net pour peak shaving (inclut charge/décharge batterie)
    net_before = (load - pv).clip(lower=0)
    net_after = (load - pv - discharged + charged).clip(lower=0)

    return (
        charged, discharged, charged_from_pv, charged_from_grid,
        rev_auto.sum(), rev_arb.sum(),
        net_before, net_after
    )

charged, discharged, charged_from_pv, charged_from_grid, rev_auto, rev_arb, net_before, net_after = simulate_dispatch(
    load, pv, prices, batt_kwh, batt_kw, eff_rt, dod, marche_libre=="Oui"
)

# -----------------------------
# PV split & sources d'énergie du bâtiment
# -----------------------------
pv_self_no_bess = np.minimum(pv, load)                  # sans batterie
grid_to_load_no_bess = (load - pv_self_no_bess).clip(lower=0)

pv_self = np.minimum(pv, load)                          # avec batterie (direct PV → charge)
pv_to_batt = charged_from_pv.clip(min=0)
pv_export = np.maximum(0.0, pv - pv_self - pv_to_batt)

# Sources vers la charge du bâtiment (pas les charges batterie)
bess_to_load = discharged                               # batterie → charge du bâtiment
grid_to_load = (load - pv_self - bess_to_load).clip(lower=0)

# Autoconsommation bâtiment (taux)
autoconso_no_bess = (pv_self_no_bess.sum()/pv.sum()*100) if pv.sum()>0 else 0.0
autoconso_with_bess = ((pv_self.sum()+pv_to_batt.sum())/pv.sum()*100) if pv.sum()>0 else 0.0

# -----------------------------
# Revenus
# -----------------------------
if marche_libre == "Non":
    rev_pv_auto = pv_self.sum() * price_buy_fixed
    rev_pv_export = pv_export.sum() * price_sell_fixed
else:
    rev_pv_auto = (pv_self * prices).sum()
    rev_pv_export = (pv_export * prices).sum()

rev_peak = (net_before.max() - net_after.max()) * peak_tariff
rev_services = services_kw * services_tariff

revenus = {
    "Autoconsommation PV": rev_pv_auto,
    "Export PV": rev_pv_export,
    "Optimisation batterie": rev_auto,
    "Arbitrage": rev_arb,
    "Peak shaving": rev_peak,
    "Services système": rev_services
}
total_rev = sum(revenus.values())
capex_total = pv_kwc * capex_pv_per_kwc + batt_kwh * capex_batt_per_kwh

# Cashflow cumulé actualisé (année 0 = -CAPEX, années 1..N = total_rev)
years_int = int(round(years))
cashflows = [-capex_total] + [total_rev]*(years_int)
cum_discounted = []
acc = 0.0
for i, cf in enumerate(cashflows):
    acc += cf / ((1+disc)**i)
    cum_discounted.append(acc)
cum_years = list(range(0, years_int+1))

# -----------------------------
# Résultats — affichage
# -----------------------------
st.subheader("📊 Résultats")

# En-tête : on enlève l'autarcie comme demandé
c1, c2 = st.columns(2)
c1.metric("Revenu annuel total", f"{total_rev:,.0f} CHF")
c2.metric("CAPEX total", f"{capex_total:,.0f} CHF")

st.markdown("### 💰 Détail des revenus (CHF/an)")
rev_df = pd.DataFrame(revenus.items(), columns=["Source", "CHF/an"])
st.dataframe(rev_df.style.format({"CHF/an": "{:,.0f}"}))

# --- Panneau PV : camembert PV split + autoconso sans/avec BESS
st.markdown("### ☀️ PV — Répartition & Autoconsommation")
col_pv1, col_pv2 = st.columns([1.2, 1.0])

with col_pv1:
    fig, ax = plt.subplots(figsize=(5,5))
    labels = ["Autoconsommation directe", "Vers batterie", "Export"]
    sizes = [pv_self.sum(), pv_to_batt.sum(), pv_export.sum()]
    if sum(sizes) <= 0: sizes = [1,0,0]
    ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90)
    ax.set_title("Répartition de la production PV")
    st.pyplot(fig)

with col_pv2:
    st.write("**Autoconsommation du bâtiment**")
    ac_df = pd.DataFrame({
        "Scénario": ["PV sans BESS", "PV avec BESS"],
        "Autoconsommation (%)": [autoconso_no_bess, autoconso_with_bess]
    })
    st.dataframe(ac_df.style.format({"Autoconsommation (%)": "{:,.0f}"}))

# --- Panneau "Sources d'énergie du bâtiment" : sans/avec BESS (deux camemberts)
st.markdown("### 🏢 Sources d'énergie du bâtiment")
col_src1, col_src2 = st.columns(2)
with col_src1:
    fig, ax = plt.subplots(figsize=(5,5))
    labels = ["PV direct", "BESS", "Réseau"]
    sizes = [pv_self_no_bess.sum(), 0.0, grid_to_load_no_bess.sum()]
    if sum(sizes) <= 0: sizes = [1,0,0]
    ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90)
    ax.set_title("Sans batterie")
    st.pyplot(fig)

with col_src2:
    fig, ax = plt.subplots(figsize=(5,5))
    labels = ["PV direct", "BESS", "Réseau"]
    sizes = [pv_self.sum(), bess_to_load.sum(), grid_to_load.sum()]
    if sum(sizes) <= 0: sizes = [1,0,0]
    ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90)
    ax.set_title("Avec batterie")
    st.pyplot(fig)

# --- Profils (été / hiver) : seulement pertinent si PV + BESS + bâtiment
if ("bâtiment" in system_type.lower()) and has_pv and (batt_kwh > 0) and (batt_kw > 0):
    st.markdown("### 📈 Profils — Journées type été / hiver")

    def day_slice(date_str):
        d0 = pd.Timestamp(date_str)
        mask = (idx >= d0) & (idx < d0 + pd.Timedelta(days=1))
        return mask

    # Choix des journées type
    mask_summer = day_slice("2024-07-15")
    mask_winter = day_slice("2024-01-15")

    for label, mask in [("Été (15 juillet)", mask_summer), ("Hiver (15 janvier)", mask_winter)]:
        lday = load[mask].values
        pvday = pv[mask].values
        tday = load[mask].index

        # Conso + PV + hachures autoconsommation
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(tday, lday, label="Consommation bâtiment (kWh/h)")
        ax.plot(tday, pvday, label="Production PV (kWh/h)")
        auto_day = np.minimum(lday, pvday)
        ax.fill_between(tday, 0, auto_day, hatch='//', alpha=0.3, label="Autoconsommation (hachuré)")
        ax.set_title(f"Conso vs PV — {label}")
        ax.legend()
        st.pyplot(fig)

        # Charge / décharge batterie sur la journée
        ch_day = charged[mask].values
        dis_day = discharged[mask].values
        fig2, ax2 = plt.subplots(figsize=(10,3))
        ax2.bar(tday, ch_day, width=0.03, label="Charge (kWh/h)")
        ax2.bar(tday, dis_day, width=0.03, label="Décharge (kWh/h)")
        ax2.set_title(f"Flux batterie — {label}")
        ax2.legend()
        st.pyplot(fig2)

# --- Peak shaving annuel (si marché libre)
if ("bâtiment" in system_type.lower()) and (marche_libre == "Oui"):
    st.markdown("### 🏔️ Peak shaving sur l'année (pics mensuels)")
    month_index = pd.Index(idx).month
    before_monthly_peak = pd.Series(net_before.values, index=idx).groupby(month_index).max()
    after_monthly_peak = pd.Series(net_after.values, index=idx).groupby(month_index).max()

    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(before_monthly_peak.index-0.15, before_monthly_peak.values, width=0.3, label="Avant BESS")
    ax.bar(after_monthly_peak.index+0.15, after_monthly_peak.values, width=0.3, label="Après BESS")
    ax.set_xlabel("Mois")
    ax.set_ylabel("kW")
    ax.set_xticks(range(1,13))
    ax.set_title("Pics mensuels de puissance (avant/après)")
    ax.legend()
    st.pyplot(fig)

# --- Cashflow cumulé (actualisé)
st.markdown("### 💵 Cashflow cumulé (actualisé)")
fig, ax = plt.subplots(figsize=(10,3))
ax.plot(cum_years, cum_discounted, marker="o")
ax.axhline(0, color="gray", linewidth=1)
ax.set_xlabel("Années")
ax.set_ylabel("CHF (actualisés)")
ax.set_title("Évolution du cashflow cumulé (CAPEX + revenus)")
st.pyplot(fig)

st.caption(
    "Remarques : Les prix ENTSO-E (Swissgrid) sont utilisés si la clé API est fournie. "
    "Les fichiers CSV doivent contenir 8760 valeurs horaires (kWh/h) sans en-tête. "
    "Les parts d'énergie 'vers batterie' distinguent PV→batterie et réseau→batterie."
)
