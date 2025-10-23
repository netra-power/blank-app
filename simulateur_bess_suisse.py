# simulateur_bess_suisse_v2.py
# Streamlit app — Simulation revenus BESS (Suisse)
# v2 with PV/BESS inputs, ENTSO-E robust parsing, revenus, NPV/IRR, autarcie

import os
from datetime import datetime, timedelta
import math
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulateur BESS Suisse v2", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def ensure_len(arr, n=8760):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == n:
        return arr
    if len(arr) == 8784:  # leap year -> trim
        return arr[:n]
    if len(arr) < n:
        tail = np.full(n - len(arr), arr[-1] if len(arr) else 0.0, dtype=float)
        return np.concatenate([arr, tail])
    return arr[:n]

def year_hours(start_year=2024):
    start = datetime(start_year,1,1,0,0)
    idx = pd.date_range(start, periods=8760, freq="H")
    return idx

def build_consumption_profile(kind:str, annual_kwh:float, seed=7, start_year=2024):
    """Generate synthetic hourly load profile for building types in CH context"""
    rng = np.random.RandomState(seed)
    t = np.arange(8760)
    # base pattern by building type
    if kind == "Résidentiel":
        daily = 0.8 + 0.4*np.sin(2*np.pi*((t%24)-7)/24)**2  # evenings
        seasonal = 1.0 + 0.25*np.sin(2*np.pi*t/(24*365)-0.3)
    elif kind == "Tertiaire (bureaux)":
        daily = 0.6 + 0.7*( ((t%24)>=7) & ((t%24)<=18) )  # office hours
        seasonal = 1.0 + 0.15*np.sin(2*np.pi*t/(24*365)-0.2)
    elif kind == "Industriel léger":
        daily = 0.9 + 0.3*( ((t%24)>=6) & ((t%24)<=22) )
        seasonal = 1.0 + 0.10*np.sin(2*np.pi*t/(24*365)-0.1)
    else:  # Industriel lourd
        daily = np.full_like(t, 1.0, dtype=float)
        seasonal = 1.0 + 0.05*np.sin(2*np.pi*t/(24*365))
    noise = rng.normal(1.0, 0.05, len(t))
    prof = daily*seasonal*noise
    prof[prof<0] = 0
    scale = annual_kwh / prof.sum() if prof.sum()>0 else 0.0
    prof *= scale
    return pd.Series(prof, index=year_hours(start_year))

def build_pv_profile(kWc:float, start_year=2024):
    """Very simple PV shape for CH latitude; returns kWh/h production"""
    idx = year_hours(start_year)
    t = np.arange(len(idx))
    day = np.clip(np.sin(2*np.pi*((t%24)-6)/24), 0, None)  # mid-day peak
    seasonal = (np.sin(2*np.pi*t/(24*365)-0.1)+1)/2 * 0.7 + 0.3
    # assume specific yield ~ 1000 kWh/kWc/year baseline
    raw = kWc * day * seasonal
    scale = (1000.0*kWc)/raw.sum() if raw.sum()>0 else 0.0
    return pd.Series(raw*scale, index=idx)

def parse_entsoe_a44(xml_text:str):
    """Parse ENTSO-E A44 XML day-ahead prices into hourly EUR/MWh series"""
    root = ET.fromstring(xml_text)
    ns = {"ns": root.tag.split("}")[0].strip("{")}
    prices = []
    times = []
    for ts in root.findall(".//ns:TimeSeries", ns):
        for period in ts.findall(".//ns:Period", ns):
            start_s = period.find("./ns:timeInterval/ns:start", ns).text
            res = period.find("./ns:resolution", ns)
            resolution = res.text if res is not None else "PT60M"
            start_dt = datetime.fromisoformat(start_s.replace("Z","+00:00")).replace(tzinfo=None)
            step = timedelta(minutes=60 if resolution=="PT60M" else 15)
            for pt in period.findall("./ns:Point", ns):
                pos = int(pt.find("./ns:position", ns).text)
                val_el = pt.find("./ns:price.amount", ns)
                if val_el is None:
                    # fallback: find any child containing 'price'
                    for ch in pt:
                        if "price" in ch.tag:
                            val_el = ch; break
                if val_el is None:
                    continue
                try:
                    val = float(val_el.text)
                except:
                    continue
                t = start_dt + step*(pos-1)
                times.append(t)
                prices.append(val)  # EUR/MWh
    if not prices:
        return None
    df = pd.DataFrame({"datetime": times, "EUR_MWh": prices})
    df = df.sort_values("datetime").reset_index(drop=True)
    # resample hourly if 15-min data
    df = df.set_index("datetime")
    if df.index.freq is None:
        df = df.asfreq("H", method="pad")
    df["EUR_kWh"] = df["EUR_MWh"]/1000.0
    return df.reset_index()

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
        return None, f"Parsing error: {e}"

def irr(cashflows):
    # Newton iteration fallback
    try:
        guess = 0.1
        for _ in range(100):
            f = sum(cf/((1+guess)**i) for i, cf in enumerate(cashflows))
            df = sum(-i*cf/((1+guess)**(i+1)) for i, cf in enumerate(cashflows[1:], start=1))
            if abs(df) < 1e-12: break
            new = guess - f/df
            if abs(new-guess) < 1e-8: return new
            guess = new
        return guess if not math.isnan(guess) else None
    except Exception:
        return None

# -----------------------------
# UI — Inputs
# -----------------------------
st.title("🔋 Simulateur revenus BESS Suisse — v2")
st.caption("Devise: CHF — profils horaires sur 1 an (8760 h).")

with st.sidebar:
    st.header("Configuration")
    system_type = st.selectbox("Type de système", [
        "PV + Batterie sur bâtiment",
        "Batterie sur bâtiment (sans PV)",
        "Batterie raccordée au réseau (BT/MT)"
    ])
    marche_libre = st.radio("Bâtiment sur le marché libre ?", ["Oui", "Non"], index=1 if "réseau" in system_type.lower() else 1)
    eur_chf = st.number_input("Taux EUR→CHF (pour prix ENTSO-E)", min_value=0.5, max_value=2.0, value=1.0, step=0.01)

    st.subheader("Bâtiment — consommation")
    cons_upload = st.file_uploader("Importer profil conso (CSV: 8760 valeurs)", type=["csv"])
    if cons_upload is None:
        building_kind = st.selectbox("Profil bâtiment (si pas de CSV)", ["Résidentiel","Tertiaire (bureaux)","Industriel léger","Industriel lourd"])
        annual_kwh = st.number_input("Conso annuelle (kWh)", min_value=0.0, value=50000.0, step=100.0)

    st.subheader("PV — production & infos")
    has_pv = ("PV" in system_type)
    if has_pv:
        pv_kwc = st.number_input("Puissance PV installée (kWc)", min_value=0.0, value=100.0, step=1.0)
        pv_kva = st.number_input("Puissance PV apparente (kVA)", min_value=0.0, value=100.0, step=1.0)
        pv_upload = st.file_uploader("Importer profil PV (CSV: 8760 valeurs kWh)", type=["csv"])
        # Option: saisie manuelle totaux (si pas de profil)
        pv_total_kwh = st.number_input("Production PV annuelle estimée (kWh) (si pas de profil)", min_value=0.0, value=100000.0, step=100.0)

    st.subheader("Batterie — paramètres")
    batt_kwh = st.number_input("Capacité batterie (kWh)", min_value=0.0, value=200.0, step=10.0)
    batt_kw = st.number_input("Puissance batterie (kW)", min_value=0.0, value=100.0, step=10.0)
    eff_rt = st.number_input("Rendement round-trip (%)", min_value=50.0, max_value=100.0, value=90.0, step=0.5)/100.0
    dod = st.number_input("Profondeur de décharge (DoD %)", min_value=10.0, max_value=100.0, value=90.0, step=1.0)/100.0

    st.subheader("Finances / coûts")
    capex_pv_per_kwc = st.number_input("CAPEX PV (CHF/kWc)", min_value=0.0, value=900.0, step=10.0)
    capex_batt_per_kwh = st.number_input("CAPEX Batterie (CHF/kWh)", min_value=0.0, value=350.0, step=10.0)
    opex_pct = st.number_input("OPEX annuel (% CAPEX total)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)/100.0
    years = st.number_input("Durée de simulation (années)", min_value=1, max_value=30, value=10, step=1)
    disc = st.number_input("Taux d'actualisation (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)/100.0

    st.subheader("Tarification & services")
    if marche_libre=="Non":
        price_buy_fixed = st.number_input("Tarif achat (CHF/kWh)", min_value=0.0, value=0.18, step=0.005)
        price_sell_fixed = st.number_input("Tarif revente surplus PV (CHF/kWh)", min_value=0.0, value=0.08, step=0.005)
    else:
        price_buy_fixed = None
        price_sell_fixed = None
    peak_tariff = st.number_input("Tarif puissance (CHF/kW/an) — peak shaving", min_value=0.0, value=150.0, step=5.0)
    services_kw = st.number_input("Puissance contractée services système (kW)", min_value=0.0, value=0.0, step=10.0)
    services_tariff = st.number_input("Rémunération services (CHF/kW/an)", min_value=0.0, value=40.0, step=1.0)

# -----------------------------
# Load & PV profiles
# -----------------------------
idx = year_hours(2024)
if cons_upload is not None:
    cons_df = pd.read_csv(cons_upload, header=None)
    load = ensure_len(cons_df.iloc[:,0].values, 8760)
    load = pd.Series(load, index=idx, name="load_kwh")
else:
    load = build_consumption_profile(building_kind, annual_kwh, start_year=2024).rename("load_kwh")

if has_pv:
    if pv_upload is not None:
        pv_df = pd.read_csv(pv_upload, header=None)
        pv = ensure_len(pv_df.iloc[:,0].values, 8760)
        pv = pd.Series(pv, index=idx, name="pv_kwh")
    else:
        pv = build_pv_profile(pv_kwc, start_year=2024).rename("pv_kwh")
        # if user provided annual total override
        if pv_total_kwh>0:
            scale = pv_total_kwh / pv.sum() if pv.sum()>0 else 0.0
            pv = pv*scale
else:
    pv = pd.Series(np.zeros(8760), index=idx, name="pv_kwh")
    pv_kwc = 0.0; pv_kva = 0.0

# -----------------------------
# Prices (ENTSO-E or fixed)
# -----------------------------
if marche_libre=="Oui":
    st.info("Tentative de récupération des prix day-ahead (ENTSO-E / Swissgrid)...")
    entsoe_df, err = fetch_entsoe_prices(days=30)
    if entsoe_df is not None:
        prices = entsoe_df.set_index("datetime")["EUR_KWh".lower()]  # ensure lower
        prices = prices.reindex(idx, method="ffill").fillna(method="bfill") * eur_chf
        prices.name = "price_chf_kwh"
        st.success("Prix ENTSO-E chargés (day-ahead).")
    else:
        st.warning(f"Données réelles indisponibles ({err}). Utilisation d'un profil de prix synthétique.")
        t = np.arange(8760)
        prices = pd.Series(0.12 + 0.03*np.sin(2*np.pi*(t%24)/24) + 0.02*np.sin(2*np.pi*t/(24*365)), index=idx, name="price_chf_kwh")
else:
    prices = pd.Series(np.full(8760, price_buy_fixed), index=idx, name="price_chf_kwh")

# -----------------------------
# Dispatch & revenues
# -----------------------------
def simulate_dispatch(load_kwh, pv_kwh, price_chf_kwh, cap_kwh, p_kw, eff_rt, dod, market_free:bool):
    """Greedy dispatch: prioritize PV self-consumption, then arbitrage on prices (market_free)"""
    soc = 0.5*cap_kwh
    soc_min = (1-dod)*cap_kwh
    soc_max = cap_kwh
    charged = np.zeros(8760)
    discharged = np.zeros(8760)
    rev_autocons = np.zeros(8760)
    rev_arb = np.zeros(8760)

    # arbitrage thresholds
    if market_free:
        low = np.quantile(price_chf_kwh.values, 0.25)
        high = np.quantile(price_chf_kwh.values, 0.75)
    else:
        low = high = None

    for i, _ in enumerate(load_kwh.index):
        load_h = load_kwh.iloc[i]
        pv_h = pv_kwh.iloc[i]
        price = price_chf_kwh.iloc[i]

        # Step 1: PV for load, surplus charges battery
        net_after_pv = load_h - pv_h  # >0 deficit, <0 surplus
        usable_p = p_kw

        if net_after_pv < -1e-9:
            surplus = -net_after_pv
            charge = min(surplus, usable_p, soc_max - soc)
            soc += charge * eff_rt
            charged[i] += charge
            # surplus remaining would be exported
        else:
            deficit = net_after_pv
            if deficit > 1e-9:
                discharge = min(deficit, usable_p, soc - soc_min)
                delivered = discharge * eff_rt
                soc -= discharge
                discharged[i] += delivered
                # avoided purchase at price
                rev_autocons[i] += delivered * price

        # Step 2: Arbitrage (market free only) — charge when low, discharge when high
        if market_free:
            if price <= low*0.98 and soc < soc_max:
                grid_charge = min(usable_p, soc_max - soc)
                soc += grid_charge * eff_rt
                charged[i] += grid_charge

            if price >= high*1.02 and soc > soc_min:
                grid_discharge = min(usable_p, soc - soc_min)
                delivered = grid_discharge * eff_rt
                soc -= grid_discharge
                discharged[i] += delivered
                # sell at current price; assume average buy ~ low
                rev_arb[i] += max(0.0, price - low) * delivered

        # clamp
        soc = min(max(soc, soc_min), soc_max)

    df = pd.DataFrame({
        "load_kwh": load_kwh.values,
        "pv_kwh": pv_kwh.values,
        "price_chf_kwh": price_chf_kwh.values,
        "charged_kwh": charged,
        "discharged_kwh": discharged
    }, index=load_kwh.index)

    # Peak shaving revenue
    net_before = (load_kwh - pv_kwh).clip(lower=0)
    net_after = (load_kwh - pv_kwh - df["discharged_kwh"] + df["charged_kwh"]).clip(lower=0)
    orig_peak = net_before.max()
    new_peak = net_after.max()
    peak_red = max(0.0, orig_peak - new_peak)

    return df, orig_peak, new_peak, peak_red, rev_autocons.sum(), rev_arb.sum()

df_dispatch, orig_peak, new_peak, peak_red, rev_auto, rev_arb = simulate_dispatch(
    load, pv, prices, batt_kwh, batt_kw, eff_rt, dod, marche_libre=="Oui"
)

# PV split & autarky
pv_self = np.minimum(pv, load)
pv_export = np.maximum(0.0, pv - pv_self - df_dispatch["charged_kwh"])
load_after = (load - pv_self - df_dispatch["discharged_kwh"] + df_dispatch["charged_kwh"]).clip(lower=0)
autarky = 1.0 - (load_after.sum()/load.sum() if load.sum()>0 else 0.0)

# Revenus
if marche_libre=="Non":
    # value PV export at fixed sell price; autocons saved at buy price
    rev_pv_auto = pv_self.sum() * price_buy_fixed
    rev_pv_export = pv_export.sum() * price_sell_fixed
else:
    # market free: value both at spot price
    rev_pv_auto = (pv_self * prices).sum()
    rev_pv_export = (pv_export * prices).sum()

# Peak shaving revenue
rev_peak = peak_red * peak_tariff

# Services revenue (simple annual per contracted kW)
rev_services = services_kw * services_tariff

revenues_annual = {
    "Autoconsommation PV": float(rev_pv_auto),
    "Export PV": float(rev_pv_export),
    "Optimisation batterie (autoconso)": float(rev_auto),
    "Arbitrage": float(rev_arb),
    "Peak shaving": float(rev_peak),
    "Services système": float(rev_services),
}

total_rev_year = sum(revenues_annual.values())

# Finances
capex_pv = pv_kwc * capex_pv_per_kwc if has_pv else 0.0
capex_batt = batt_kwh * capex_batt_per_kwh
capex_total = capex_pv + capex_batt
opex_year = capex_total * opex_pct

cashflows = [-capex_total] + [(total_rev_year - opex_year)]*years
npv = sum(cf/((1+disc)**i) for i, cf in enumerate(cashflows))
irr_val = irr(cashflows)
irr_pct = (irr_val*100.0) if (irr_val is not None and irr_val>-0.999) else None

# -----------------------------
# Outputs
# -----------------------------
st.subheader("🔎 Résultats clés")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenu total annuel", f"{total_rev_year:,.0f} CHF/an")
c2.metric("Peak avant/après (kW)", f"{orig_peak:.1f} → {new_peak:.1f}")
c3.metric("Autarcie bâtiment", f"{autarky*100:.1f} %")
c4.metric("CAPEX total", f"{capex_total:,.0f} CHF")

c5, c6, c7 = st.columns(3)
c5.metric("OPEX annuel", f"{opex_year:,.0f} CHF/an")
c6.metric("VAN (NPV)", f"{npv:,.0f} CHF")
c7.metric("TRI (IRR)", f"{irr_pct:.1f} %" if irr_pct is not None else "N/A")

st.markdown("### 💰 Revenus annuels par source (CHF)")
rev_df = pd.DataFrame.from_dict(revenues_annual, orient="index", columns=["CHF/an"])
st.dataframe(rev_df.style.format({"CHF/an":"{:,.0f}"}))

st.markdown("### ☀️ PV — répartition & indicateurs")
pv_stats = pd.DataFrame({
    "Production PV annuelle (kWh)": [pv.sum()],
    "Autoconsommation PV (kWh)": [pv_self.sum()],
    "Export PV (kWh)": [pv_export.sum()],
    "Taux d'autoconsommation (%)": [100.0*(pv_self.sum()/pv.sum() if pv.sum()>0 else 0.0)],
    "Autarcie (%)": [autarky*100.0]
})
st.dataframe(pv_stats.style.format("{:,.1f}"))

st.markdown("### 📈 Graphiques")
fig, ax = plt.subplots(3,1, figsize=(11,8), sharex=True)
sample = slice(0, 168)  # first week
ax[0].plot(load.index[sample], load.values[sample], label="Conso (kWh)")
ax[0].plot(pv.index[sample], pv.values[sample], label="PV (kWh)")
ax[0].legend(); ax[0].set_title("Profils (1ère semaine)")

soc = np.cumsum(df_dispatch["charged_kwh"].values - df_dispatch["discharged_kwh"].values)
soc = np.clip(soc, 0, batt_kwh)
ax[1].plot(load.index[sample], soc[sample], label="SOC approx (kWh)")
ax[1].legend(); ax[1].set_title("État de charge approx")

ax[2].plot(load.index[sample], df_dispatch["discharged_kwh"].values[sample], label="Décharge (kWh)")
ax[2].plot(load.index[sample], df_dispatch["charged_kwh"].values[sample], label="Charge (kWh)")
ax[2].legend(); ax[2].set_title("Flux batterie (1ère semaine)")

st.pyplot(fig)

st.caption(
    "Remarques: les revenus 'Services système' doivent être saisis selon Swissgrid. "
    "Les prix day-ahead proviennent d'ENTSO-E lorsque disponibles. "
    "Le dispatch est heuristique (greedy) et peut être optimisé."
)
