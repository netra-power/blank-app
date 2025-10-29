# simulateur_bess_suisse.py
# Simulateur revenus BESS Suisse — Version pastel pro (graphiques 2x2, camemberts & cashflow réduits, palette personnalisée)
# Conserve toutes les fonctionnalités précédentes + corrections et UI améliorée

from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from io import BytesIO
import base64

def fig_to_svg(fig):
    buf = BytesIO()
    fig.savefig(buf, format="svg", dpi=300)   # ← on retire bbox_inches="tight"
    return buf.getvalue().decode("utf-8")




# -----------------------------
# Configuration générale (UI + Matplotlib)
# -----------------------------
st.set_page_config(page_title="Simulateur revenus BESS Suisse", layout="wide")
plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": "#DADADA",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#333333",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "legend.frameon": False,
})

# Palette couleurs (validée)
COLORS = {
    "pv": "#FFEE8C",
    "bess_charge": "#BFE3F1",
    "bess_discharge": "#7ABAD6",
    "load": "#FFAB72",
    "grid_export": "#D7F4C2",
    "grid_import": "#A8D79E",
    "text": "#2F3A4A",
    "grey": "#9AA0A6",
}

# -----------------------------
# Helpers
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

    if "bâtiment" in system_type.lower():
        marche_libre = st.radio("Bâtiment sur le marché libre ?", ["Oui", "Non"], index=1)
    else:
        marche_libre = "Non"

    eur_chf = st.number_input("Taux EUR→CHF", min_value=0.5, max_value=2.0, value=1.0, step=0.01)

    st.subheader("📊 Bâtiment — Consommation")
    st.markdown("### 📂 Profil de consommation du bâtiment")

    cons_upload = st.file_uploader("Importer un fichier CSV de consommation (optionnel)", type=["csv"])

    if cons_upload is not None:
        import pandas as pd
    
        df = pd.read_csv(cons_upload, sep=";", header=None, dtype=str)
        df.columns = ["DateHeure", "Consommation"]
    
        unit = df.iloc[1,1].strip()
        df = df.iloc[2:].copy()
    
        df["DateHeure"] = pd.to_datetime(df["DateHeure"], format="%d.%m.%Y %H:%M", errors="coerce")
        df["Consommation"] = df["Consommation"].str.replace(",", ".", regex=False).astype(float)
    
        unit_clean = unit.lower().replace(" ", "")
    
        if unit_clean in ["(kw)", "kw"]:
            st.success("✅ Profil importé en puissance (kW)")
            consum_kW = df.set_index("DateHeure")["Consommation"].rename("Consommation_kW")

    
        elif unit_clean in ["(kwh)", "kwh"]:
            st.success("✅ Profil importé en énergie (kWh) → conversion en kW")
            df = df.sort_values("DateHeure")

        # ✅ Conversion kWh 15 min → kW
            consum_KW = df["Consommation"].astype(float) * 4
            consum_KW.index = df["DateHeure"]

        # ✅ Standardiser le nom de la série
            consum_kW = consum_KW.rename("Consommation_kW")
    
        else:
            st.error("⚠️ L'unité en B2 doit être `(kW)` ou `(kWh)`.")
            st.stop()
    
            # ✅ Tri + suppression des doublons
        consum_kW = consum_kW.sort_index()
        consum_kW = consum_kW[~consum_kW.index.duplicated(keep="first")]

        # ✅ Supprimer toutes les valeurs où la date est NaT (important)
        consum_kW = consum_kW[consum_kW.index.notna()]

    
        # ✅ Détection du pas de temps automatique
        dt_seconds = (
            consum_kW.index.to_series().diff().dropna().dt.total_seconds().mode()[0]
        )
    
        if dt_seconds == 900:
            st.info("⏱️ Pas de temps détecté : **15 min** ✅")
        elif dt_seconds == 1800:
            st.warning("⏱️ Pas de temps détecté : **30 min** → conversion en 15 min")
        elif dt_seconds == 3600:
            st.warning("⏱️ Pas de temps détecté : **1h** → conversion en 15 min")
        else:
            st.warning(f"⏱️ Pas de temps irrégulier ({int(dt_seconds)} sec) → conversion en 15 min")

        # ✅ S'assurer que les valeurs sont bien des floats
        consum_kW = consum_kW.astype(float)

    
        # ✅ Mise sur grille 15 min uniforme
        idx_15m = pd.date_range(consum_kW.index.min(), consum_kW.index.max(), freq="15T")
        consum_kW = consum_kW.reindex(idx_15m, method="nearest")
        
        # ✅ Calcul consommation annuelle réelle à partir du profil importé
        annual_kwh_from_csv = (consum_kW * 0.25).sum()

        # Format nombre avec espace comme séparateur
        annual_kwh_display = f"{annual_kwh_from_csv:,.0f}".replace(",", " ")
        
        st.markdown(
            f"""
            <div style="margin-top:0.5rem; margin-bottom:0.5rem;">
                <span style="font-size:0.9rem; color:#6c757d;">Consommation annuelle (profil importé)</span><br>
                <span style="font-size:1.1rem; font-weight:600;">{annual_kwh_display} kWh</span>
            </div>
            """,
            unsafe_allow_html=True
        )



    else:
        st.info("ℹ️ Aucun fichier importé — saisissez un profil type ci-dessous.")

        building_kind = st.selectbox(
            "Type de bâtiment",
            ["Résidentiel", "Tertiaire", "Industriel"]
        )

        annual_kwh = st.number_input(
            "Consommation annuelle (kWh/an)",
            min_value=100.0, max_value=10_000_000.0, value=670_000.0
        )

        consum_kW = build_consumption_profile(building_kind, annual_kwh, start_year=2024)


    has_pv = "PV" in system_type
    if has_pv:
        st.subheader("☀️ Photovoltaïque")
        pv_kwc = st.number_input("Puissance installée (kWc)", min_value=0.0, value=275.0, step=1.0, format="%.0f")
        pv_kva = st.number_input("Puissance apparente (kVA)", min_value=0.0, value=200.0, step=1.0, format="%.0f")
        st.markdown("> ⚙️ **Format CSV PV attendu :** 1 colonne, 8760 valeurs horaires (kWh/h), sans en-tête.")
        pv_upload = st.file_uploader("Importer profil PV (CSV)", type=["csv"])
        pv_total_kwh = st.number_input("Production annuelle estimée (kWh)", min_value=0.0, value=300000.0, step=1000.0, format="%.0f")
    else:
        pv_kwc = pv_kva = pv_total_kwh = 0.0
        pv_upload = None

    st.subheader("🔋 Batterie")
    batt_kwh = st.number_input("Capacité (kWh)", min_value=0.0, value=225.0, step=10.0, format="%.0f")
    batt_kw = st.number_input("Puissance (kW)", min_value=0.0, value=110.0, step=10.0, format="%.0f")
    eff_rt = st.number_input("Rendement (%)", min_value=50.0, max_value=100.0, value=90.0, step=1.0, format="%.0f") / 100
    dod = st.number_input("Profondeur de décharge (%)", min_value=10.0, max_value=100.0, value=90.0, step=1.0, format="%.0f") / 100

    st.subheader("💰 Paramètres économiques")
    capex_pv_per_kwc = st.number_input("CAPEX PV (CHF/kWc)", min_value=0.0, value=900.0, step=10.0, format="%.0f")
    capex_batt_per_kwh = st.number_input("CAPEX Batterie (CHF/kWh)", min_value=0.0, value=350.0, step=10.0, format="%.0f")
    years = st.number_input("Durée de simulation (ans)", min_value=1.0, max_value=30.0, value=20.0, step=1.0, format="%.0f")
    disc = st.number_input("Taux d'actualisation (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1) / 100

    st.subheader("🔌 Tarification & services")
    if marche_libre == "Non":
        price_buy_fixed = st.number_input("Tarif achat (CHF/kWh)", min_value=0.0, value=0.229, step=0.01)
        price_sell_fixed = st.number_input("Tarif revente PV (CHF/kWh)", min_value=0.0, value=0.053, step=0.01)
    else:
        price_buy_fixed = price_sell_fixed = None
    peak_tariff = st.number_input("Tarif puissance (CHF/kW/an)", min_value=0.0, value=150.0, step=5.0, format="%.0f")
    services_kw = st.number_input("Puissance contractée (kW)", min_value=0.0, value=0.0, step=10.0, format="%.0f")
    services_tariff = st.number_input("Rémunération services (CHF/kW/an)", min_value=0.0, value=40.0, step=1.0, format="%.0f")

# -----------------------------
# Profils (conso & PV)
# -----------------------------
idx = year_hours(2024)

# ✅ Toujours utiliser consum_kW (qu'il vienne du CSV ou du profil synthétique)
load = consum_kW.copy()

# ✅ Harmonisation du pas de temps sur l’année complète
load = load.reindex(idx, method="nearest")


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
# Simulation + dispatch (distinction charge PV / charge réseau)
# -----------------------------
def simulate_dispatch(load, pv, prices, cap_kwh, p_kw, eff_rt, dod, market_free):
    n = len(load)
    soc = 0.5 * cap_kwh
    soc_min, soc_max = (1-dod)*cap_kwh, cap_kwh

    charged = np.zeros(n)
    discharged = np.zeros(n)
    charged_from_pv = np.zeros(n)
    charged_from_grid = np.zeros(n)

    rev_auto = np.zeros(n)  # évite achat
    rev_arb = np.zeros(n)   # arbitrage

    if market_free:
        low, high = np.quantile(prices, [0.25, 0.75])
    else:
        low = high = None

    for i in range(n):
        pv_h, load_h, price = pv[i], load[i], prices[i]
        net = load_h - pv_h

        # PV direct + charge PV si surplus
        if net < -1e-12:
            surplus = -net
            charge_pv = min(surplus, p_kw, soc_max - soc)
            soc += charge_pv * eff_rt
            charged[i] += charge_pv
            charged_from_pv[i] += charge_pv
        else:
            # décharge pour couvrir déficit
            deficit = net
            if deficit > 1e-12:
                discharge = min(deficit, p_kw, soc - soc_min)
                delivered = discharge * eff_rt
                soc -= discharge
                discharged[i] += delivered
                rev_auto[i] = delivered * price

        # Arbitrage
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

# Convert to Series for masking operations
charged_s = pd.Series(charged, index=idx)
discharged_s = pd.Series(discharged, index=idx)

# -----------------------------
# PV split & sources d'énergie
# -----------------------------
pv_self_no_bess = np.minimum(pv, load)                  # sans batterie
grid_to_load_no_bess = (load - pv_self_no_bess).clip(lower=0)

pv_self = np.minimum(pv, load)                          # avec batterie (direct PV→charge)
pv_to_batt = charged_from_pv.clip(min=0)
pv_export = np.maximum(0.0, pv - pv_self - pv_to_batt)

bess_to_load = discharged_s                             # batterie → charge du bâtiment
grid_to_load = (load - pv_self - bess_to_load).clip(lower=0)

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

# Cashflow cumulé (actualisé)
years_int = int(round(years))
cashflows = [-capex_total] + [total_rev]*(years_int)
cum_discounted = []
acc = 0.0
for i, cf in enumerate(cashflows):
    acc += cf / ((1+disc)**i)
    cum_discounted.append(acc)
cum_years = list(range(0, years_int+1))

# -----------------------------
# Résultats
# -----------------------------
st.subheader("📊 Résultats")

# Détail revenus + Cashflow (graphiques réduits et côte à côte)
left_rev, right_cf = st.columns([1.2, 1.0])
with left_rev:
    st.markdown("### 💰 Détail des revenus (CHF/an)")

    # Renomme la colonne et formate les valeurs
    rev_df = pd.DataFrame(revenus.items(), columns=["Source", "Revenus (CHF/an)"])
    rev_df["Revenus (CHF/an)"] = rev_df["Revenus (CHF/an)"].astype(float).round(0)

    # Tableau propre, centré, redimensionnable
    st.data_editor(
        rev_df,
        hide_index=True,              # enlève la colonne 0,1,2...
        use_container_width=True,     # garde l'alignement avec le Cashflow à droite
        column_config={
            "Source": st.column_config.TextColumn(
                "Source",
                width="medium"        # tu peux mettre "small", "medium", "large" ou ex: width=140
            ),
            "Revenus (CHF/an)": st.column_config.NumberColumn(
                "Revenus (CHF/an)",
                format="%.0f CHF",    # format financier
                width="small",        # colonne compacte & redimensionnable à la souris
                help="Revenus annuels générés par cette source"
            ),
        },
    )



with right_cf:
    st.markdown("### 💵 Cashflow cumulé")
    plt.rcParams.update({
    "font.size": 6,
    "axes.labelsize": 6,
    "axes.titlesize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    })
    fig, ax = plt.subplots(figsize=(3,1.5))  # 50% plus petit
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)   # bordures fines
        spine.set_color("#CCCCCC") # gris doux
    ax.plot(cum_years, cum_discounted, linewidth=1.3, color=COLORS["bess_charge"])
    ax.axhline(0, color="#CCCCCC", linewidth=0.8, linestyle="--")
    # Graduation tous les 50k CHF
    ymin, ymax = ax.get_ylim()
    step = 50000
    ax.set_yticks(np.arange(
    round(ymin / step) * step,
    round(ymax / step) * step + step,
    step
    ))
    # Format des nombres en ordonnée avec séparateur de milliers
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ",")))


    ax.set_xlabel("Années", fontsize=4.5, color=COLORS["text"])
    ax.set_ylabel("CHF (actualisés)", fontsize=4.5, color=COLORS["text"])
    ax.tick_params(axis="both", labelsize=4.5)
    st.image(fig_to_svg(fig), use_container_width=True)




# En-tête métriques
m1, m2 = st.columns(2)
m1.metric("Revenu annuel total", f"{total_rev:,.0f} CHF")
m2.metric("CAPEX total", f"{capex_total:,.0f} CHF")

# ---------- Ligne 1 (2x2) : PV split + Autoconso table ----------
row1_col1, row1_col2 = st.columns([1, 1])
with row1_col1:
    st.markdown("#### ☀️ Répartition de production PV")

    fig, ax = plt.subplots(figsize=(2,2), dpi=300)

    labels = ["Bâtiment", "BESS", "Export"]
    sizes = [pv_self.sum(), pv_to_batt.sum(), pv_export.sum()]
    if sum(sizes) <= 0: sizes = [1,0,0]
    colors = [COLORS["pv"], COLORS["bess_charge"], COLORS["grid_export"]]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=lambda p: f"{p:.0f}%",
        startangle=90,
        colors=colors,
        radius=0.8
    )


    for t in texts:
        t.set_fontsize(6)
    for t in autotexts:
        t.set_fontsize(6)


    ax.set_title("")
    ax.set_aspect('equal', adjustable='datalim')
    fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.12)


    svg = fig_to_svg(fig)
    st.image(svg, width=500)

with row1_col2:
    st.markdown("#### 🏢 Autoconsommation du bâtiment")

    ac_df = pd.DataFrame({
        "Scénario": ["PV sans BESS", "PV avec BESS"],
        "Autoconsommation (%)": [autoconso_no_bess, autoconso_with_bess]
    })

    # Formater proprement les % + centrer
    ac_df["Autoconsommation (%)"] = ac_df["Autoconsommation (%)"].round(0).astype(int).astype(str) + " %"

    st.dataframe(
        ac_df.style.set_properties(
            subset=["Autoconsommation (%)"],
            **{"text-align": "center"}
        ),
        hide_index=True,
        use_container_width=True
    )


# ---------- Ligne 2 (2x2) : Sources d'énergie sans/avec BESS ----------
row2_col1, row2_col2 = st.columns([1, 1])
with row2_col1:
    st.markdown("#### 🔌 Sources d'énergie du bâtiment — PV seul")

    fig, ax = plt.subplots(figsize=(2,2), dpi=300)

    labels = ["PV", "Réseau"]
    sizes = [pv_self_no_bess.sum(), grid_to_load_no_bess.sum()]
    if sum(sizes) <= 0:
        sizes = [1, 0]
    colors = [COLORS["pv"], COLORS["grid_import"]]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=lambda p: f"{p:.0f}%",
        startangle=90,
        colors=colors,
        radius=0.8
    )

    for t in texts:
        t.set_fontsize(6)
    for t in autotexts:
        t.set_fontsize(6)

    ax.set_title("")
    ax.set_aspect('equal', adjustable='datalim')
    fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.12)


    svg = fig_to_svg(fig)
    st.image(svg, width=500)


with row2_col2:
    st.markdown("#### 🔋 Sources d'énergie du bâtiment — PV + BESS")

    fig, ax = plt.subplots(figsize=(2,2), dpi=300)

    labels = ["PV", "BESS", "Réseau"]
    sizes = [pv_self.sum(), bess_to_load.sum(), grid_to_load.sum()]
    if sum(sizes) <= 0:
        sizes = [1, 0, 0]
    colors = [COLORS["pv"], COLORS["bess_discharge"], COLORS["grid_import"]]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=lambda p: f"{p:.0f}%",
        startangle=90,
        colors=colors,
        radius=0.8
    )

    for t in texts:
        t.set_fontsize(6)
    for t in autotexts:
        t.set_fontsize(6)

    ax.set_title("")
    ax.set_aspect('equal', adjustable='datalim')
    fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.12)


    svg = fig_to_svg(fig)
    st.image(svg, width=500)



# ---------- Profils été/hiver (moyennes mensuelles) ----------
if ("bâtiment" in system_type.lower()) and has_pv and (batt_kwh > 0) and (batt_kw > 0):
    st.markdown("### 📈 Profils — Profils moyens été / hiver")

    # Été = Juillet
    ls = load[load.index.month == 7]
    conso_summer = ls.groupby(ls.index.time).mean()

    ps = pv[pv.index.month == 7]
    pv_summer = ps.groupby(ps.index.time).mean()

    cs = bess_charge[bess_charge.index.month == 7]
    charge_summer = cs.groupby(cs.index.time).mean()

    ds = bess_discharge[bess_discharge.index.month == 7]
    discharge_summer = ds.groupby(ds.index.time).mean()

    # Hiver = Décembre
    lw = load[load.index.month == 12]
    conso_winter = lw.groupby(lw.index.time).mean()

    pw = pv[pv.index.month == 12]
    pv_winter = pw.groupby(pw.index.time).mean()

    cw = bess_charge[bess_charge.index.month == 12]
    charge_winter = cw.groupby(cw.index.time).mean()

    dw = bess_discharge[bess_discharge.index.month == 12]
    discharge_winter = dw.groupby(dw.index.time).mean()

    # 🔄 Convertir index -> datetime pour tracé Matplotlib
    import pandas as pd
    for s in [conso_summer, conso_winter, pv_summer, pv_winter, charge_summer, discharge_summer, charge_winter, discharge_winter]:
        s.index = pd.to_datetime(s.index.astype(str))

    # --- Graphes Conso vs PV ---
    r3c1, r3c2 = st.columns(2)

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(conso_summer.index, conso_summer.values, label="Conso (kW)", color=COLORS["load"], linewidth=1.8)
    ax.plot(pv_summer.index, pv_summer.values, label="PV (kW)", color=COLORS["pv"], linewidth=1.6)
    ax.fill_between(conso_summer.index, 0, np.minimum(conso_summer.values, pv_summer.values), alpha=0.18, color=COLORS["pv"])
    ax.set_title("Été — Profil moyen (juillet)")
    ax.legend()
    r3c1.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(8,3))
    ax2.plot(conso_winter.index, conso_winter.values, label="Conso (kW)", color=COLORS["load"], linewidth=1.8)
    ax2.plot(pv_winter.index, pv_winter.values, label="PV (kW)", color=COLORS["pv"], linewidth=1.6)
    ax2.fill_between(conso_winter.index, 0, np.minimum(conso_winter.values, pv_winter.values), alpha=0.18, color=COLORS["pv"])
    ax2.set_title("Hiver — Profil moyen (décembre)")
    ax2.legend()
    r3c2.pyplot(fig2)

    # --- Graphes Batterie ---
    r4c1, r4c2 = st.columns(2)

    fig3, ax3 = plt.subplots(figsize=(8,3))
    ax3.bar(charge_summer.index, charge_summer.values, color=COLORS["bess_charge"], alpha=0.8, label="Charge (kW)")
    ax3.bar(discharge_summer.index, -discharge_summer.values, color=COLORS["bess_discharge"], alpha=0.8, label="Décharge (kW)")
    ax3.set_title("Flux batterie — Été (juillet)")
    ax3.legend()
    r4c1.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(8,3))
    ax4.bar(charge_winter.index, charge_winter.values, color=COLORS["bess_charge"], alpha=0.8, label="Charge (kW)")
    ax4.bar(discharge_winter.index, -discharge_winter.values, color=COLORS["bess_discharge"], alpha=0.8, label="Décharge (kW)")
    ax4.set_title("Flux batterie — Hiver (décembre)")
    ax4.legend()
    r4c2.pyplot(fig4)



# ---------- Peak shaving annuel (si marché libre) ----------
if ("bâtiment" in system_type.lower()) and (marche_libre == "Oui"):
    st.markdown("### 🏔️ Peak shaving sur l'année (pics mensuels)")
    month_index = pd.Index(idx).month
    before_monthly_peak = pd.Series(net_before.values, index=idx).groupby(month_index).max()
    after_monthly_peak = pd.Series(net_after.values, index=idx).groupby(month_index).max()

    fig, ax = plt.subplots(figsize=(10,3))
    ax.bar(before_monthly_peak.index-0.2, before_monthly_peak.values, width=0.4, label="Avant BESS", color=COLORS["grid_import"])
    ax.bar(after_monthly_peak.index+0.2, after_monthly_peak.values, width=0.4, label="Après BESS", color=COLORS["bess_charge"])
    ax.set_xlabel("Mois"); ax.set_ylabel("kW")
    ax.set_xticks(range(1,13))
    ax.set_title("Pics mensuels de puissance (avant / après)", color=COLORS["text"])
    ax.legend()
    st.pyplot(fig)

st.caption(
    "Remarques : Les prix ENTSO-E (Swissgrid) sont utilisés si la clé API est fournie. "
    "Les fichiers CSV doivent contenir 8760 valeurs horaires (kWh/h) sans en-tête. "
    "Charte : PV=#FFEE8C, BESS charge=#62A9C6, BESS décharge=#4B94B0, Conso=#B7D9B1, Import réseau=#F3C77A, Export réseau=#E2B007."
)
