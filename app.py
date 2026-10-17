import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Player Value Analyzer", layout="wide")

@st.cache_data
def load():
    df = pd.read_csv("data/predictions.csv")
    df["market_M"] = df["market_value_in_eur"] / 1e6
    df["model_M"] = df["pred_value"] / 1e6
    df["caution"] = df["caution"].fillna("")
    return df

df = load()

st.title("Premier League Player Value Analyzer")
st.caption("2024/25 outfield players with 900+ minutes. Market value is Transfermarkt's, "
           "end of season. The model is a ridge regression on age, position and per-90 "
           "stats; the club is not an input.")

with st.sidebar:
    st.header("Filters")
    clubs = st.multiselect("Club", sorted(df["Club"].unique()))
    positions = st.multiselect("Position", sorted(df["Position"].unique()))
    lo, hi = int(df["age"].min()), int(df["age"].max()) + 1
    ages = st.slider("Age", lo, hi, (lo, hi))
    hide_flagged = st.checkbox("Hide players with a caution flag")
    view = st.radio("Gap measure", ["Compared with squad-mates", "Raw model vs market"])

f = df[df["age"].between(*ages)]
if clubs:
    f = f[f["Club"].isin(clubs)]
if positions:
    f = f[f["Position"].isin(positions)]
if hide_flagged:
    f = f[f["caution"] == ""]

gap_col = "gap_adj" if view.startswith("Compared") else "gap"
f = f.assign(diff_pct=(np.exp(f[gap_col]) - 1) * 100)

tab1, tab2, tab3 = st.tabs(["Ranking", "Player", "About"])

with tab1:
    st.write(f"{len(f)} players. Positive means the model values him above the market.")
    show = f.sort_values("diff_pct", ascending=False)[
        ["Player Name", "Club", "Position", "age", "market_M", "model_M", "diff_pct", "caution"]]
    st.dataframe(show, hide_index=True, column_config={
        "age": st.column_config.NumberColumn("Age", format="%.1f"),
        "market_M": st.column_config.NumberColumn("Market (€M)", format="%.1f"),
        "model_M": st.column_config.NumberColumn("Model (€M)", format="%.1f"),
        "diff_pct": st.column_config.NumberColumn("Model vs market (%)", format="%+.0f"),
        "caution": "Caution",
    })

with tab2:
    name = st.selectbox("Choose a player", sorted(df["Player Name"]))
    p = df[df["Player Name"] == name].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Market value", f"€{p['market_M']:.1f}M")
    c2.metric("Model value", f"€{p['model_M']:.1f}M")
    c3.metric("Model vs market", f"{(np.exp(p['gap']) - 1) * 100:+.0f}%")
    st.write(f"{p['Club']} · {p['Position']} · age {p['age']:.1f}")
    st.write(f"Compared with his squad-mates: {(np.exp(p['gap_adj']) - 1) * 100:+.0f}%")
    if p["caution"]:
        st.warning("Read with care: " + p["caution"].replace("_", " "))
    st.bar_chart(pd.DataFrame({"€M": [p["market_M"], p["model_M"]]}, index=["Market", "Model"]))

with tab3:
    st.markdown("""
**What this is.** A model trained on 292 outfield players from one season. It estimates a
player's market value from his age, position and per-90 stats, and the app shows where the
estimate and the real price disagree.

**How accurate is it?** Typical predictions are off by a factor of about 1.6, so only
large gaps mean much. Treat the ranking as a list of players worth a closer look.

**What it can't see.** Potential, injuries, contract length, reputation and commercial
appeal. That is why young players and superstars often look "overvalued", and why very old
players near the price floor look "undervalued". The caution column marks these cases.

**Limits.** One season, no goalkeepers, small sample, and the model was chosen by comparing
variants on the same players, so its accuracy is somewhat optimistic.
""")