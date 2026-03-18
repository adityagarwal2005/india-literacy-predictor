import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

st.set_page_config(page_title="India Literacy Predictor", page_icon="📚", layout="wide")

# ── Global font size for all matplotlib charts ──────────────
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_raw():
    return pd.read_csv("indian.csv")

@st.cache_data
def load_data():
    return pd.read_csv("districts_with_features.csv")

payload  = load_model()
model    = payload["model"]
FEATURES = payload["features"]
metrics  = payload["metrics"]
df       = load_data()
raw      = load_raw()

CLASS_COLORS = {"Low": "#E53935", "Medium": "#FF9800", "High": "#4CAF50"}
CLASS_EMOJI  = {"Low": "🔴", "Medium": "🟡", "High": "🟢"}
CLASS_RANGE  = {"Low": "< 60%", "Medium": "60–80%", "High": "> 80%"}
CLASS_DESC   = {
    "Low":    "This district has serious literacy challenges. Fewer than 60% of people are literate. Significant education investment is needed.",
    "Medium": "This district is around the national average. 60–80% literacy. There is room for improvement.",
    "High":   "This is a well-educated district. Over 80% of people are literate — well above the national average."
}

# Friendly feature name mapping
FEATURE_NAMES = {
    "gender_ratio":  "Gender Ratio (F÷M)",
    "hindu_pct":     "Hindu %",
    "muslim_pct":    "Muslim %",
    "christian_pct": "Christian %",
    "sikh_pct":      "Sikh %",
    "buddhist_pct":  "Buddhist %",
    "youth_pct":     "Youth % (age 0–29)",
    "working_pct":   "Working Age % (30–49)",
    "elderly_pct":   "Elderly % (50+)",
    "pop_log":       "Population (log scale)",
}

st.title("📚 Indian District — Literacy Level Predictor")
st.markdown("**Census 2011 · 640 districts · Gradient Boosting Classifier**  \nPredicts whether a district is Low / Medium / High literacy using religion, gender ratio and age groups — without using literacy data directly.")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Explore", "🤖 Model", "📈 EDA Charts"])

# ══════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════
with tab1:
    st.subheader("Predict Literacy Level of a District")
    mode = st.radio("Input mode", ["🏙️ Pick existing district", "✏️ Enter custom values"], horizontal=True)

    if mode == "🏙️ Pick existing district":
        districts = sorted(df["District name"].unique())
        selected  = st.selectbox("Select District", districts, index=districts.index("Mumbai") if "Mumbai" in districts else 0)
        row = df[df["District name"] == selected].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("District", selected)
        c2.metric("State", row["State name"])
        c3.metric("Actual Literacy", f"{row['literacy_rate']*100:.1f}%")
        X_input      = pd.DataFrame([row[FEATURES]])
        actual_class = str(row["literacy_class"])
    else:
        st.markdown("#### Enter demographic values")
        c1, c2 = st.columns(2)
        with c1:
            gender_ratio  = st.slider("Gender ratio (F÷M)", 0.50, 1.20, 0.94, 0.01)
            youth_pct     = st.slider("Youth % — age 0–29", 40, 75, 59) / 100
            working_pct   = st.slider("Working age % — 30 to 49 years", 15, 35, 25) / 100
            elderly_pct   = st.slider("Elderly % — 50+ years", 5, 35, 15) / 100
            pop_log       = np.log1p(st.number_input("Population", 10000, 12000000, 1000000, 50000))
        with c2:
            hindu_pct     = st.slider("Hindu %", 0, 100, 70) / 100
            muslim_pct    = st.slider("Muslim %", 0, 100, 15) / 100
            christian_pct = st.slider("Christian %", 0, 100, 5) / 100
            sikh_pct      = st.slider("Sikh %", 0, 100, 2) / 100
            buddhist_pct  = st.slider("Buddhist %", 0, 100, 1) / 100
        X_input      = pd.DataFrame([{"gender_ratio": gender_ratio, "hindu_pct": hindu_pct,
            "muslim_pct": muslim_pct, "christian_pct": christian_pct, "sikh_pct": sikh_pct,
            "buddhist_pct": buddhist_pct, "youth_pct": youth_pct, "working_pct": working_pct,
            "elderly_pct": elderly_pct, "pop_log": pop_log}])
        actual_class = None

    st.divider()
    if st.button("🔮 Predict Literacy Level", type="primary", use_container_width=True):
        pred    = model.predict(X_input[FEATURES])[0]
        proba   = model.predict_proba(X_input[FEATURES])[0]
        classes = list(model.classes_)
        color   = CLASS_COLORS[pred]

        # ── Big result card ──────────────────────────────────
        st.markdown(f"""
        <div style='background:{color}15; border:2px solid {color};
                    padding:28px 24px; border-radius:14px; margin:14px 0; text-align:center'>
            <div style='font-size:52px; margin-bottom:6px'>{CLASS_EMOJI[pred]}</div>
            <div style='font-size:32px; font-weight:700; color:{color}'>{pred} Literacy District</div>
            <div style='font-size:18px; color:{color}; margin:6px 0'>Literacy range: {CLASS_RANGE[pred]}</div>
            <div style='font-size:15px; color:#555; margin-top:10px; max-width:600px; margin-left:auto; margin-right:auto'>
                {CLASS_DESC[pred]}
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Correct / wrong badge ────────────────────────────
        if actual_class:
            if actual_class == pred:
                st.success(f"✅ Model is correct! Actual class is also **{actual_class}**.")
            else:
                st.warning(f"⚠️ Model predicted **{pred}** but actual class is **{actual_class}**. The model is correct 73% of the time.")

        st.divider()

        # ── Three confidence cards side by side ──────────────
        st.markdown("#### How confident is the model?")
        cols = st.columns(3)
        order = ["Low","Medium","High"]
        for i, cls in enumerate(order):
            idx = classes.index(cls)
            p   = proba[idx]
            c   = CLASS_COLORS[cls]
            is_pred = cls == pred
            border = f"3px solid {c}" if is_pred else f"1px solid {c}44"
            bg     = f"{c}18" if is_pred else "transparent"
            cols[i].markdown(f"""
            <div style='background:{bg}; border:{border}; border-radius:12px;
                        padding:20px; text-align:center'>
                <div style='font-size:28px'>{CLASS_EMOJI[cls]}</div>
                <div style='font-size:18px; font-weight:600; color:{c}'>{cls}</div>
                <div style='font-size:36px; font-weight:700; color:{c}'>{p*100:.1f}%</div>
                <div style='font-size:13px; color:#888'>{CLASS_RANGE[cls]}</div>
                {"<div style='font-size:12px; color:" + c + "; margin-top:6px'>← predicted</div>" if is_pred else ""}
            </div>""", unsafe_allow_html=True)

        # ── Interesting facts about this district ────────────
        st.divider()
        st.markdown("#### 📌 District snapshot")
        if mode == "🏙️ Pick existing district":
            r = raw2[raw2["District name"] == selected].iloc[0] if selected in raw2["District name"].values else None
            if r is not None:
                dom_religion = ["Hindus","Muslims","Christians","Sikhs","Buddhists"]
                dom = max(dom_religion, key=lambda x: r[x])
                dom_pct = r[dom] / r["Population"] * 100
                lit_pct = r["literate education"] / r["Population"] * 100
                gender  = r["Female"] / r["Male"]
                youth   = r["age 0-29"] / r["Population"] * 100
                elderly = r["age 50>"] / r["Population"] * 100

                f1, f2, f3, f4, f5 = st.columns(5)
                f1.metric("Population",       f"{int(r['Population']):,}")
                f2.metric("Literacy Rate",    f"{lit_pct:.1f}%")
                f3.metric("Dominant Religion",f"{dom} ({dom_pct:.0f}%)")
                f4.metric("Gender Ratio",     f"{gender:.3f}")
                f5.metric("Youth (0–29)",     f"{youth:.1f}%")

                # compare vs national average
                st.markdown("#### 📊 How does this district compare to national average?")
                nat_lit    = raw2["literate education"].sum() / raw2["Population"].sum() * 100
                nat_gender = (raw2["Female"].sum() / raw2["Male"].sum())
                nat_youth  = raw2["age 0-29"].sum() / raw2["Population"].sum() * 100

                fig_cmp, axes = plt.subplots(1, 3, figsize=(12, 3.5))
                metrics_cmp = [
                    ("Literacy Rate (%)", lit_pct, nat_lit),
                    ("Gender Ratio",      gender,  nat_gender),
                    ("Youth % (0–29)",    youth,   nat_youth),
                ]
                for ax, (title, dist_val, nat_val) in zip(axes, metrics_cmp):
                    bars_cmp = ax.bar([selected, "National Avg"],
                                      [dist_val, nat_val],
                                      color=["#2196F3","#BDBDBD"],
                                      edgecolor="white", width=0.5)
                    for b in bars_cmp:
                        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002*nat_val,
                                f"{b.get_height():.2f}", ha="center", fontsize=12, fontweight="bold")
                    ax.set_title(title, fontsize=13, fontweight="bold")
                    ax.spines[["top","right"]].set_visible(False)
                    ax.tick_params(labelsize=11)
                plt.tight_layout()
                st.pyplot(fig_cmp); plt.close()
        else:
            st.info("Pick an existing district to see its snapshot and comparison vs national average.")

# ══════════════════════════════════════════
# TAB 2 — EXPLORE
# ══════════════════════════════════════════
with tab2:
    st.subheader("Explore the Dataset")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Districts", "640")
    c2.metric("🔴 Low literacy",    f"{(df['literacy_class']=='Low').sum()} districts")
    c3.metric("🟡 Medium literacy", f"{(df['literacy_class']=='Medium').sum()} districts")
    c4.metric("🟢 High literacy",   f"{(df['literacy_class']=='High').sum()} districts")
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Class distribution")
        counts = df["literacy_class"].value_counts()[["Low","Medium","High"]]
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        bars = ax1.bar(counts.index, counts.values,
                       color=[CLASS_COLORS[c] for c in counts.index],
                       edgecolor="white", width=0.5)
        for b in bars:
            ax1.text(b.get_x()+b.get_width()/2, b.get_height()+4,
                     str(int(b.get_height())), ha="center", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Number of Districts", fontsize=13)
        ax1.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig1); plt.close()

    with col_r:
        st.markdown("#### Gender ratio vs literacy rate")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        for cls, grp in df.groupby("literacy_class"):
            ax2.scatter(grp["gender_ratio"], grp["literacy_rate"]*100,
                        label=cls, color=CLASS_COLORS[str(cls)], alpha=0.6, s=25)
        ax2.set_xlabel("Gender ratio (F÷M)")
        ax2.set_ylabel("Literacy rate (%)")
        ax2.legend(title="Literacy Class", fontsize=12)
        ax2.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.divider()
    st.markdown("#### Top 10 most literate districts")
    top = df.nlargest(10,"literacy_rate")[["District name","State name","literacy_rate","literacy_class"]].copy()
    top["literacy_rate"] = (top["literacy_rate"]*100).round(1).astype(str)+"%"
    top.columns = ["District","State","Literacy Rate","Class"]
    st.dataframe(top, use_container_width=True, hide_index=True)

    st.markdown("#### Bottom 10 least literate districts")
    bot = df.nsmallest(10,"literacy_rate")[["District name","State name","literacy_rate","literacy_class"]].copy()
    bot["literacy_rate"] = (bot["literacy_rate"]*100).round(1).astype(str)+"%"
    bot.columns = ["District","State","Literacy Rate","Class"]
    st.dataframe(bot, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════
# TAB 3 — MODEL
# ══════════════════════════════════════════
with tab3:
    st.subheader("Model Details")
    c1,c2,c3 = st.columns(3)
    c1.metric("Algorithm",   "Gradient Boosting Classifier")
    c2.metric("Accuracy",    f"{metrics['accuracy']*100:.1f}%")
    c3.metric("CV Accuracy", f"{metrics['cv_accuracy']*100:.1f}%")
    st.divider()

    st.markdown("""
    #### Why 73% and not 99%?
    The previous model had 99% because the answer (youth ratio) was directly calculable from the CSV.
    Here, literacy genuinely cannot be calculated from religion, gender and age alone — they are only hints.
    **73% means the model correctly classifies ~9 out of every 13 districts.** That is real — random guessing would only give 33%.
    """)

    st.divider()
    st.markdown("#### Feature importances — what matters most?")

    imp = pd.Series(payload["feature_importances"]).sort_values()
    imp.index = [FEATURE_NAMES.get(f, f) for f in imp.index]

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    clrs3 = ["#E53935" if i == len(imp)-1 else "#2196F3" for i in range(len(imp))]
    imp.plot(kind="barh", ax=ax3, color=clrs3, edgecolor="white")
    ax3.set_xlabel("Importance Score")
    ax3.set_title("Feature Importances", fontsize=16, fontweight="bold")
    ax3.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    st.markdown("""
    **Working Age % (30–49)** is the strongest predictor by far.
    Districts where a higher proportion are in their prime working years tend to be more educated.
    Gender ratio and religion composition also contribute meaningfully.
    """)

    st.divider()
    st.markdown("""
    #### Methodology
    - **Target:** Low (< 60%) / Medium (60–80%) / High (> 80%) literacy
    - **Features:** Gender ratio, religion %, age group %, log(population) — no literacy columns used
    - **Model:** Gradient Boosting Classifier · 200 trees · learning rate 0.1 · max depth 5
    - **Split:** 80/20 train-test, stratified · CV done on training set only (correct method)
    - **Data:** Census of India 2011, 640 districts
    """)

# ══════════════════════════════════════════
# TAB 4 — EDA CHARTS
# ══════════════════════════════════════════
with tab4:
    st.subheader("📈 Exploratory Data Analysis — Census 2011")
    st.markdown("All charts built during the exploration phase. Data: 640 Indian districts.")
    st.divider()

    raw2 = raw.copy()
    raw2["literacy_rate"] = raw2["literate education"] / raw2["Population"]
    raw2["gender_ratio"]  = raw2["Female"] / raw2["Male"]
    raw2["Male_pct"]      = raw2["Male"] / raw2["Population"] * 100
    raw2["Female_pct"]    = raw2["Female"] / raw2["Population"] * 100
    religion_cols = ["Hindus","Muslims","Christians","Sikhs","Buddhists","Jains"]
    for col in religion_cols:
        raw2[col+"_pct"] = raw2[col] / raw2["Population"] * 100

    # ── CHART 1: Religion heatmap ────────────────────────────
    st.markdown("### 1. Religion-wise Population Distribution — Top 50 Districts")
    top50    = raw2.nlargest(50,"Population")
    rel_pcts = [c+"_pct" for c in religion_cols]
    heat     = top50.set_index("District name")[rel_pcts]
    heat.columns = religion_cols

    fig_h, ax_h = plt.subplots(figsize=(12, 16))
    sns.heatmap(heat, cmap="YlGnBu", ax=ax_h, linewidths=0.3,
                cbar_kws={"label":"Population %"},
                annot=False)
    ax_h.set_title("Religion-wise Population Distribution\nTop 50 Indian Districts",
                   fontsize=16, fontweight="bold", pad=16)
    ax_h.set_xlabel("Religion", fontsize=14)
    ax_h.set_ylabel("District", fontsize=14)
    ax_h.tick_params(axis="both", labelsize=11)
    plt.tight_layout()
    st.pyplot(fig_h); plt.close()
    st.divider()

    # ── CHART 2: Gender pyramid ──────────────────────────────
    st.markdown("### 2. Gender-wise Population Distribution — Major Districts")
    popular = ["Mumbai","New Delhi","Bangalore","Chennai","Hyderabad","Pune","Kolkata",
               "Ahmedabad","Jaipur","Surat","Agra","Lucknow","Srinagar","Kanpur",
               "Indore","Bhopal","Patna","Vadodara","Nagpur","Ludhiana",
               "Amritsar","Noida","Gurgaon","Faridabad"]
    df_g = raw2[raw2["District name"].isin(popular)].copy()
    df_g["District name"] = pd.Categorical(df_g["District name"], categories=popular, ordered=True)
    df_g = df_g.sort_values("District name")

    fig_g, ax_g = plt.subplots(figsize=(12, 9))
    ax_g.barh(df_g["District name"],  df_g["Male_pct"],   color="#4C72B0", height=0.5, label="Male")
    ax_g.barh(df_g["District name"], -df_g["Female_pct"], color="#DD8452", height=0.5, label="Female")
    ax_g.axvline(0, color="gray", linewidth=1)
    for _, row in df_g.iterrows():
        ax_g.text( row["Male_pct"]+0.3,   row["District name"], f"{row['Male_pct']:.1f}%",   va="center", ha="left",  fontsize=9, color="white", fontweight="bold")
        ax_g.text(-row["Female_pct"]-0.3, row["District name"], f"{row['Female_pct']:.1f}%", va="center", ha="right", fontsize=9, color="white", fontweight="bold")
    max_pct = max(df_g["Male_pct"].max(), df_g["Female_pct"].max())
    ax_g.set_xlim(-max_pct-5, max_pct+5)
    ax_g.set_xlabel("Population Share (%)")
    ax_g.set_title("Gender-wise Population Distribution\nMajor Indian Districts", fontsize=16, fontweight="bold", pad=16)
    ax_g.legend(loc="upper center", bbox_to_anchor=(0.5,1.04), ncol=2, frameon=False, fontsize=13)
    ax_g.grid(axis="x", linestyle="--", alpha=0.4)
    for spine in ["top","right","left"]:
        ax_g.spines[spine].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_g); plt.close()
    st.divider()

    # ── CHART 3: Age composition Mumbai vs Delhi ─────────────
    st.markdown("### 3. Literacy Rate vs Population — All 640 Districts (Density)")
    valid = raw2[raw2["Population"]>0].copy()
    fig8, ax8 = plt.subplots(figsize=(10,7))
    hb = ax8.hexbin(valid["literate education"], valid["Population"],
                    gridsize=30, cmap="viridis", mincnt=1, yscale="log")
    cb = fig8.colorbar(hb, ax=ax8)
    cb.set_label("Number of Districts", fontsize=13)
    ax8.set_xlabel("Literate Population (absolute count)")
    ax8.set_ylabel("Total Population (log scale)")
    ax8.set_title("Literacy vs Population Density\nAll 640 Indian Districts", fontsize=16, fontweight="bold")
    ax8.text(0.02, 0.97, f"Districts: {len(valid)}", transform=ax8.transAxes,
             fontsize=11, va="top", color="gray")
    plt.tight_layout()
    st.pyplot(fig8); plt.close()

st.divider()
st.caption("Built with Streamlit · Census 2011 · Gradient Boosting Classifier · scikit-learn")
    st.markdown("### 4. Literacy vs Population — Major Indian Districts")
    major = ["Mumbai","New Delhi","Bangalore","Chennai","Hyderabad","Pune","Kolkata",
             "Ahmedabad","Jaipur","Surat","Agra","Lucknow","Srinagar","Indore",
             "Bhopal","Patna","Vadodara","Nagpur","Ludhiana","Amritsar","Gurgaon","Faridabad"]
    df7 = raw2[raw2["District name"].isin(major)].copy()
    df7["Pop_millions"] = df7["Population"]/1e6

    fig7, ax7 = plt.subplots(figsize=(11,7))
    ax7.scatter(df7["literacy_rate"]*100, df7["Pop_millions"],
                color="#4C72B0", alpha=0.75, s=80, edgecolors="white", linewidths=0.8)
    for _, row in df7.iterrows():
        ax7.annotate(row["District name"],
                     (row["literacy_rate"]*100, row["Pop_millions"]),
                     fontsize=9, xytext=(5,3), textcoords="offset points")
    ax7.set_xlabel("Literacy Rate (%)")
    ax7.set_ylabel("Population (Millions)")
    ax7.set_title("Literacy vs Population\nMajor Indian Districts", fontsize=16, fontweight="bold")
    ax7.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig7); plt.close()
    st.divider()

    # ── CHART 8: Hexbin all 640 ───────────────────────────────
    st.markdown("### 5. Age-wise Population Composition — Mumbai vs New Delhi")
    age_cols = ["age 0-29","age 30-49","age 50>"]
    df3 = raw2[raw2["District name"].isin(["Mumbai","New Delhi"])].copy()
    df3_pct = df3.set_index("District name")[age_cols].div(
        df3.set_index("District name")[age_cols].sum(axis=1), axis=0) * 100
    colors3 = ["#5B8FF9","#61DDAA","#F6BD16"]
    age_labels = ["0–29 years","30–49 years","50+ years"]

    fig3c, ax3c = plt.subplots(figsize=(8, 7))
    bottom = np.zeros(len(df3_pct))
    for i, (age, lbl) in enumerate(zip(age_cols, age_labels)):
        bars = ax3c.bar(df3_pct.index, df3_pct[age], bottom=bottom,
                        label=lbl, color=colors3[i], edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, df3_pct[age]):
            ax3c.text(bar.get_x()+bar.get_width()/2, bar.get_y()+bar.get_height()/2,
                      f"{val:.1f}%", ha="center", va="center", fontsize=13, color="white", fontweight="bold")
        bottom += df3_pct[age].values
    ax3c.set_ylim(0,100)
    ax3c.set_ylabel("Population Share (%)")
    ax3c.set_title("Age-wise Population Composition\nMumbai vs New Delhi", fontsize=16, fontweight="bold")
    ax3c.legend(title="Age Group", fontsize=12)
    ax3c.grid(axis="y", linestyle="--", alpha=0.25)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    st.pyplot(fig3c); plt.close()
    st.divider()

    # ── CHART 4: Age group bar Mumbai vs Pune ────────────────
    st.markdown("### 6. Age Group Distribution — Mumbai vs Pune")
    age_pct_cols = ["Age not stated","age 0-29","age 30-49","age 50>"]
    df4 = raw2[raw2["District name"].isin(["Mumbai","Pune"])].copy()
    for col in age_pct_cols:
        df4[col+" %"] = df4[col] / df4["Population"] * 100
    df_melt = df4.melt(id_vars="District name",
                       value_vars=[c+" %" for c in age_pct_cols],
                       var_name="age", value_name="percentage")
    df_melt["age"] = df_melt["age"].str.replace(" %","")
    from matplotlib.ticker import PercentFormatter
    sns.set_theme(style="whitegrid", context="talk")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_melt, x="age", y="percentage",
                hue="District name", palette={"Mumbai":"#1f77b4","Pune":"#ff7f0e"}, ax=ax4)
    for p in ax4.patches:
        if p.get_height() > 0.5:
            ax4.annotate(f"{p.get_height():.1f}%",
                         (p.get_x()+p.get_width()/2, p.get_height()+0.3),
                         ha="center", fontsize=11)
    ax4.yaxis.set_major_formatter(PercentFormatter())
    ax4.set_title("Age Group Distribution (% of Total Population)\nMumbai vs Pune", fontsize=16, fontweight="bold")
    ax4.set_xlabel("Age Group")
    ax4.set_ylabel("Population Percentage (%)")
    ax4.legend(title="District", fontsize=12)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    st.pyplot(fig4); plt.close()
    st.divider()

    # ── CHART 5: Religion Jaipur vs Srinagar ─────────────────
    st.markdown("### 7. Hindu–Muslim Population Share — Jaipur vs Srinagar")
    df5 = raw2[raw2["District name"].isin(["Jaipur","Srinagar"])].copy()
    melt5 = df5.melt(id_vars="District name", value_vars=["Hindus","Muslims"],
                     var_name="Religion", value_name="Pop")
    total5 = melt5.groupby("District name")["Pop"].transform("sum")
    melt5["Percentage"] = melt5["Pop"] / total5 * 100

    sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)
    fig5, ax5 = plt.subplots(figsize=(9, 6))
    sns.barplot(data=melt5, x="Religion", y="Percentage", hue="District name",
                palette={"Jaipur":"#e75480","Srinagar":"#2e8b57"}, ax=ax5)
    for p in ax5.patches:
        if p.get_height() > 1:
            ax5.annotate(f"{p.get_height():.1f}%",
                         (p.get_x()+p.get_width()/2, p.get_height()+0.5),
                         ha="center", fontsize=12)
    ax5.yaxis.set_major_formatter(PercentFormatter())
    ax5.set_title("Religion-wise Population Share\nJaipur vs Srinagar", fontsize=16, fontweight="bold")
    ax5.set_xlabel("Religion")
    ax5.set_ylabel("Population Percentage (%)")
    ax5.legend(title="District", fontsize=12)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    st.pyplot(fig5); plt.close()
    st.divider()

    # ── CHART 6: Donut Thiruvananthapuram ────────────────────
    st.markdown("### 8. Religion-wise Distribution — Thiruvananthapuram")
    city6     = raw2[raw2["District name"] == "Thiruvananthapuram"].iloc[0]
    rel6_vals = [city6[c] for c in ["Hindus","Muslims","Christians","Sikhs"]]
    rel6_pct  = [v/city6["Population"]*100 for v in rel6_vals]
    rel6_lbls = [f"{n}\n{p:.1f}%" for n,p in zip(["Hindu","Muslim","Christian","Sikh"],rel6_pct)]

    fig6, ax6 = plt.subplots(figsize=(7,7))
    ax6.pie(rel6_vals, labels=rel6_lbls, colors=["#5B8FF9","#61DDAA","#F4664A","#FAAD14"],
            startangle=90, wedgeprops=dict(width=0.5),
            textprops=dict(fontsize=13))
    ax6.set_title("Religion-wise Population Distribution\nThiruvananthapuram",
                  fontsize=16, fontweight="bold", pad=20)
    ax6.text(0,0,"Census 2011\nPercentage Share", ha="center", va="center", fontsize=11, color="gray")
    plt.tight_layout()
    st.pyplot(fig6); plt.close()
    st.divider()

    # ── CHART 7: Scatter major districts ─────────────────────
