import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

st.set_page_config(page_title="India Literacy Predictor", page_icon="📚", layout="wide")

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
CLASS_DESC   = {
    "Low":    "Below 60% literacy. Needs major education investment.",
    "Medium": "60–80% literacy. Close to national average.",
    "High":   "Above 80% literacy. Well above national average."
}

st.title("📚 Indian District — Literacy Level Predictor")
st.markdown("**Census 2011 · 640 districts · Gradient Boosting Classifier**  \nPredicts literacy class using religion %, gender ratio, age groups — without using literacy data directly.")
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
        c1, c2 = st.columns(2)
        with c1:
            gender_ratio  = st.slider("Gender ratio (F÷M)", 0.50, 1.20, 0.94, 0.01)
            youth_pct     = st.slider("Youth — age 0–29 (%)", 40, 75, 59) / 100
            working_pct   = st.slider("Working — 30–49 (%)", 15, 35, 25) / 100
            elderly_pct   = st.slider("Elderly — 50+ (%)", 5, 35, 15) / 100
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
        classes = model.classes_
        color   = CLASS_COLORS[pred]

        st.markdown(f"""
        <div style='background:{color}18;border-left:5px solid {color};padding:18px;border-radius:8px;margin:10px 0'>
            <h2 style='color:{color};margin:0'>{CLASS_EMOJI[pred]} {pred} Literacy District</h2>
            <p style='margin:6px 0 0;color:#666'>{CLASS_DESC[pred]}</p>
        </div>""", unsafe_allow_html=True)

        if actual_class:
            if actual_class == pred:
                st.success(f"✅ Correct! Actual class is also **{actual_class}**.")
            else:
                st.warning(f"⚠️ Actual = **{actual_class}**, Predicted = **{pred}**. Model is correct 73% of the time.")

        st.markdown("#### Confidence per class")
        fig, ax = plt.subplots(figsize=(6, 1.8))
        bars = ax.barh(classes, proba, color=[CLASS_COLORS[c] for c in classes], height=0.45, edgecolor="white")
        for b, p in zip(bars, proba):
            ax.text(b.get_width()+0.01, b.get_y()+b.get_height()/2, f"{p*100:.1f}%", va="center", fontsize=10)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Probability")
        ax.spines[["top","right","left"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════
# TAB 2 — EXPLORE
# ══════════════════════════════════════════
with tab2:
    st.subheader("Explore the Dataset")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Districts", "640")
    c2.metric("Low literacy",    f"{(df['literacy_class']=='Low').sum()}")
    c3.metric("Medium literacy", f"{(df['literacy_class']=='Medium').sum()}")
    c4.metric("High literacy",   f"{(df['literacy_class']=='High').sum()}")
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Class distribution")
        counts = df["literacy_class"].value_counts()[["Low","Medium","High"]]
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        bars = ax1.bar(counts.index, counts.values, color=[CLASS_COLORS[c] for c in counts.index], edgecolor="white", width=0.5)
        for b in bars:
            ax1.text(b.get_x()+b.get_width()/2, b.get_height()+3, str(int(b.get_height())), ha="center", fontsize=10)
        ax1.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig1); plt.close()

    with col_r:
        st.markdown("#### Gender ratio vs literacy rate")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        for cls, grp in df.groupby("literacy_class"):
            ax2.scatter(grp["gender_ratio"], grp["literacy_rate"]*100,
                        label=cls, color=CLASS_COLORS[str(cls)], alpha=0.5, s=18)
        ax2.set_xlabel("Gender ratio (F/M)")
        ax2.set_ylabel("Literacy rate (%)")
        ax2.legend(title="Class")
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
    73% means the model correctly classifies ~9 out of every 13 districts. Much better than random guessing (33%).
    """)

    st.divider()
    st.markdown("#### Feature importances")
    imp = pd.Series(payload["feature_importances"]).sort_values()
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    clrs3 = ["#E53935" if i == len(imp)-1 else "#2196F3" for i in range(len(imp))]
    imp.plot(kind="barh", ax=ax3, color=clrs3, edgecolor="white")
    ax3.set_xlabel("Importance score")
    ax3.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    st.markdown("""
    **working_pct** (30–49 age share) is the strongest signal.
    Districts with a higher working-age population tend to be more educated.
    Gender ratio and religion composition also contribute meaningfully.
    """)

    st.divider()
    st.markdown("""
    #### Methodology
    - **Target:** Low (<60%) / Medium (60–80%) / High (>80%) literacy
    - **Features:** Gender ratio, religion %, age group %, log(population)
    - **Not used:** literate education, Illiterate_Education columns (that would be cheating)
    - **Model:** Gradient Boosting Classifier · 200 trees · learning rate 0.1 · max depth 5
    - **Split:** 80/20 train-test, stratified · CV done on training set only
    - **Data:** Census of India 2011, 640 districts
    """)

# ══════════════════════════════════════════
# TAB 4 — EDA CHARTS
# ══════════════════════════════════════════
with tab4:
    st.subheader("📈 Exploratory Data Analysis — Census 2011")
    st.markdown("All charts built during the exploration phase of this project. Data: 640 Indian districts.")
    st.divider()

    # ── Prepare raw data ──────────────────────────────────────
    raw2 = raw.copy()
    raw2["literacy_rate"]  = raw2["literate education"] / raw2["Population"]
    raw2["gender_ratio"]   = raw2["Female"] / raw2["Male"]
    raw2["Male_pct"]       = raw2["Male"] / raw2["Population"] * 100
    raw2["Female_pct"]     = raw2["Female"] / raw2["Population"] * 100

    religion_cols = ["Hindus","Muslims","Christians","Sikhs","Buddhists","Jains"]
    for col in religion_cols:
        raw2[col+"_pct"] = raw2[col] / raw2["Population"] * 100

    # ── CHART 1: Religion heatmap top 50 ─────────────────────
    st.markdown("### 1. Religion-wise Population Distribution — Top 50 Districts")
    top50 = raw2.nlargest(50, "Population")
    rel_pct_cols = [c+"_pct" for c in religion_cols]
    heat_data = top50.set_index("District name")[rel_pct_cols]
    heat_data.columns = [c.replace("_pct","") for c in rel_pct_cols]

    fig_h, ax_h = plt.subplots(figsize=(12, 14))
    sns.heatmap(heat_data, cmap="YlGnBu", ax=ax_h, linewidths=0.3,
                cbar_kws={"label":"Population %"})
    ax_h.set_title("Religion-wise Population Distribution — Top 50 Indian Districts", fontsize=14, fontweight="bold", pad=14)
    ax_h.set_xlabel("Religion")
    ax_h.set_ylabel("District")
    plt.tight_layout()
    st.pyplot(fig_h); plt.close()

    st.divider()

    # ── CHART 2: Gender pyramid ───────────────────────────────
    st.markdown("### 2. Gender-wise Population Distribution — Major Districts")
    popular_districts = [
        "Mumbai","New Delhi","Bangalore","Chennai","Hyderabad",
        "Pune","Kolkata","Ahmedabad","Jaipur","Surat",
        "Agra","Lucknow","Srinagar","Kanpur","Indore",
        "Bhopal","Patna","Vadodara","Nagpur","Ludhiana",
        "Amritsar","Noida","Gurgaon","Faridabad"
    ]
    df_g = raw2[raw2["District name"].isin(popular_districts)].copy()
    df_g = df_g.dropna(subset=["Male_pct","Female_pct"])
    df_g["District name"] = pd.Categorical(df_g["District name"], categories=popular_districts, ordered=True)
    df_g = df_g.sort_values("District name")

    fig_g, ax_g = plt.subplots(figsize=(11, 8))
    ax_g.barh(df_g["District name"],  df_g["Male_pct"],   color="#4C72B0", height=0.5, label="Male")
    ax_g.barh(df_g["District name"], -df_g["Female_pct"], color="#DD8452", height=0.5, label="Female")
    ax_g.axvline(0, color="gray", linewidth=1)

    for _, row in df_g.iterrows():
        ax_g.text( row["Male_pct"]+0.5,   row["District name"], f"{row['Male_pct']:.1f}%",   va="center", ha="left",  fontsize=8, color="white", fontweight="bold")
        ax_g.text(-row["Female_pct"]-0.5, row["District name"], f"{row['Female_pct']:.1f}%", va="center", ha="right", fontsize=8, color="white", fontweight="bold")

    max_pct = max(df_g["Male_pct"].max(), df_g["Female_pct"].max())
    ax_g.set_xlim(-max_pct-4, max_pct+4)
    ax_g.set_xticks(np.arange(-max_pct-4, max_pct+5, 5))
    ax_g.set_xticklabels([abs(int(x)) for x in ax_g.get_xticks()])
    ax_g.set_xlabel("Population Share")
    ax_g.set_ylabel("District")
    ax_g.set_title("Gender-wise Population Distribution — Major Indian Districts", fontsize=14, fontweight="bold", pad=16)
    ax_g.legend(loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False)
    ax_g.grid(axis="x", linestyle="--", alpha=0.4)
    ax_g.grid(axis="y", visible=False)
    for spine in ["top","right","left"]:
        ax_g.spines[spine].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_g); plt.close()

    st.divider()

    # ── CHART 3: Age composition Mumbai vs New Delhi ──────────
    st.markdown("### 3. Age-wise Population Composition — Mumbai vs New Delhi")
    age_cols = ["age 0-29","age 30-49","age 50>"]
    cities3  = ["Mumbai","New Delhi"]
    df3 = raw2[raw2["District name"].isin(cities3)].copy()
    df3_age = df3.set_index("District name")[age_cols]
    df3_pct = df3_age.div(df3_age.sum(axis=1), axis=0) * 100
    colors3 = ["#5B8FF9","#61DDAA","#F6BD16"]

    fig3c, ax3c = plt.subplots(figsize=(8, 6))
    bottom = np.zeros(len(df3_pct))
    for i, age in enumerate(age_cols):
        bars = ax3c.bar(df3_pct.index, df3_pct[age], bottom=bottom,
                        label=age.replace("age ",""), color=colors3[i], edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, df3_pct[age]):
            ax3c.text(bar.get_x()+bar.get_width()/2,
                      bar.get_y()+bar.get_height()/2,
                      f"{val:.1f}%", ha="center", va="center", fontsize=11, color="white", fontweight="bold")
        bottom += df3_pct[age].values

    ax3c.set_ylim(0, 100)
    ax3c.set_ylabel("Population Share (%)")
    ax3c.set_title("Age-wise Population Composition (100%)\nMumbai vs New Delhi", fontsize=13, fontweight="bold")
    ax3c.legend(title="Age Group")
    ax3c.grid(axis="y", linestyle="--", alpha=0.25)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    st.pyplot(fig3c); plt.close()

    st.divider()

    # ── CHART 4: Age group bar — Mumbai vs Pune ───────────────
    st.markdown("### 4. Age Group Distribution — Mumbai vs Pune")
    age_pct_cols = ["Age not stated","age 0-29","age 30-49","age 50>"]
    cities4 = ["Mumbai","Pune"]
    df4 = raw2[raw2["District name"].isin(cities4)].copy()
    for col in age_pct_cols:
        df4[col+" %"] = df4[col] / df4["Population"] * 100
    age_pct_names = [c+" %" for c in age_pct_cols]

    df_melt = df4.melt(id_vars="District name", value_vars=age_pct_names,
                        var_name="age", value_name="percentage")
    df_melt["age"] = df_melt["age"].str.replace(" %","")

    from matplotlib.ticker import PercentFormatter
    palette = {"Mumbai":"#1f77b4","Pune":"#ff7f0e"}
    sns.set_theme(style="whitegrid", context="talk")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_melt, x="age", y="percentage", hue="District name",
                palette=palette, ax=ax4)
    for p in ax4.patches:
        ax4.annotate(f"{p.get_height():.1f}%",
                     (p.get_x()+p.get_width()/2, p.get_height()+0.3),
                     ha="center", fontsize=9)
    ax4.yaxis.set_major_formatter(PercentFormatter())
    ax4.set_title("Age Group Distribution (% of Total Population)", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Age Group")
    ax4.set_ylabel("Population Percentage (%)")
    ax4.legend(title="District")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    st.pyplot(fig4); plt.close()

    st.divider()

    # ── CHART 5: Religion comparison Jaipur vs Srinagar ──────
    st.markdown("### 5. Hindu–Muslim Population Share — Jaipur vs Srinagar")
    cities5 = ["Jaipur","Srinagar"]
    df5 = raw2[raw2["District name"].isin(cities5)].copy()
    rel5 = ["Hindus","Muslims"]
    melt5 = df5.melt(id_vars="District name", value_vars=rel5,
                     var_name="Religion", value_name="Pop")
    total5 = melt5.groupby("District name")["Pop"].transform("sum")
    melt5["Percentage"] = melt5["Pop"] / total5 * 100

    palette5 = {"Jaipur":"#e75480","Srinagar":"#2e8b57"}
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)
    fig5, ax5 = plt.subplots(figsize=(9, 6))
    sns.barplot(data=melt5, x="Religion", y="Percentage", hue="District name",
                palette=palette5, ax=ax5)
    for p in ax5.patches:
        ax5.annotate(f"{p.get_height():.1f}%",
                     (p.get_x()+p.get_width()/2, p.get_height()+0.5),
                     ha="center", fontsize=10)
    ax5.yaxis.set_major_formatter(PercentFormatter())
    ax5.set_title("Religion [Hindu–Muslim]-wise Population Share (%)\nJaipur vs Srinagar", fontsize=13, fontweight="bold")
    ax5.set_xlabel("Religion")
    ax5.set_ylabel("Population Percentage (%)")
    ax5.legend(title="District")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    st.pyplot(fig5); plt.close()

    st.divider()

    # ── CHART 6: Donut chart Thiruvananthapuram ───────────────
    st.markdown("### 6. Religion-wise Distribution — Thiruvananthapuram")
    city6 = raw2[raw2["District name"] == "Thiruvananthapuram"].iloc[0]
    rel6_cols = ["Hindus","Muslims","Christians","Sikhs"]
    rel6_vals = [city6[c] for c in rel6_cols]
    rel6_pct  = [v/city6["Population"]*100 for v in rel6_vals]
    rel6_labels = [f"{c} %\n{p:.1f}%" for c,p in zip(["Hindu","Muslim","Christian","Sikh"], rel6_pct)]
    colors6 = ["#5B8FF9","#61DDAA","#F4664A","#FAAD14"]

    fig6, ax6 = plt.subplots(figsize=(7, 7))
    wedges, texts = ax6.pie(rel6_vals, labels=rel6_labels, colors=colors6,
                             startangle=90, wedgeprops=dict(width=0.5),
                             textprops=dict(fontsize=11))
    ax6.set_title("Religion-wise Population Distribution — Thiruvananthapuram",
                  fontsize=12, fontweight="bold", pad=20)
    ax6.text(0, 0, "Census 2011\nPercentage Share", ha="center", va="center",
             fontsize=9, color="gray")
    plt.tight_layout()
    st.pyplot(fig6); plt.close()

    st.divider()

    # ── CHART 7: Literacy vs Population scatter ───────────────
    st.markdown("### 7. Literacy vs Population — Major Indian Districts")
    major = [
        "Mumbai","New Delhi","Bangalore","Chennai","Hyderabad","Pune","Kolkata",
        "Ahmedabad","Jaipur","Surat","Agra","Lucknow","Srinagar","Indore",
        "Bhopal","Patna","Vadodara","Nagpur","Ludhiana","Amritsar","Gurgaon",
        "Faridabad","Noida"
    ]
    df7 = raw2[raw2["District name"].isin(major)].copy()
    df7["Pop_millions"] = df7["Population"] / 1e6

    fig7, ax7 = plt.subplots(figsize=(10, 7))
    ax7.scatter(df7["literacy_rate"]*100, df7["Pop_millions"],
                color="#4C72B0", alpha=0.7, s=60, edgecolors="white")
    for _, row in df7.iterrows():
        ax7.annotate(row["District name"],
                     (row["literacy_rate"]*100, row["Pop_millions"]),
                     fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax7.set_xlabel("Literacy Rate (%)")
    ax7.set_ylabel("Population (Millions)")
    ax7.set_title("Literacy vs Population — Major Indian Districts", fontsize=13, fontweight="bold")
    ax7.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig7); plt.close()

    st.divider()

    # ── CHART 8: Hexbin — all 640 districts ──────────────────
    st.markdown("### 8. Literacy Rate vs Population — All 640 Districts (Density)")
    raw2["lit_abs"] = raw2["literate education"]
    valid = raw2.dropna(subset=["lit_abs","Population"])
    valid = valid[valid["Population"] > 0]

    fig8, ax8 = plt.subplots(figsize=(9, 7))
    hb = ax8.hexbin(valid["lit_abs"], valid["Population"],
                    gridsize=30, cmap="viridis", mincnt=1,
                    yscale="log", xscale="linear")
    cb = fig8.colorbar(hb, ax=ax8)
    cb.set_label("Number of Districts")
    ax8.set_xlabel("Literacy Rate (%)")
    ax8.set_ylabel("Population (log scale)")
    ax8.set_title("Literacy vs Population — All Indian Districts", fontsize=13, fontweight="bold")
    ax8.text(0.02, 0.97, f"Number of districts used: {len(valid)}",
             transform=ax8.transAxes, fontsize=9, va="top", color="gray")
    plt.tight_layout()
    st.pyplot(fig8); plt.close()

st.divider()
st.caption("Built with Streamlit · Census 2011 · Gradient Boosting Classifier · scikit-learn")
