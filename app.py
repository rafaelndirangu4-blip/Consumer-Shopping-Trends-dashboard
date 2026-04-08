# Final_Project
# Group No: 2
# Group Members: 
# Raphael Ndirangu Ndegwa (200650808)
# Pardeep Kaur (200632295)
# Gurleen Kaur (200604519)


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

# ─────────────────────────────────────────────
# 1. PAGE CONFIGURATION - # RAPHAEL NDEGWA (200650808)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ShopLens – Shopping Trends Analytics",
    page_icon="🛍️",
    layout="wide"
)

# ─────────────────────────────────────────────
# 2. CUSTOM Cascading Stylesheet
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size: 3.2rem; font-weight: 800; color: #1a237e; margin-bottom: 0; }
    .sub-title   { font-size: 1.5rem; color: #555; margin-top: 0.2rem; }
    .metric-card { background: #f0f4ff; border-left: 5px solid #1a237e;
                   padding: 14px 18px; border-radius: 8px; margin-bottom: 10px; }
    .metric-val  { font-size: 1.7rem; font-weight: 700; color: #1a237e; }
    .metric-lbl  { font-size: 0.85rem; color: #555; }
    .member-card { background: #fff; border: 1px solid #dde; border-radius: 10px;
                   padding: 16px; text-align: center; transition: box-shadow .2s; }
    .member-card:hover { box-shadow: 0 4px 16px rgba(26,35,126,.15); }
    .member-name { font-weight: 700; font-size: 1rem; color: #1a237e; }
    .member-role { font-size: 0.82rem; color: #777; }
    .section-header { font-size: 1.3rem; font-weight: 700; color: #1a237e;
                       border-bottom: 2px solid #1a237e; padding-bottom: 4px; margin: 18px 0 10px; }
    .regex-badge { background:#e8f5e9; border:1px solid #66bb6a; border-radius:20px;
                   padding:2px 12px; font-size:0.82rem; color:#2e7d32; display:inline-block; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. HEADER
# ─────────────────────────────────────────────
st.markdown('<p style="font-size:3.2rem; font-weight:800; color:#1a237e; margin-bottom:0;">🛍️ ShopLens: Consumer Shopping Trends Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:1.5rem; color:#555; margin-top:0.2rem;">Uncovering Patterns in Modern Retail Behavior · Data Analytics Programming Final Project · Presented with Streamlit</p>', unsafe_allow_html=True)
st.divider()

col_info, col_members = st.columns([2, 1])

with col_info:
    st.subheader("📋 Project Summary")
    st.write("""
    This application loads and analyzes a **Consumer Shopping Trends** dataset sourced from Kaggle.
    It cleans and preprocesses the data, computes key retail metrics, enables **Regex-powered search**
    for filtering customer records, and visualizes insights using **Matplotlib** charts — all wrapped
    in an interactive **Streamlit** dashboard.
    """)
    st.markdown(
        "**Dataset:** Consumer Shopping Trends Dataset — Kaggle  \n"
        "**Records:** 3,900 customers · 18 columns  \n"
        "**Categories:** Clothing, Footwear, Outerwear, Accessories"
    )

with col_members:
    st.subheader("👥 Group Members")

    members = [
        {
            "name": "Raphael Ndegwa",
            "role": "Data Cleaning & Preprocessing",
            "info": "Student ID: 200650808 | Handles data loading, null treatment, and column normalization.",
            "icon": "🧑‍💻"
        },
        {
            "name": "Pardeep Kaur",
            "role": "Data Analysis & Metrics",
            "info": "Student ID: 200632295 | Computes KPIs, purchase averages, and category-level summaries.",
            "icon": "📊"
        },
        {
            "name": "Gurleen Kaur",
            "role": "Visualization & UI Design",
            "info": "Student ID: 200604519 | Builds all 4 Matplotlib charts and the Streamlit layout.",
            "icon": "🎨"
        },
    ]

    for m in members:
        st.markdown(
            f"""<div class="member-card" title="{m['info']}">
                <span style="font-size:1.6rem">{m['icon']}</span><br>
                <span class="member-name">{m['name']}</span><br>
                <span class="member-role">{m['role']}</span>
            </div><br>""",
            unsafe_allow_html=True
        )

st.divider()

# ─────────────────────────────────────────────
# 4. DATA LOADING & CLEANING
# ─────────────────────────────────────────────
st.markdown('<p class="section-header">📂 Data Loading & Cleaning</p>', unsafe_allow_html=True)

# Caching prevents reloading the dataset every time the app updates
@st.cache_data

# Function to load dataset and cache it for better performance on subsequent runs
def load_data():
    df = pd.read_csv("Shopping_trends.csv")
    df.columns = df.columns.str.strip()

# Rename columns for consistency and easier access in code

    df.rename(columns={
        "Customer ID":            "Customer_ID",
        "Purchase Amount (USD)":  "Purchase_Amount",
        "Review Rating":          "Review_Rating",
        "Subscription Status":    "Subscriber",
        "Discount Applied":       "Discount_Applied",
        "Promo Code Used":        "Promo_Used",
        "Previous Purchases":     "Previous_Purchases",
        "Shipping Type":          "Shipping_Type",
        "Payment Method":         "Payment_Method",
        "Frequency of Purchases": "Purchase_Frequency",
        "Item Purchased":         "Item"
    }, inplace=True)
    df.drop_duplicates(inplace=True)

# Convert columns to numeric types for accurate calculations and handle non-numeric entries gracefully

    df["Purchase_Amount"] = pd.to_numeric(df["Purchase_Amount"], errors="coerce")
    df["Review_Rating"]   = pd.to_numeric(df["Review_Rating"],   errors="coerce")
    df["Age"]             = pd.to_numeric(df["Age"],             errors="coerce")

# Remove rows with missing important values to ensure clean analysis

    df.dropna(subset=["Purchase_Amount", "Age"], inplace=True)

 # Categorize customers into age groups for segmentation analysis

    def age_group(age):
        if age < 25:   return "18-24"
        elif age < 35: return "25-34"
        elif age < 50: return "35-49"
        else:          return "50+"
    df["Age_Group"] = df["Age"].apply(age_group)
    return df

df = load_data()

c1, c2, c3 = st.columns(3)
c1.success(f"✅ Rows loaded: **{len(df):,}**")
c2.success(f"✅ Columns: **{df.shape[1]}**")
c3.success(f"✅ No missing values after cleaning")

with st.expander("🔍 Preview raw data (first 10 rows)"):
    st.dataframe(df.head(10), use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# 5. SIDEBAR FILTERS (including Regex) - # PARDEEP KAUR (200632295)
# ─────────────────────────────────────────────
st.sidebar.header("🔎 Search & Filter")
st.sidebar.markdown("**Regex Location / Item Search**")
regex_input = st.sidebar.text_input("Pattern (e.g. `^Cal`, `jean.*`, `S[0-9]+`)", "")

cat_filter    = st.sidebar.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
season_filter = st.sidebar.multiselect("Season",   sorted(df["Season"].unique()),   default=sorted(df["Season"].unique()))
gender_filter = st.sidebar.multiselect("Gender",   sorted(df["Gender"].unique()),   default=sorted(df["Gender"].unique()))

# Apply user-selected filters from sidebar to dataset

filtered_df = df[
    df["Category"].isin(cat_filter) &
    df["Season"].isin(season_filter) &
    df["Gender"].isin(gender_filter)
].copy()

# Apply regex-based search on Location and Item columns  
# Allows flexible and advanced filtering by user input

regex_active = False
if regex_input.strip():
    try:
        pattern = re.compile(regex_input.strip(), re.IGNORECASE)
        mask = (
            filtered_df["Location"].astype(str).str.contains(pattern, regex=True) |
            filtered_df["Item"].astype(str).str.contains(pattern, regex=True)
        )
        filtered_df = filtered_df[mask]
        regex_active = True
        st.sidebar.markdown(
            f'<span class="regex-badge">✅ Regex matched {len(filtered_df):,} rows</span>',
            unsafe_allow_html=True
        )
    except re.error as e:
        st.sidebar.error(f"❌ Invalid Regex: {e}  \nTry: `^Cal`, `jean.*`, `S[0-9]+`")

# ─────────────────────────────────────────────
# 6. DATA ANALYTICS
# ─────────────────────────────────────────────
st.markdown('<p class="section-header">📊 Data Filtering & Analytics</p>', unsafe_allow_html=True)

if regex_active:
    st.info(f"🔎 Regex `{regex_input}` is active — showing **{len(filtered_df):,}** matching rows")

# Calculate key performance indicators (KPIs)    
# Includes average purchase, total revenue, rating, and subscriber percentage
if not filtered_df.empty:
    avg_purchase  = filtered_df["Purchase_Amount"].mean()
    total_revenue = filtered_df["Purchase_Amount"].sum()
    avg_rating    = filtered_df["Review_Rating"].mean()
    top_category  = filtered_df["Category"].value_counts().idxmax()
    sub_pct       = (filtered_df["Subscriber"] == "Yes").mean() * 100

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("💰 Avg Purchase",    f"${avg_purchase:.2f}")
    m2.metric("🏆 Total Revenue",   f"${total_revenue:,.0f}")
    m3.metric("⭐ Avg Rating",      f"{avg_rating:.2f}/5")
    m4.metric("🥇 Top Category",    top_category)
    m5.metric("🔔 Subscriber Rate", f"{sub_pct:.1f}%")

st.markdown("#### Filtered Customer Records")
display_cols = ["Customer_ID", "Age", "Gender", "Item", "Category",
                "Purchase_Amount", "Season", "Location", "Review_Rating", "Subscriber"]
st.dataframe(filtered_df[display_cols].reset_index(drop=True), use_container_width=True, height=280)

# Identify top 5 highest purchase transactions in the filtered dataset and display them in a formatted table

st.markdown("#### 🏆 Top 5 Purchases")
top5 = filtered_df.sort_values("Purchase_Amount", ascending=False).head(5)
top5_display = top5[["Customer_ID", "Item", "Category", "Purchase_Amount", "Location"]].reset_index(drop=True)
top5_display["Purchase_Amount"] = top5_display["Purchase_Amount"].map("${:,.2f}".format)
st.table(top5_display)

st.divider()

# ─────────────────────────────────────────────
# 7. VISUALIZATIONS (4 charts) - # GURLEEN KAUR (200604519)
# ─────────────────────────────────────────────
# # Creating visual representations of data to identify trends and patterns
# Includes bar chart, pie chart, line chart, and stacked area chart

st.markdown('<p class="section-header">📉 Visualizations</p>', unsafe_allow_html=True)

COLORS = ["#1a237e", "#0288d1", "#00897b", "#f4511e", "#8e24aa", "#fdd835"]

if filtered_df.empty:
    st.warning("⚠️ No data matches your current filters. Please adjust the sidebar.")
    st.stop()

# ── Row 1: Chart 1 + Chart 2 ──────────────────
row1_c1, row1_c2 = st.columns(2)

# Chart 1: Avg Purchase Amount by Category (bar chart)
# Shows which product categories generate higher spending

with row1_c1:
    st.markdown("##### 📊 Avg Purchase Amount by Category")
    cat_df = (filtered_df.groupby("Category")["Purchase_Amount"]
              .mean().reset_index().sort_values("Purchase_Amount", ascending=False))

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    bars = ax1.bar(cat_df["Category"], cat_df["Purchase_Amount"],
                   color=COLORS[:len(cat_df)], alpha=0.88, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, cat_df["Purchase_Amount"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"${val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Average Purchase (USD)")
    ax1.set_title("Average Purchase Amount by Category", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, cat_df["Purchase_Amount"].max() * 1.2)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.spines[["top", "right"]].set_visible(False)
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

# Chart 2: Revenue Distribution by Season (pie chart)
# Displays how revenue is distributed across different seasons
with row1_c2:
    st.markdown("##### 🥧 Revenue Distribution by Season")
    season_rev = filtered_df.groupby("Season")["Purchase_Amount"].sum()

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    wedges, texts, autotexts = ax2.pie(
        season_rev.values,
        labels=season_rev.index,
        autopct="%1.1f%%",
        colors=COLORS[:len(season_rev)],
        startangle=90,
        textprops={"fontsize": 9},
        wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax2.set_title("Revenue Share by Season", fontsize=10, fontweight="bold")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ── Row 2: Chart 3 + Chart 4 ──────────────────
row2_c1, row2_c2 = st.columns(2)

# Chart 3: Avg Purchase by Age Group & Gender (line chart)
# Compares spending behavior across age groups and gender

with row2_c1:
    st.markdown("##### 📈 Avg Purchase Trend by Age Group & Gender")
    age_order = ["18-24", "25-34", "35-49", "50+"]

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    for i, gender in enumerate(sorted(filtered_df["Gender"].unique())):
        subset = filtered_df[filtered_df["Gender"] == gender]
        g_avg = (
            subset.groupby("Age_Group")["Purchase_Amount"]
            .mean()
            .reindex([a for a in age_order if a in subset["Age_Group"].unique()])
        )
        ax3.plot(g_avg.index, g_avg.values,
                 marker="o", linewidth=2,
                 color=COLORS[i % len(COLORS)], label=gender)
    ax3.set_xlabel("Age Group")
    ax3.set_ylabel("Avg Purchase (USD)")
    ax3.set_title("Avg Purchase Trend by Age Group & Gender", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(linestyle="--", alpha=0.4)
    ax3.spines[["top", "right"]].set_visible(False)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

# Chart 4: Stacked Area – Revenue by Category across Age Groups
# Visualizes contribution of each category across age groups
with row2_c2:
    st.markdown("##### 🎨 Revenue by Category & Age Group")
    age_order  = ["18-24", "25-34", "35-49", "50+"]
    categories = ["Clothing", "Footwear", "Outerwear", "Accessories"]

    pivot = (
        filtered_df.groupby(["Age_Group", "Category"])["Purchase_Amount"]
        .sum()
        .unstack(fill_value=0)
        .reindex([a for a in age_order if a in filtered_df["Age_Group"].unique()])
    )
    present_cats   = [c for c in categories if c in pivot.columns]
    present_colors = [COLORS[categories.index(c)] for c in present_cats]

    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.stackplot(
        pivot.index,
        [pivot[c].values for c in present_cats],
        labels=present_cats,
        colors=present_colors,
        alpha=0.85
    )
    ax4.set_ylabel("Total Purchase Amount (USD)")
    ax4.set_xlabel("Age Group")
    ax4.set_title("Purchase Revenue by Category & Age Group", fontsize=10, fontweight="bold")
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax4.legend(loc="upper left", fontsize=8, framealpha=0.7)
    ax4.grid(linestyle="--", alpha=0.3)
    ax4.spines[["top", "right"]].set_visible(False)
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

st.divider()

# ─────────────────────────────────────────────
# 8. DATASET DESCRIPTION
# ─────────────────────────────────────────────
st.markdown('<p class="section-header">📚 Dataset Description</p>', unsafe_allow_html=True)
st.markdown("""
**Name:** Consumer Shopping Trends Dataset  
**Source:** Kaggle  
**Rows:** 3,900 · **Columns:** 18

The dataset captures shopping behavior and purchasing patterns of consumers across various
product categories, seasons, and demographics. Each row represents one customer transaction.

| Field | Description |
|---|---|
| Customer ID | Unique identifier per customer |
| Age / Gender | Demographic information |
| Item Purchased / Category | Product name and its category |
| Purchase Amount (USD) | Transaction value |
| Location | US state where purchase was made |
| Season | Season during which purchase occurred |
| Review Rating | Customer satisfaction score (1–5) |
| Subscription Status | Whether the customer is a subscriber |
| Shipping Type | Delivery method selected |
| Payment Method | How the customer paid |
| Frequency of Purchases | How often the customer shops |
""")

st.caption("🚀 Final Project — Data Analytics · Built with Streamlit, Pandas, NumPy, Matplotlib & Regex")
