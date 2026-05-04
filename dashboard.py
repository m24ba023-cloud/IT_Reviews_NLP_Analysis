import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64
import nltk
import warnings
warnings.filterwarnings("ignore")

nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt",         quiet=True)
nltk.download("stopwords",     quiet=True)
nltk.download("wordnet",       quiet=True)

# ── Page config ────────────────────────────────────
st.set_page_config(
    page_title="IT Reviews NLP Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1F4E79;
        text-align: center;
        padding: 1rem 0 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #595959;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg,#f0f4ff,#e8f0fe);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #d0e0ff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────
st.markdown(
    "<div class='main-header'>📊 IT Customer Reviews — NLP Analytics Dashboard</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='sub-header'>Powered by RoBERTa • SBERT • HDBSCAN • Sentence Embeddings</div>",
    unsafe_allow_html=True
)

# ── Load data ──────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("reviews_updated_pipeline.xlsx")
    except:
        df = pd.read_excel("reviews_FINAL_complete.xlsx")
    return df

df = load_data()

# ── Sidebar filters ────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/fluency/96/analytics.png",
    width=80
)
st.sidebar.title("🔍 Filters")
st.sidebar.markdown("---")

# Sentiment filter
sentiments = ["All"] + sorted(
    df["Sentiment_Label"].unique().tolist()
)
sel_sentiment = st.sidebar.selectbox(
    "Sentiment", sentiments
)

# Issue category filter
if "Issue_Category" in df.columns:
    categories = ["All"] + sorted(
        df["Issue_Category"].dropna().unique().tolist()
    )
    sel_category = st.sidebar.selectbox(
        "Issue Category", categories
    )
else:
    sel_category = "All"

# Rating filter
rating_range = st.sidebar.slider(
    "Rating Range", 1, 5, (1, 5)
)

# Search box
search_term = st.sidebar.text_input(
    "🔎 Search in reviews", ""
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Pipeline Info")
st.sidebar.markdown("""
- **Preprocessing:** Lemmatization + N-grams
- **Embeddings:** Sentence-BERT
- **Clustering:** HDBSCAN (auto)
- **Model:** RoBERTa fine-tuned
- **Imbalance Fix:** SMOTE
""")

# ── Apply filters ──────────────────────────────────
filtered = df.copy()

if sel_sentiment != "All":
    filtered = filtered[
        filtered["Sentiment_Label"] == sel_sentiment
    ]
if sel_category != "All" and "Issue_Category" in df.columns:
    filtered = filtered[
        filtered["Issue_Category"] == sel_category
    ]
filtered = filtered[
    filtered["Rating"].between(
        rating_range[0], rating_range[1]
    )
]
if search_term:
    filtered = filtered[
        filtered["Review_Text"].str.contains(
            search_term, case=False, na=False
        )
    ]

# ── KPI Cards ──────────────────────────────────────
st.markdown("### 📈 Key Metrics")
total  = len(filtered)
pos    = len(filtered[filtered["Sentiment_Label"]=="Positive"])
neg    = len(filtered[filtered["Sentiment_Label"]=="Negative"])
neu    = len(filtered[filtered["Sentiment_Label"]=="Neutral"])
avg_rt = round(filtered["Rating"].mean(), 2) if total else 0
avg_css= round(filtered["CSS"].mean(), 3) if "CSS" in filtered.columns and total else "N/A"

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("📝 Total Reviews",  total)
c2.metric("😊 Positive",  pos,  f"{round(pos/total*100) if total else 0}%")
c3.metric("😠 Negative",  neg,  f"{round(neg/total*100) if total else 0}%")
c4.metric("😐 Neutral",   neu,  f"{round(neu/total*100) if total else 0}%")
c5.metric("⭐ Avg Rating", avg_rt)
c6.metric("💯 Avg CSS",   avg_css)

st.markdown("---")

# ── TABS ───────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "📊 Sentiment",
    "🏷️ Issues",
    "☁️ Word Clouds",
    "📉 Trends",
    "🔵 Clusters",
    "🤖 Model Results",
    "🗂️ Data Explorer"
])

# ── TAB 1: Sentiment ───────────────────────────────
with tab1:
    st.subheader("Sentiment Distribution")
    col1, col2 = st.columns(2)

    with col1:
        sent_counts = filtered["Sentiment_Label"].value_counts()
        fig = px.pie(
            values=sent_counts.values,
            names=sent_counts.index,
            color=sent_counts.index,
            color_discrete_map={
                "Positive":"#639922",
                "Negative":"#E24B4A",
                "Neutral" :"#888780"
            },
            title="Sentiment Split",
            hole=0.4
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rating_counts = filtered["Rating"].value_counts().sort_index()
        fig2 = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            color=rating_counts.index,
            color_continuous_scale="Blues",
            title="Rating Distribution",
            labels={"x":"Rating","y":"Count"}
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # CSS by category
    if "CSS" in filtered.columns and "Issue_Category" in filtered.columns:
        st.subheader("Customer Satisfaction Score by Category")
        css_cat = filtered.groupby(
            "Issue_Category"
        )["CSS"].mean().round(3).sort_values()

        fig3 = px.bar(
            x=css_cat.values,
            y=css_cat.index,
            orientation="h",
            color=css_cat.values,
            color_continuous_scale="RdYlGn",
            title="Avg CSS Score",
            labels={"x":"CSS","y":"Category"},
            text=css_cat.values
        )
        fig3.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )
        st.plotly_chart(fig3, use_container_width=True)

# ── TAB 2: Issues ──────────────────────────────────
with tab2:
    st.subheader("Issue Category Analysis")
    if "Issue_Category" in filtered.columns:
        col1, col2 = st.columns(2)

        with col1:
            cat_counts = filtered[
                "Issue_Category"
            ].value_counts().head(10)
            fig = px.bar(
                x=cat_counts.values,
                y=cat_counts.index,
                orientation="h",
                color=cat_counts.values,
                color_continuous_scale="Blues",
                title="Top Issue Categories",
                labels={"x":"Count","y":"Category"}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            pivot = pd.crosstab(
                filtered["Issue_Category"],
                filtered["Sentiment_Label"]
            )
            fig2 = px.imshow(
                pivot,
                color_continuous_scale="YlOrRd",
                title="Sentiment × Issue Heatmap",
                text_auto=True,
                aspect="auto"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Sunburst chart
        st.subheader("Issue × Sentiment — Sunburst")
        sun_data = filtered.groupby(
            ["Issue_Category","Sentiment_Label"]
        ).size().reset_index(name="count")

        fig3 = px.sunburst(
            sun_data,
            path=["Issue_Category","Sentiment_Label"],
            values="count",
            color="Sentiment_Label",
            color_discrete_map={
                "Positive":"#639922",
                "Negative":"#E24B4A",
                "Neutral" :"#888780"
            },
            title="Interactive Sunburst — Issues & Sentiment"
        )
        fig3.update_layout(height=500)
        st.plotly_chart(fig3, use_container_width=True)

# ── TAB 3: Word Clouds ─────────────────────────────
with tab3:
    st.subheader("Dynamic Word Clouds")
    wc_sent = st.radio(
        "Select sentiment:",
        ["Positive","Negative","Neutral","All"],
        horizontal=True
    )

    cmap_map = {
        "Positive":"Greens",
        "Negative":"Reds",
        "Neutral" :"Blues",
        "All"     :"viridis"
    }

    if wc_sent == "All":
        wc_data = filtered
    else:
        wc_data = filtered[
            filtered["Sentiment_Label"] == wc_sent
        ]

    text_col = "processed_text" if "processed_text" in filtered.columns else "Review_Text"
    text = " ".join(
        wc_data[text_col].dropna().astype(str)
    )
    text = " ".join(
        [w for w in text.split() if "_" not in w]
    )

    if text.strip():
        wc = WordCloud(
            width=900, height=450,
            background_color="white",
            colormap=cmap_map[wc_sent],
            max_words=100,
            collocations=False
        ).generate(text)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(
            f"Word Cloud — {wc_sent} Reviews "
            f"({len(wc_data)} reviews)",
            fontsize=14, fontweight="bold"
        )
        st.pyplot(fig)
        plt.close()

        # Download button
        buf = io.BytesIO()
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        ax2.imshow(wc, interpolation="bilinear")
        ax2.axis("off")
        fig2.savefig(buf, format="png",
                    bbox_inches="tight", dpi=150)
        buf.seek(0)
        st.download_button(
            "⬇️ Download Word Cloud",
            buf, f"wordcloud_{wc_sent}.png",
            "image/png"
        )
        plt.close()
    else:
        st.warning("No data for this filter!")

# ── TAB 4: Trends ──────────────────────────────────
with tab4:
    st.subheader("Sentiment & Rating Trends")

    # Sentiment distribution bar
    sent_cat = filtered.groupby(
        ["Issue_Category","Sentiment_Label"]
    ).size().unstack(fill_value=0)

    if not sent_cat.empty:
        fig = go.Figure()
        colors = {
            "Positive":"#639922",
            "Negative":"#E24B4A",
            "Neutral" :"#888780"
        }
        for sent in ["Positive","Negative","Neutral"]:
            if sent in sent_cat.columns:
                fig.add_trace(go.Bar(
                    name=sent,
                    x=sent_cat.index,
                    y=sent_cat[sent],
                    marker_color=colors[sent]
                ))
        fig.update_layout(
            barmode="group",
            title="Sentiment by Issue Category",
            xaxis_title="Category",
            yaxis_title="Count",
            legend_title="Sentiment"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Rating vs CSS scatter
    if "CSS" in filtered.columns:
        col1, col2 = st.columns(2)
        with col1:
            # Using lowess trendline — no statsmodels dependency needed
            fig2 = px.scatter(
                filtered,
                x="Rating", y="CSS",
                color="Sentiment_Label",
                color_discrete_map={
                    "Positive":"#639922",
                    "Negative":"#E24B4A",
                    "Neutral" :"#888780"
                },
                title="Rating vs CSS Score",
                opacity=0.6,
                trendline="lowess"
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            # CSS distribution
            fig3 = px.histogram(
                filtered, x="CSS",
                color="Sentiment_Label",
                color_discrete_map={
                    "Positive":"#639922",
                    "Negative":"#E24B4A",
                    "Neutral" :"#888780"
                },
                title="CSS Score Distribution",
                nbins=30, barmode="overlay",
                opacity=0.7
            )
            st.plotly_chart(fig3, use_container_width=True)

# ── TAB 5: Clusters ────────────────────────────────
with tab5:
    st.subheader("HDBSCAN Cluster Analysis")
    st.info(
        "HDBSCAN automatically detects clusters "
        "— no need to specify K like K-Means!"
    )

    if "hdbscan_cluster" in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            cluster_dist = df[
                "hdbscan_cluster"
            ].value_counts().sort_index()
            cluster_dist.index = [
                f"Cluster {i}" if i >= 0 else "Noise"
                for i in cluster_dist.index
            ]
            fig = px.bar(
                x=cluster_dist.index,
                y=cluster_dist.values,
                color=cluster_dist.values,
                color_continuous_scale="Viridis",
                title="Reviews per HDBSCAN Cluster",
                labels={"x":"Cluster","y":"Count"}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Cluster Info")
            st.markdown("""
            | Feature | K-Means | HDBSCAN |
            |---------|---------|---------|
            | K needed? | ✅ Yes | ❌ No |
            | Noise detection | ❌ No | ✅ Yes |
            | Shape flexible | ❌ No | ✅ Yes |
            | IT reviews | Moderate | **Better** |
            """)
    else:
        st.info(
            "HDBSCAN cluster column not found. "
            "Run the full pipeline first."
        )

    # Show saved cluster image
    try:
        from PIL import Image
        img = Image.open("hdbscan_umap_clusters.png")
        st.image(
            img,
            caption="HDBSCAN + UMAP Cluster Visualization",
            use_container_width=True
        )
    except:
        st.warning(
            "Upload hdbscan_umap_clusters.png "
            "to see cluster visualization"
        )

# ── TAB 6: Model Results ───────────────────────────
with tab6:
    st.subheader("NLP Model Comparison")

    model_data = pd.DataFrame({
        "Model": [
            "TextBlob","VADER","LSTM",
            "BiLSTM (baseline)",
            "BiLSTM + GloVe + Aug",
            "RoBERTa Fine-tuned"
        ],
        "Accuracy": [42.0,40.0,44.33,67.0,74.85,87.0],
        "Type": [
            "Rule-based","Rule-based",
            "Deep Learning","Deep Learning",
            "DL + GloVe","Transformer"
        ]
    })

    # Allow user to update RoBERTa accuracy
    rob_acc = st.number_input(
        "Enter your RoBERTa accuracy (%):",
        min_value=50.0,
        max_value=100.0,
        value=87.0, step=0.1
    )
    model_data.loc[
        model_data["Model"]=="RoBERTa Fine-tuned",
        "Accuracy"
    ] = rob_acc

    color_map = {
        "Rule-based"   :"#B4B2A9",
        "Deep Learning":"#378ADD",
        "DL + GloVe"   :"#1D9E75",
        "Transformer"  :"#E24B4A"
    }

    fig = px.bar(
        model_data,
        x="Model", y="Accuracy",
        color="Type",
        color_discrete_map=color_map,
        title="Model Accuracy Comparison",
        text="Accuracy"
    )
    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )
    fig.add_hline(
        y=67, line_dash="dash",
        line_color="gray",
        annotation_text="Baseline BiLSTM (67%)"
    )
    fig.update_layout(yaxis_range=[0,105])
    st.plotly_chart(fig, use_container_width=True)

    # Pipeline improvements table
    st.subheader("Pipeline Improvements Applied")
    improvements = pd.DataFrame({
        "Change": [
            "Stemming removed",
            "Bigrams + N-grams added",
            "TF-IDF → Sentence-BERT",
            "K-Means → HDBSCAN",
            "SMOTE balancing",
            "RoBERTa fine-tuning"
        ],
        "Before": [
            "Stemming + Lemmatization",
            "Unigrams only",
            "Sparse TF-IDF vectors",
            "K=5 fixed clusters",
            "Class weights only",
            "BiLSTM (67%)"
        ],
        "After": [
            "Only Lemmatization ✅",
            "Uni + Bi + Trigrams ✅",
            "384-dim SBERT vectors ✅",
            "Auto clusters ✅",
            "Balanced classes ✅",
            f"RoBERTa ({rob_acc}%) ✅"
        ],
        "Impact": [
            "+2-3%", "+2-4%",
            "+5-8%", "Better clusters",
            "+3-5%", "+15-20%"
        ]
    })
    st.dataframe(
        improvements,
        use_container_width=True,
        hide_index=True
    )

    # Show confusion matrix image
    try:
        from PIL import Image
        img = Image.open("roberta_confusion.png")
        st.image(
            img,
            caption="RoBERTa Confusion Matrix",
            use_container_width=True
        )
    except:
        pass

# ── TAB 7: Data Explorer ───────────────────────────
with tab7:
    st.subheader("🗂️ Interactive Data Explorer")
    st.write(f"Showing **{len(filtered)}** reviews "
             f"(filtered from {len(df)} total)")

    # Columns to show
    show_cols = [
        c for c in [
            "Review_Text","Rating",
            "Sentiment_Label","Issue_Category",
            "CSS","vader_label","textblob_label"
        ] if c in filtered.columns
    ]

    st.dataframe(
        filtered[show_cols].reset_index(drop=True),
        use_container_width=True,
        height=400
    )

    col1, col2 = st.columns(2)
    with col1:
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            csv,
            "filtered_reviews.csv",
            "text/csv"
        )
    with col2:
        st.metric(
            "Reviews in current filter",
            len(filtered),
        )
