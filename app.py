import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import io

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. Page Configuration & Custom CSS (เขียว-ดำ)
# ==========================================
st.set_page_config(
    page_title="Iris Model Training Studio",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyber Emerald / Black-Green Theme CSS
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', 'Kanit', sans-serif;
    }

    /* Main background & base styling */
    .stApp {
        background: linear-gradient(180deg, #060907 0%, #0a110d 50%, #070d09 100%);
        color: #e0f2e9;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #050806 !important;
        border-right: 1px solid rgba(0, 255, 136, 0.15) !important;
    }

    /* Card styling */
    .cyber-card {
        background: rgba(15, 23, 19, 0.85);
        border: 1px solid rgba(0, 255, 136, 0.25);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5), 0 0 15px rgba(0, 255, 136, 0.05);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .cyber-card:hover {
        border-color: rgba(0, 255, 136, 0.5);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 255, 136, 0.15);
    }

    /* Glowing Titles */
    .cyber-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00ff88 0%, #34d399 50%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 255, 136, 0.3);
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }

    .cyber-subtitle {
        color: #88ab96;
        font-size: 1.05rem;
        margin-bottom: 25px;
        font-weight: 300;
    }

    /* Metric Badges */
    .metric-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-setosa {
        background: rgba(16, 185, 129, 0.18);
        color: #34d399;
        border: 1px solid #10b981;
    }
    .badge-versicolor {
        background: rgba(5, 150, 105, 0.22);
        color: #00ff88;
        border: 1px solid #00ff88;
    }
    .badge-virginica {
        background: rgba(4, 120, 87, 0.25);
        color: #6ee7b7;
        border: 1px solid #6ee7b7;
    }

    /* Metric Containers in Streamlit */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 19, 0.9);
        border: 1px solid rgba(0, 255, 136, 0.2);
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stMetric"] label {
        color: #88ab96 !important;
        font-size: 0.9rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #00ff88 !important;
        font-weight: 700 !important;
    }

    /* Custom Streamlit Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #059669 0%, #00ff88 100%) !important;
        color: #041208 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.3) !important;
        transition: all 0.25s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 255, 136, 0.5) !important;
        color: #020904 !important;
    }
    .stButton>button:active {
        transform: translateY(0) !important;
    }

    /* Secondary Buttons */
    .stDownloadButton>button {
        background: rgba(16, 185, 129, 0.15) !important;
        color: #00ff88 !important;
        border: 1px solid #00ff88 !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
    }
    .stDownloadButton>button:hover {
        background: rgba(16, 185, 129, 0.3) !important;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.3) !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(12, 18, 15, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(0, 255, 136, 0.15);
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        color: #88ab96;
        border-radius: 8px;
        padding: 0 18px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0, 255, 136, 0.15) !important;
        color: #00ff88 !important;
        font-weight: 600 !important;
        border: 1px solid rgba(0, 255, 136, 0.4) !important;
    }

    /* Inputs, Selectors & Sliders */
    .stSelectbox>div>div, .stTextInput>div>div, .stNumberInput>div>div {
        background-color: #0c1410 !important;
        color: #e0f2e9 !important;
        border: 1px solid rgba(0, 255, 136, 0.25) !important;
        border-radius: 8px !important;
    }
    .stSelectbox>div>div:focus-within, .stTextInput>div>div:focus-within {
        border-color: #00ff88 !important;
        box-shadow: 0 0 10px rgba(0, 255, 136, 0.3) !important;
    }

    /* Table / Dataframe wrapper */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(0, 255, 136, 0.2);
        border-radius: 10px;
        overflow: hidden;
    }

    /* Highlight accent lines */
    hr {
        border-color: rgba(0, 255, 136, 0.2) !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ==========================================
# 2. Dataset Helper & Session State
# ==========================================
@st.cache_data
def load_iris_dataframe():
    raw_data = load_iris()
    df = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
    df['target'] = raw_data.target
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['target'].map(species_map)
    return raw_data, df

raw_data, df = load_iris_dataframe()

# Initialize session state for model & results
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = "RandomForestClassifier"
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = None
if 'train_accuracy' not in st.session_state:
    st.session_state.train_accuracy = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = raw_data.feature_names
if 'target_names' not in st.session_state:
    st.session_state.target_names = list(raw_data.target_names)

# Auto-train default model if not yet trained (same as Model Training.ipynb)
if st.session_state.model is None:
    X = df.drop(['target', 'species'], axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    default_model = RandomForestClassifier(random_state=42)
    default_model.fit(X_train, y_train)
    y_pred = default_model.predict(X_test)
    
    st.session_state.model = default_model
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.y_pred = y_pred
    st.session_state.accuracy = accuracy_score(y_test, y_pred)
    st.session_state.train_accuracy = accuracy_score(y_train, default_model.predict(X_train))


# ==========================================
# 3. Header Section
# ==========================================
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; margin-bottom: 10px;">
    <div>
        <div class="cyber-title">🌿 IRIS MODEL TRAINING STUDIO</div>
        <div class="cyber-subtitle">แอปพลิเคชันเทรนและประเมินผลโมเดล Machine Learning (อิงตามไฟล์ Model Training.ipynb) ธีมเขียว-ดำ</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Overview Metrics Banner
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(label="📊 จำนวนข้อมูลทั้งหมด", value=f"{len(df)} แถว", delta="150 ตัวอย่าง")
with m2:
    st.metric(label="🧬 จำนวน Features", value=f"{len(raw_data.feature_names)} คุณลักษณะ", delta="4 ตัวแปรต้น")
with m3:
    st.metric(label="🎯 จำนวน Classes", value=f"{len(raw_data.target_names)} สายพันธุ์", delta="3 คลาส")
with m4:
    acc_val = f"{st.session_state.accuracy * 100:.1f}%" if st.session_state.accuracy else "ยังไม่ได้เทรน"
    st.metric(label="⚡ Test Accuracy ล่าสุด", value=acc_val, delta="Model Active" if st.session_state.model else "None")


# ==========================================
# 4. Main Tab Interface
# ==========================================
tab_eda, tab_train, tab_eval, tab_predict, tab_model_file = st.tabs([
    "📊 1. ข้อมูล & EDA",
    "⚙️ 2. ตั้งค่า & เทรนโมเดล",
    "📈 3. ผลการประเมิน (Evaluation)",
    "🔮 4. ทดสอบทำนายผล (Prediction)",
    "💾 5. จัดการไฟล์โมเดล (.pkl)"
])


# ------------------------------------------
# TAB 1: ข้อมูลและการสำรวจ (EDA)
# ------------------------------------------
with tab_eda:
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.subheader("📋 ภาพรวมชุดข้อมูล (Dataset Exploration)")
    st.markdown("""
    ชุดข้อมูล **Iris Dataset** บันทึกการวัดขนาดของดอกไอริส 3 สายพันธุ์:
    <span class="metric-badge badge-setosa">0: Iris-Setosa</span>
    <span class="metric-badge badge-versicolor">1: Iris-Versicolour</span>
    <span class="metric-badge badge-virginica">2: Iris-Virginica</span>
    """, unsafe_allow_html=True)

    view_option = st.radio(
        "เลือกมุมมองการแสดงข้อมูล:",
        ["ส่วนหัวข้อมูล (df.head())", "ส่วนท้ายข้อมูล (df.tail())", "ข้อมูลสถิติเชิงพรรณนา (df.describe())", "ดูข้อมูลทั้งหมด (All Data)"],
        horizontal=True
    )

    if view_option == "ส่วนหัวข้อมูล (df.head())":
        st.dataframe(df.head(), use_container_width=True)
    elif view_option == "ส่วนท้ายข้อมูล (df.tail())":
        st.dataframe(df.tail(), use_container_width=True)
    elif view_option == "ข้อมูลสถิติเชิงพรรณนา (df.describe())":
        st.dataframe(df.describe().T, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Charts in Tab 1
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.subheader("🔥 Correlation Heatmap (ความสัมพันธ์ของฟีเจอร์)")
        st.caption("อิงจากโค้ด: `sns.heatmap(df.corr(), annot=True)` ใน Notebook")
        
        # Plot styled correlation heatmap
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fig.patch.set_facecolor('#0b130f')
        ax.set_facecolor('#0b130f')
        
        corr = df.drop('species', axis=1).corr()
        cmap = sns.dark_palette("#00ff88", as_cmap=True)
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap=cmap,
            cbar_kws={'label': 'Correlation'}, ax=ax,
            linewidths=0.5, linecolor='#060907',
            annot_kws={"color": "#e0f2e9", "fontsize": 9, "fontweight": "bold"}
        )
        ax.tick_params(colors='#88ab96', labelsize=8)
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='#88ab96')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#88ab96')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chart2:
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.subheader("🎯 การกระจายตัวของเป้าหมาย & ฟีเจอร์")
        st.caption("Petal Length vs Petal Width (คู่คุณลักษณะที่มีพลังแยกคลาสสูงสุด)")
        
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fig.patch.set_facecolor('#0b130f')
        ax.set_facecolor('#0b130f')
        
        colors = {'setosa': '#10b981', 'versicolor': '#00ff88', 'virginica': '#a7f3d0'}
        for species_name, group in df.groupby('species'):
            ax.scatter(
                group['petal length (cm)'],
                group['petal width (cm)'],
                label=species_name.capitalize(),
                color=colors[species_name],
                alpha=0.85,
                s=45,
                edgecolors='#060907'
            )
            
        ax.set_xlabel("Petal Length (cm)", color='#88ab96', fontsize=10)
        ax.set_ylabel("Petal Width (cm)", color='#88ab96', fontsize=10)
        ax.tick_params(colors='#88ab96')
        for spine in ax.spines.values():
            spine.set_color('#193325')
        legend = ax.legend(facecolor='#0b130f', edgecolor='#10b981', labelcolor='#e0f2e9')
        ax.grid(True, linestyle='--', alpha=0.15, color='#00ff88')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------
# TAB 2: ตั้งค่า & เทรนโมเดล (Model Training)
# ------------------------------------------
with tab_train:
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.subheader("⚙️ ปรับแต่งพารามิเตอร์และการเทรน (Hyperparameter Tuning)")
    st.markdown("กำหนดการแบ่ง Train/Test Split และเลือกอัลกอริทึมในการสร้างโมเดล Machine Learning")

    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("##### 1. การแบ่งข้อมูล (Train / Test Split)")
        test_size = st.slider("สัดส่วนชุดทดสอบ (Test Size)", min_value=0.1, max_value=0.5, value=0.3, step=0.05, 
                              help="ในสมุดโน้ตกำหนด test_size=0.3 (30%)")
        random_state = st.number_input("Random State (Seed)", min_value=0, max_value=999, value=42, step=1,
                                       help="ในสมุดโน้ตกำหนด random_state=42 เพื่อผลลัพธ์ที่สม่ำเสมอ")
        
        train_samples = int(len(df) * (1 - test_size))
        test_samples = len(df) - train_samples
        st.info(f"📊 ชุดฝึกสอน (Train Set): **{train_samples} ตัวอย่าง** | ชุดทดสอบ (Test Set): **{test_samples} ตัวอย่าง**")

    with col_t2:
        st.markdown("##### 2. เลือกโมเดลและพารามิเตอร์ (Algorithm)")
        algo_choice = st.selectbox(
            "อัลกอริทึมโมเดล:",
            [
                "RandomForestClassifier (อิงจาก Notebook)",
                "DecisionTreeClassifier",
                "KNeighborsClassifier",
                "SVC (Support Vector Machine)",
                "LogisticRegression"
            ]
        )

        # Dynamic Hyperparameters
        if "RandomForest" in algo_choice:
            n_estimators = st.slider("Number of Trees (n_estimators)", 10, 300, 100, step=10)
            criterion = st.selectbox("เกณฑ์การแยก (Criterion)", ["gini", "entropy", "log_loss"])
            max_depth = st.selectbox("Max Depth", [None, 3, 5, 10, 15], index=0)
        elif "DecisionTree" in algo_choice:
            criterion = st.selectbox("Criterion", ["gini", "entropy"])
            max_depth = st.selectbox("Max Depth", [None, 3, 5, 10], index=0)
        elif "KNeighbors" in algo_choice:
            n_neighbors = st.slider("จำนวนเพื่อนบ้าน (n_neighbors)", 1, 15, 5)
        elif "SVC" in algo_choice:
            kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
            c_val = st.slider("Regularization (C)", 0.1, 10.0, 1.0)
        elif "LogisticRegression" in algo_choice:
            max_iter = st.slider("Max Iterations", 100, 1000, 200)

    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Train Button
    train_clicked = st.button("🚀 เริ่มเทรนโมเดลใหม่ (Train Model)", use_container_width=True)
    
    if train_clicked:
        with st.spinner("กำลังประมวลผลการเทรนโมเดล..."):
            X = df.drop(['target', 'species'], axis=1)
            y = df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            if "RandomForest" in algo_choice:
                trained_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    criterion=criterion,
                    max_depth=max_depth,
                    random_state=random_state
                )
                m_name = "RandomForestClassifier"
            elif "DecisionTree" in algo_choice:
                trained_model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=random_state)
                m_name = "DecisionTreeClassifier"
            elif "KNeighbors" in algo_choice:
                trained_model = KNeighborsClassifier(n_neighbors=n_neighbors)
                m_name = "KNeighborsClassifier"
            elif "SVC" in algo_choice:
                trained_model = SVC(kernel=kernel, C=c_val, probability=True, random_state=random_state)
                m_name = "SVC"
            else:
                trained_model = LogisticRegression(max_iter=max_iter, random_state=random_state)
                m_name = "LogisticRegression"

            # Fitting
            trained_model.fit(X_train, y_train)
            y_pred = trained_model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            train_acc = accuracy_score(y_train, trained_model.predict(X_train))

            # Store in session state
            st.session_state.model = trained_model
            st.session_state.model_name = m_name
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.accuracy = test_acc
            st.session_state.train_accuracy = train_acc

            st.success(f"✅ เทรนโมเดล {m_name} สำเร็จเรียบร้อย! ได้ค่า Test Accuracy = {test_acc * 100:.2f}%")
            st.toast("Model Trained Successfully!", icon="🌿")
    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------
# TAB 3: ผลการประเมิน (Evaluation)
# ------------------------------------------
with tab_eval:
    if st.session_state.model is None:
        st.warning("⚠️ ยังไม่มีโมเดลที่ถูกเทรน กรุณาไปที่แท็บ 'ตั้งค่า & เทรนโมเดล' เพื่อเทรนโมเดลก่อน")
    else:
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.subheader(f"📈 ผลการประเมินประสิทธิภาพโมเดล: {st.session_state.model_name}")

        e1, e2, e3 = st.columns(3)
        with e1:
            st.metric("🎯 Test Accuracy", f"{st.session_state.accuracy * 100:.2f}%")
        with e2:
            st.metric("🏋️ Train Accuracy", f"{st.session_state.train_accuracy * 100:.2f}%")
        with e3:
            diff = (st.session_state.train_accuracy - st.session_state.accuracy) * 100
            diff_label = "Good Fit" if abs(diff) < 5 else "Check Overfit"
            st.metric("⚖️ Train-Test Gap", f"{diff:.2f}%", delta=diff_label)
        st.markdown('</div>', unsafe_allow_html=True)

        col_ev1, col_ev2 = st.columns(2)
        
        with col_ev1:
            st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
            st.subheader("🧩 Confusion Matrix")
            st.caption("อิงจากโค้ด: `confusion_matrix(y_test, y_pred)` ใน Notebook")
            
            cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
            
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#0b130f')
            ax.set_facecolor('#0b130f')
            
            sns.heatmap(
                cm, annot=True, fmt='d',
                cmap=sns.dark_palette("#00ff88", as_cmap=True),
                xticklabels=st.session_state.target_names,
                yticklabels=st.session_state.target_names,
                ax=ax,
                linewidths=0.5, linecolor='#060907',
                annot_kws={"color": "#e0f2e9", "fontsize": 12, "fontweight": "bold"}
            )
            ax.set_xlabel('Predicted Label', color='#88ab96', fontsize=10)
            ax.set_ylabel('True Label', color='#88ab96', fontsize=10)
            ax.tick_params(colors='#88ab96')
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color='#88ab96')
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#88ab96')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_ev2:
            st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
            st.subheader("📊 Classification Report")
            st.caption("อิงจากโค้ด: `classification_report(y_test, y_pred)` ใน Notebook")
            
            report_dict = classification_report(
                st.session_state.y_test,
                st.session_state.y_pred,
                target_names=st.session_state.target_names,
                output_dict=True
            )
            report_df = pd.DataFrame(report_dict).T
            st.dataframe(report_df.style.format("{:.2f}").background_gradient(cmap='Greens'), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Feature Importance (if applicable)
        if hasattr(st.session_state.model, 'feature_importances_'):
            st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
            st.subheader("🌟 Feature Importances (ความสำคัญของฟีเจอร์)")
            
            importances = st.session_state.model.feature_importances_
            feat_df = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True)

            fig, ax = plt.subplots(figsize=(8, 3.2))
            fig.patch.set_facecolor('#0b130f')
            ax.set_facecolor('#0b130f')
            
            bars = ax.barh(feat_df['Feature'], feat_df['Importance'], color='#00ff88', edgecolor='#10b981', height=0.55)
            ax.set_xlabel('Importance Score', color='#88ab96')
            ax.tick_params(colors='#88ab96')
            for spine in ax.spines.values():
                spine.set_color('#193325')
            ax.grid(axis='x', linestyle='--', alpha=0.2, color='#00ff88')
            
            # Label bar values
            for bar in bars:
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.3f}",
                        va='center', color='#e0f2e9', fontsize=9, fontweight='bold')
                        
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------
# TAB 4: ทดสอบทำนายผล (Prediction & Inference)
# ------------------------------------------
with tab_predict:
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.subheader("🔮 ทดสอบป้อนข้อมูลเพื่อทำนายผลลัพธ์ (Interactive Prediction)")
    st.markdown("อิงจากการทดสอบด้วย `input_data = (5.1, 3.5, 1.4, 0.2)` ในไฟล์ `Model Training.ipynb`")

    # Sample Presets from Notebook
    st.markdown("##### ⚡ ตัวอย่างรวดเร็ว (Quick Presets):")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    preset_vals = None
    with p_col1:
        if st.button("🌱 Setosa (Notebook: 5.1, 3.5, 1.4, 0.2)", use_container_width=True):
            preset_vals = (5.1, 3.5, 1.4, 0.2)
    with p_col2:
        if st.button("🌿 Versicolor ตัวอย่าง (5.9, 3.0, 4.2, 1.5)", use_container_width=True):
            preset_vals = (5.9, 3.0, 4.2, 1.5)
    with p_col3:
        if st.button("🌺 Virginica ตัวอย่าง (6.5, 3.0, 5.8, 2.2)", use_container_width=True):
            preset_vals = (6.5, 3.0, 5.8, 2.2)
    with p_col4:
        if st.button("🎲 ค่าเฉลี่ยของข้อมูล (Mean)", use_container_width=True):
            preset_vals = (5.84, 3.05, 3.76, 1.20)

    # Initialize preset values in session state if clicked
    if preset_vals:
        st.session_state['sl'] = preset_vals[0]
        st.session_state['sw'] = preset_vals[1]
        st.session_state['pl'] = preset_vals[2]
        st.session_state['pw'] = preset_vals[3]

    st.markdown("<hr>", unsafe_allow_html=True)

    # Feature input sliders / number fields
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        sepal_length = st.slider("Sepal Length (ความยาวกลีบเลี้ยง - cm)", 4.0, 8.0, 
                                 float(st.session_state.get('sl', 5.1)), step=0.1)
        sepal_width = st.slider("Sepal Width (ความกว้างกลีบเลี้ยง - cm)", 2.0, 4.5, 
                                float(st.session_state.get('sw', 3.5)), step=0.1)
    with col_in2:
        petal_length = st.slider("Petal Length (ความยาวกลีบดอก - cm)", 1.0, 7.0, 
                                 float(st.session_state.get('pl', 1.4)), step=0.1)
        petal_width = st.slider("Petal Width (ความกว้างกลีบดอก - cm)", 0.1, 2.6, 
                                float(st.session_state.get('pw', 0.2)), step=0.1)

    predict_btn = st.button("✨ กดเพื่อทำนายสายพันธุ์ (Predict Species)", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn or preset_vals:
        if st.session_state.model is None:
            st.error("❌ กรุณาเทรนโมเดลก่อนทำการพยากรณ์!")
        else:
            # Reshape input as in Notebook: input_data_as_numpy_array.reshape(1, -1)
            input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            pred_class = st.session_state.model.predict(input_features)[0]
            species_names = ['Iris-Setosa', 'Iris-Versicolor', 'Iris-Virginica']
            species_emojis = ['🌱', '🌿', '🌺']
            pred_species = species_names[pred_class]
            pred_emoji = species_emojis[pred_class]

            # Probabilities if available
            has_proba = hasattr(st.session_state.model, "predict_proba")
            if has_proba:
                probabilities = st.session_state.model.predict_proba(input_features)[0]
            else:
                probabilities = [1.0 if i == pred_class else 0.0 for i in range(3)]

            # Result Box
            st.markdown(f"""
            <div class="cyber-card" style="border: 2px solid #00ff88; background: rgba(10, 26, 18, 0.95);">
                <div style="text-align: center;">
                    <span style="font-size: 3rem;">{pred_emoji}</span>
                    <h2 style="color: #00ff88; margin-top: 5px; margin-bottom: 5px;">ผลการทำนาย: {pred_species}</h2>
                    <p style="color: #88ab96; font-size: 1.1rem;">(Target Class Index: <b>{pred_class}</b> | โมเดล: {st.session_state.model_name})</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probabilities distribution
            st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
            st.subheader("📊 ความน่าจะเป็นของแต่ละสายพันธุ์ (Class Probabilities):")
            
            pb1, pb2, pb3 = st.columns(3)
            with pb1:
                p0 = probabilities[0] * 100
                st.markdown(f"**🌱 Iris-Setosa:** `{p0:.1f}%`")
                st.progress(float(probabilities[0]))
            with pb2:
                p1 = probabilities[1] * 100
                st.markdown(f"**🌿 Iris-Versicolor:** `{p1:.1f}%`")
                st.progress(float(probabilities[1]))
            with pb3:
                p2 = probabilities[2] * 100
                st.markdown(f"**🌺 Iris-Virginica:** `{p2:.1f}%`")
                st.progress(float(probabilities[2]))
            st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------
# TAB 5: บันทึกและจัดการไฟล์โมเดล (.pkl)
# ------------------------------------------
with tab_model_file:
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.subheader("💾 จัดการไฟล์โมเดล (Model Persistence & joblib)")
    st.markdown("""
    ในไฟล์ `Model Training.ipynb` มีการบันทึกโมเดลด้วย:
    ```python
    import joblib
    joblib.dump(model, 'iris_model.pkl')
    ```
    และโหลดกลับมาใช้งานด้วย:
    ```python
    with open('iris_model.pkl', 'rb') as f:
        loaded_model = joblib.load(f)
    ```
    """)

    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("##### 📥 บันทึกโมเดลปัจจุบัน")
        if st.session_state.model is not None:
            # Save to disk
            if st.button("💾 บันทึกทับไฟล์ iris_model.pkl ในเครื่อง", use_container_width=True):
                joblib.dump(st.session_state.model, 'iris_model.pkl')
                st.success("✅ บันทึกโมเดลลงไฟล์ `iris_model.pkl` สำเร็จเรียบร้อยแล้ว!")
            
            # Download buffer
            model_bytes = io.BytesIO()
            joblib.dump(st.session_state.model, model_bytes)
            model_bytes.seek(0)
            st.download_button(
                label="⬇️ ดาวน์โหลด iris_model.pkl สู่คอมพิวเตอร์ของคุณ",
                data=model_bytes,
                file_name="iris_model.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )
        else:
            st.info("ยังไม่มีโมเดลที่ถูกเทรน กรุณาเทรนโมเดลก่อนบันทึก")

    with col_m2:
        st.markdown("##### 📤 โหลดโมเดลจากไฟล์")
        local_pkl_exists = os.path.exists('iris_model.pkl')
        
        if local_pkl_exists:
            file_size_kb = os.path.getsize('iris_model.pkl') / 1024
            st.info(f"📁 ตรวจพบไฟล์ `iris_model.pkl` ในโฟลเดอร์โปรเจกต์ (ขนาด: {file_size_kb:.1f} KB)")
            if st.button("🔄 โหลดโมเดลจาก iris_model.pkl ในโฟลเดอร์นี้", use_container_width=True):
                try:
                    loaded = joblib.load('iris_model.pkl')
                    st.session_state.model = loaded
                    st.session_state.model_name = type(loaded).__name__
                    st.success(f"✅ โหลดโมเดลสำเร็จ! ประเภท: {st.session_state.model_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        else:
            st.warning("⚠️ ไม่พบไฟล์ iris_model.pkl ในโฟลเดอร์ปัจจุบัน")

        # Custom Upload
        uploaded_file = st.file_uploader("หรืออัปโหลดไฟล์โมเดล .pkl ของคุณเอง:", type=['pkl'])
        if uploaded_file is not None:
            try:
                user_model = joblib.load(uploaded_file)
                st.session_state.model = user_model
                st.session_state.model_name = type(user_model).__name__
                st.success(f"✅ โหลดโมเดลจากไฟล์ที่อัปโหลดสำเร็จ: {st.session_state.model_name}")
            except Exception as e:
                st.error(f"ไม่สามารถโหลดไฟล์โมเดลได้: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# 5. Sidebar Summary & Model Info
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 15px 0 10px 0;">
        <span style="font-size: 2.5rem;">🌿</span>
        <h3 style="color: #00ff88; margin: 0; font-weight: 700;">CYBER IRIS</h3>
        <p style="color: #88ab96; font-size: 0.85rem; margin-top: 3px;">Machine Learning Control Panel</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: rgba(0, 255, 136, 0.2);'>", unsafe_allow_html=True)
    
    st.markdown("#### 📌 ข้อมูลโมเดลปัจจุบัน")
    st.markdown(f"• **อัลกอริทึม:** `{st.session_state.model_name}`")
    if st.session_state.accuracy is not None:
        st.markdown(f"• **Test Accuracy:** `{st.session_state.accuracy * 100:.2f}%`")
    if st.session_state.train_accuracy is not None:
        st.markdown(f"• **Train Accuracy:** `{st.session_state.train_accuracy * 100:.2f}%`")
        
    st.markdown("<hr style='border-color: rgba(0, 255, 136, 0.2);'>", unsafe_allow_html=True)
    
    st.markdown("#### 🎯 สายพันธุ์เป้าหมาย (Classes):")
    st.markdown("""
    - `0`: **Iris-Setosa** 🌱
    - `1`: **Iris-Versicolor** 🌿
    - `2`: **Iris-Virginica** 🌺
    """)
    
    st.markdown("<hr style='border-color: rgba(0, 255, 136, 0.2);'>", unsafe_allow_html=True)
    st.caption("อิงจากโค้ดการทำงานทั้งหมดในไฟล์ `Model Training.ipynb` ตกแต่งด้วย CSS สไตล์ Modern Cyber Dark & Neon Green (เขียว-ดำ)")
