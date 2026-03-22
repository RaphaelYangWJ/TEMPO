import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="TEMPO Deep Learning System",
    page_icon="🧠",
    layout="wide"
)

# --- Language Selection ---
with st.sidebar:
    lang = st.radio("Language / 语言", ["English", "中文"])


def t(en, zh):
    """Helper function to switch language"""
    return en if lang == "English" else zh


# --- Professional Styling ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #1c3d5a; color: white; border-radius: 4px; height: 3em; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: bold; }
    .quadrant-box { padding: 15px; border-radius: 8px; text-align: center; color: white; margin-top: 10px;}
    </style>
    """, unsafe_allow_html=True)

# --- Title and Privacy Disclosure ---
st.title(t("🏥 TEMPO Deep Learning Forecast System", "🏥 TEMPO 深度学习模型预测系统"))
st.markdown(t(
    "*A time-aware multimodal model for dynamic prognostication in lung adenocarcinoma with leptomeningeal metastasis.*",
    "*基于时间感知的多模态模型，用于肺腺癌软脑膜转移的动态预后评估。*"
))

with st.expander(t("ℹ️ Privacy and Compliance - Research Demo Only", "ℹ️ 隐私与合规说明 - 仅供学术演示")):
    st.write(t(
        "- **No PII Collection**: This system does not store Names or Patient Identifiers.\n"
        "- **Research Use Only**: This tool is for demonstration of the predictive model and is not for independent clinical diagnosis.\n"
        "- **Data Processing**: Computations are performed locally in-session. Uploaded images are cleared upon session close.",
        "- **无个人隐私收集**：本系统不存储姓名或患者标识符，请使用匿名编号。\n"
        "- **仅限科研使用**：本工具仅供演示预测模型，不可独立用于临床诊断。\n"
        "- **数据处理**：计算在本地会话中进行，上传的影像在会话结束后将自动清除。"
    ))

# --- Sidebar: Multimodal Imaging ---
with st.sidebar:
    st.header(t("🖼️ Multimodal Inputs", "🖼️ 多模态影像输入"))
    st.info(t("Upload imaging for the current assessment.", "上传当前评估节点的影像资料。"))
    mri_input = st.file_uploader(t("Upload Brain MRI (NIfTI/DICOM)", "上传脑部 MRI (NIfTI/DICOM)"),
                                 type=['nii', 'dicom'])
    pathology_input = st.file_uploader(t("Upload CSF Cytology Slide", "上传脑脊液细胞学病理切片"),
                                       type=['PNG', 'JPEG'])

    if mri_input: st.success(t("MRI Data Loaded", "MRI 数据已加载"))
    if pathology_input: st.success(t("Pathology Slide Loaded", "病理切片已加载"))

# --- Main Interface: Tabbed Navigation ---
tab_baseline, tab_dynamic, tab_results = st.tabs([
    t("📂 Baseline Profile", "📂 基线临床特征"),
    t("📈 Longitudinal & Current Assessment", "📈 纵向监测与当前评估"),
    t("📊 Dual-Outcome Prediction", "📊 双目标预测结果")
])

# --- TAB 1: Baseline Profile ---
with tab_baseline:
    st.subheader(t("Clinical Baseline Characteristics", "基线临床特征"))
    col1, col2, col3 = st.columns(3)

    with col1:
        patient_id = st.text_input(t("Anonymized Patient ID", "匿名患者 ID"), placeholder="e.g., CASE-001")
        sex = st.selectbox(t("Sex at Birth", "性别"), [t("Male", "男"), t("Female", "女")])
        age = st.number_input(t("Age at Diagnosis", "确诊年龄"), min_value=0, max_value=120, value=60)
        smoking = st.selectbox(t("Smoking Status", "吸烟史"), [t("Never", "从不"), t("Ever", "有吸烟史")])

    with col2:
        primary_diag_date = st.date_input(t("Primary Lung Cancer Diagnosis Date", "原发性肺癌诊断日期"),
                                          value=datetime(2023, 1, 1))
        lm_diag_date = st.date_input(t("LM Diagnosis Date", "软脑膜转移 (LM) 诊断日期"), value=datetime(2024, 1, 1))
        line_therapy = st.selectbox(t("Systemic Therapy Line at LM", "确诊LM时的全身治疗线数"),
                                    ["0", "1", "2", "3", ">4"])
        mutation = st.selectbox(t("Primary NGS Mutation", "原发灶 NGS 基因突变"),
                                ["EGFR L858R", "EGFR Exon 19 Del", "EGFR uncommon", "Other actionable (ALK/ROS1/MET)",
                                 "No driver alteration"])

    with col3:
        radical_resection = st.selectbox(t("Prior Radical Resection of Primary", "原发灶根治性切除史"),
                                         [t("Yes", "是"), t("No", "否")])
        st.write(t("**Central Nervous System (CNS) Status**", "**中枢神经系统 (CNS) 状态**"))
        bm_initial = st.selectbox(t("CNS Metastasis at Initial Diagnosis", "初诊肺癌时是否存在CNS转移"),
                                  [t("Yes", "是"), t("No", "否")])
        bm_num = st.selectbox(t("Number of BM Lesions at LM", "确诊LM时的脑转移病灶数"), ["0", "1", "2", "≥3"])
        brain_rt = st.selectbox(t("Prior Brain Radiotherapy", "既往脑部放疗史"), [t("Yes", "是"), t("No", "否")])

# --- TAB 2: Longitudinal Monitoring ---
with tab_dynamic:
    st.subheader(t("Current Assessment Date & Longitudinal History", "当前评估日期与纵向随访史"))

    current_date = st.date_input(t("📅 Current Assessment Date (Prediction Anchor)", "📅 当前评估日期 (预测基准点)"),
                                 value=datetime.today())
    st.caption(
        t("The model will compute gap features (e.g., 'Days since LM diagnosis', 'Time since last MRI') based on this anchor date.",
          "模型将基于此评估日期计算时间间隔特征（如“距LM诊断时间”、“距上次MRI时间”等）。"))
    st.divider()

    st.markdown(
        t("##### Longitudinal CSF Analysis (Include Current Visit)", "##### 纵向脑脊液生化分析 (请包含本次访视数据)"))

    csf_template = pd.DataFrame(columns=[
        "collection_date", "ADA", "NGLU", "C-Pro", "NCL", "LAC", "Chloride", "atypical_cells"
    ])

    csf_data = st.data_editor(
        csf_template,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "collection_date": st.column_config.DateColumn(t("Collection Date", "采集日期")),
            "ADA": st.column_config.NumberColumn("ADA (U/L)"),
            "NGLU": st.column_config.NumberColumn("NGLU (mmol/L)"),
            "C-Pro": st.column_config.NumberColumn("C-Pro (mg/L)"),
            "NCL": st.column_config.NumberColumn("NCL"),
            "LAC": st.column_config.NumberColumn("LAC (mmol/L)"),
            "Chloride": st.column_config.NumberColumn(t("Chloride (mmol/L)", "氯化物 (mmol/L)")),
            "atypical_cells": st.column_config.SelectboxColumn(
                t("Atypical Cells", "异型细胞比例"), options=["Negative", "Suspicious", "Positive"]
            )
        }
    )

# --- TAB 3: Results & Analysis ---
with tab_results:
    if st.button(t("🚀 Run TEMPO Analysis", "🚀 运行 TEMPO 分析"), type="primary"):
        with st.spinner(
                t('Integrating Multimodal Data and Running Inference...', '正在融合多模态数据并进行模型推理...')):
            # [Backend Inference logic goes here]
            import time

            time.sleep(1.5)

            # Mock Data for Visualization based on paper metrics
            progression_risk = 0.72  # 8-week CNS Progression Risk
            mortality_risk = 0.35  # 6-month Mortality Risk

        st.success(f"{t('Analysis Complete for ID', '分析已完成，患者 ID')}: {patient_id if patient_id else 'N/A'}")

        # Metrics Display
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.metric(label=t("8-Week CNS Progression Risk", "8周 CNS 进展风险"),
                      value=f"{progression_risk * 100:.1f}%")
            st.caption(t("Probability of CNS progression per RANO-LM criteria within 56 days.",
                         "56天内符合 RANO-LM 标准的中枢神经系统进展概率。"))

        with col_res2:
            st.metric(label=t("6-Month Mortality Risk", "6个月死亡风险"), value=f"{mortality_risk * 100:.1f}%")
            st.caption(
                t("Predicted probability of all-cause mortality within 6 months.", "6个月内发生全因死亡的预测概率。"))

        st.divider()

        # Four-Quadrant Pathway Framework
        st.subheader(t("Risk-Stratified Clinical Pathway", "风险分层临床路径"))

        quadrant = ""
        box_color = ""
        if mortality_risk < 0.5 and progression_risk < 0.5:
            quadrant = t("Quadrant I: Low Mortality / Low Progression", "象限 I: 低死亡风险 / 低进展风险")
            box_color = "#2e7d32"  # Green
            advice = t("Consider de-escalation of surveillance or maintaining current therapy.",
                       "考虑降低随访频率或维持当前治疗。")
        elif mortality_risk < 0.5 and progression_risk >= 0.5:
            quadrant = t("Quadrant II: Low Mortality / High Progression", "象限 II: 低死亡风险 / 高进展风险")
            box_color = "#f57c00"  # Orange
            advice = t(
                "High short-term progression risk but favorable medium-term prognosis. Escalation of CNS-directed management may be prioritized to preserve neurologic function.",
                "短期进展风险高，但中期预后尚可。建议优先考虑加强中枢神经系统靶向治疗，以保护神经功能。")
        elif mortality_risk >= 0.5 and progression_risk < 0.5:
            quadrant = t("Quadrant III: High Mortality / Low Progression", "象限 III: 高死亡风险 / 低进展风险")
            box_color = "#c2185b"  # Pink
            advice = t(
                "Short-term stability, but limited medium-term survival. Focus on broader systemic control and supportive care.",
                "短期病情稳定，但中期生存受限。建议将重点放在全身控制和支持性护理上。")
        else:
            quadrant = t("Quadrant IV: High Mortality / High Progression", "象限 IV: 高死亡风险 / 高进展风险")
            box_color = "#c62828"  # Red
            advice = t(
                "High risk in both domains. Prompt evaluation for palliative support and symptom management is highly recommended.",
                "双重高风险。强烈建议尽早评估姑息支持治疗和症状管理。")

        st.markdown(
            f"<div class='quadrant-box' style='background-color: {box_color};'><h3>{quadrant}</h3><p>{advice}</p></div>",
            unsafe_allow_html=True)

        st.caption(t("Note: Risk cutoffs are defined at 0.50 per the TEMPO validation study framework.",
                     "注：根据 TEMPO 验证研究框架，风险截断值设定为 0.50。"))

# --- Footer ---
st.divider()
st.caption(t("Developed based on the TEMPO study framework.", "基于 TEMPO 研究框架开发。"))