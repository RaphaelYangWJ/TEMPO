# TEMPO: Time-Aware Multimodal Prognostication for Lung Adenocarcinoma with Leptomeningeal Metastasis

[](https://opensource.org/licenses/MIT)

This repository contains the official implementation of **TEMPO**, a visit-updated multimodal deep learning model designed for dynamic prognostication in patients with lung adenocarcinoma and leptomeningeal metastasis (LM).

## 📖 Overview

Leptomeningeal metastasis from lung adenocarcinoma exhibits rapid progression and clinical heterogeneity[cite: 18]. [cite_start]Routine follow-up data are inherently irregular, and multimodal data are frequently incomplete.

**TEMPO** addresses this by leveraging routinely collected longitudinal assessments without imputing absent modalities[cite: 21]. [cite_start]Instead, it uses modality masks, missingness indicators, and prespecified temporal-gap features to represent data availability[cite: 22]. The model jointly estimates two key clinical outcomes at each routine assessment:

  * **Short-term risk:** 8-week central nervous system (CNS) progression.
  * **Medium-term prognosis:** 6-month all-cause mortality.

In our temporal validation cohort, TEMPO demonstrated moderate discrimination for 6-month mortality (C-index, 0.69; 95% CI, 0.64-0.74) and 8-week CNS progression (time-dependent AUC, 0.70; 95% CI, 0.61-0.80).

## 🧠 Model Architecture

TEMPO integrates data across multiple clinical modalities:

  * **Clinical Data Branch:** Processes structured clinical baselines and longitudinal dynamics with time-positional encoding.
  * **Image Branch:** Utilizes a `ResNet18` backbone (pretrained on MRI) to extract features from brain MRI slices and cerebrospinal fluid (CSF) cytology/pathology images.
  * **Fusion & Prediction:** Combines the embeddings and outputs the paired risk estimates via separate prediction heads.


## 📝 Citation

If you find this code or our work useful in your research, please consider citing our paper:

```bibtex
@article{tempo2026,
  title={Time-Aware Multimodal Prognostication for Lung Adenocarcinoma with Leptomeningeal Metastasis},
  author={Wu, Qichao and Yang, Weijie and Qiu, Lei and An, Juan and Xu, Weiran and Cao, Beihe and Zheng, Yutong and Shi, Weiwei and Li, Xiaoyan},
  year={2026}
}
```
