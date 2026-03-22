# TEMPO: Time-Aware Multimodal Prognostication for Lung Adenocarcinoma with Leptomeningeal Metastasis

[](https://opensource.org/licenses/MIT)

[cite_start]This repository contains the official implementation of **TEMPO**, a visit-updated multimodal deep learning model designed for dynamic prognostication in patients with lung adenocarcinoma and leptomeningeal metastasis (LM)[cite: 53, 54, 125].

## 📖 Overview

[cite_start]Leptomeningeal metastasis from lung adenocarcinoma exhibits rapid progression and clinical heterogeneity[cite: 18]. [cite_start]Routine follow-up data are inherently irregular, and multimodal data are frequently incomplete[cite: 19].

[cite_start]**TEMPO** addresses this by leveraging routinely collected longitudinal assessments without imputing absent modalities[cite: 21]. [cite_start]Instead, it uses modality masks, missingness indicators, and prespecified temporal-gap features to represent data availability[cite: 22]. The model jointly estimates two key clinical outcomes at each routine assessment:

  * [cite_start]**Short-term risk:** 8-week central nervous system (CNS) progression[cite: 158].
  * [cite_start]**Medium-term prognosis:** 6-month all-cause mortality[cite: 158].

[cite_start]In our temporal validation cohort, TEMPO demonstrated moderate discrimination for 6-month mortality (C-index, 0.69; 95% CI, 0.64-0.74) and 8-week CNS progression (time-dependent AUC, 0.70; 95% CI, 0.61-0.80)[cite: 88].

## 🧠 Model Architecture

TEMPO integrates data across multiple clinical modalities:

  * [cite_start]**Clinical Data Branch:** Processes structured clinical baselines and longitudinal dynamics with time-positional encoding[cite: 430, 431, 437].
  * [cite_start]**Image Branch:** Utilizes a `ResNet18` backbone (pretrained on MRI) to extract features from brain MRI slices and cerebrospinal fluid (CSF) cytology/pathology images[cite: 438, 449, 453, 461].
  * [cite_start]**Fusion & Prediction:** Combines the embeddings and outputs the paired risk estimates via separate prediction heads[cite: 470, 484, 503].


## 📝 Citation

If you find this code or our work useful in your research, please consider citing our paper:

```bibtex
@article{tempo2026,
  title={Time-Aware Multimodal Prognostication for Lung Adenocarcinoma with Leptomeningeal Metastasis},
  author={Wu, Qichao and Yang, Weijie and Qiu, Lei and An, Juan and Xu, Weiran and Cao, Beihe and Zheng, Yutong and Shi, Weiwei and Li, Xiaoyan},
  year={2026}
}
```

既然你之前的代码包没有成功上传，你需要我为你提供一个标准的项目目录结构建议，或者帮你编写像 `requirements.txt` 或 `train.py` 这样的代码骨架吗？
