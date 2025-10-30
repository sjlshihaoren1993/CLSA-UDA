# Dynamic Pseudo Labeling for Enhanced Cross-Domain PD Diagnosis via CLSA-UDA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the paper:

> **Dynamic Pseudo Labeling for Enhanced Cross-Domain PD Diagnosis via CLSA-UDA**
> *The Visual Computer* (Under Review)

---

## 🛠️ Installation & Requirements

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/CLSA-UDA.git
    cd CLSA-UDA
    ```

2.  **Create a conda environment and activate it:**
    ```bash
    conda create -n clsauda python=3.8
    conda activate clsauda
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 📊 Datasets

**⚠️ Important:** Due to data privacy and licensing agreements, we cannot distribute the raw PD datasets collected from our hospital.

*   **Source Domain:** Please refer to the [**PPMI**](<http://www.ppmi-info.org>) dataset.

To use this code, you must preprocess your data into the expected format. Dataset contains original MRI images and txt document, txt document includes image filename and corresponding label. 

---

## 🏃‍♂️ How to Run

### 1. Training

We use configuration files to manage hyperparameters. To train the model for a specific domain adaptation task:

```bash
python cdan_mcc_sdat_masking_pd_cpl.py ../../../data --epochs 30 --per-class-eval --train-resizing cen.crop --temperature 3.0 --lr 0.002 --seed 2 -a ResNet18_3D --gpu 1 --rho 0.02 --alpha 0.9 --pseudo_label_weight prob --mask_block_size 32 --mask_ratio 0.7 --log logs/cdan_mcc_sdat_ResNet18_3D_100_20_cpl/pd_ppmi2hn --log_name pd_ppmi2hn_cdan_mcc_sdat_masking_m32-ResNet18_3D_100_20_cpl --log_results

```

### 2. Evaluation

To evaluate a trained model on the target domain test set:

```bash
python test_pd.py
```

---

## 🤝 Citation

If you find this code useful in your research, please consider citing our manuscript:

```bibtex
@article{shen2025clsa-uda,
  title={Dynamic Pseudo Labeling for Enhanced Cross-Domain PD Diagnosis via CLSA-UDA},
  author={Yu Shen, Jinjin Hai, Kai Qiao, Xiangli Yang, Jian Chen, Yongli Li, Bin Yan},
  journal={The Visual Computer},
  year={2025},
  note={Under Review}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**🔗 This repository is directly associated with the manuscript currently under review at *The Visual Computer*.**

## Acknowledgements

CLSA-UDA is based on the following open-source projects. 
We thank their authors for making the source code publicly available.

* [SDAT](https://github.com/val-iisc/SDAT)
* [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library)
* [MIC](https://github.com/lhoyer/MIC)