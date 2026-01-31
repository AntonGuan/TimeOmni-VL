
<h1><b>
📈 TimeOmni-VL: Unified Models for Time Series Understanding and Generation
</b></h1>

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)]()

</div>

***This repository provides two core components for unified time series multimodal models***:

1. **🧱 TSUMM-Suite (Time Series Unified Multimodal Suite)**: A data pipeline that spans time series understanding and generation.
2. **🤖 TimeOmni-VL (Time Series Unified Multimodal Model)**: A vision-centric framework that unifies time series understanding and generation.

---

## 🧱 TSUMM-Suite

TSUMM-Suite comprises six time series understanding tasks and two time series generation tasks, including:

* **Understanding Tasks**:
  1. Variable Counting
  2. Variable Y-Range
  3. Cycle Bounding Box
  4. Mean Comparison
  5. Anomaly Detection
  6. Trend Analysis

* **Generation Tasks**:
  1. Multivariate zero-shot time series forecasting
  2. Multivariate zero-shot time series imputation


📚 **Task illustration**:
<div align="center">
<img src="figs/fig4.png" width="100%"/>
</div>


Since TimeOmni-VL is a vision-centric framework, we introduce a fidelity-preserving bidirectional mapping between time series and images (***Bi-TSI***), and build the data pipeline on top of Bi-TSI.



🚀 **Usage**:
1. How to construct datasets for both Understanding and Generation tasks? Run the [demo](data_pipeline/bi_tsi/demo.ipynb) to see the dataset build process.

2. How to implement the fidelity-preserving bidirectional mapping between time series and images (Bi-TSI)? We provide an Understanding adapter at [understanding_adapter](data_pipeline/bi_tsi/understanding_adapter.py), and Generation adapters at [forecasting_adapter](data_pipeline/bi_tsi/forecasting_adapter.py) (Forecasting) and [imputation_adapter](data_pipeline/bi_tsi/imputation_adapter.py) (Imputation).

3. What is the training data format required by TimeOmni-VL? Refer to [sample_understanding](data_pipeline/demo_level_samples/understanding_sample.jsonl) for Understanding, and [sample_forecasting](data_pipeline/demo_level_samples/forecast_samples_thinking_gen.jsonl) (Forecasting) plus [sample_imputation](data_pipeline/demo_level_samples/imputation_samples_thinking_gen.jsonl) (Imputation) for Generation.

---

## 🤖 TimeOmni-VL

TimeOmni-VL employs a joint training strategy to support both TS-image-based time series understanding and generation, and is also compatible with text-only time series reasoning training data.

📏 **Loss Functions**:

  1. **Understanding Loss**: Next-token prediction loss over the chain-of-thought and final answer.
  2. **Generation Loss**: Diffusion denoising loss for generating target TS-images.


🛠️ **Installation:**
```bash
# 1. Create environment
conda create -n timeomni_vl python==3.10
conda activate timeomni_vl
```

```bash
# 2. Install dependencies
cd training
pip install -r requirements.txt
```

🚀 **Usage:**

1. How to set the dataset configuration parameters? See the dataset config at [config.yaml](training/data/configs/example.yaml).
2. How to fine-tune the base model on time series data? Run the script at [train.sh](training/scripts/train.sh).

---

## 🤔 Inference

We provide flexible inference interfaces that support both time series understanding and generation tasks.

🚀 **Usage:**

1. **Batch parallel inference script** ([inference_parallel](inference/inference_parallel.py)):

📈 **Demos:**

We also provide a user-friendly interface for interacting with the model. Below are demo runs for Understanding and Generation tasks (Forecasting and Imputation):

* **Understanding Task Demo**:

<div align="center">
<img src="demos/understanding.png" width="100%"/>
</div>

* **Generation Task 1: Forecasting Demo**:

<div align="center">
<img src="demos/forecasting.png" width="100%"/>
</div>

* **Generation Task 2: Imputation Demo**:

<div align="center">
<img src="demos/imputation.png" width="100%"/>
</div>


---


## 📦 Data Access
***To facilitate review and functional verification, we currently provide demo-level code and demo-level samples for functional verification at this stage. The complete implementation and full-scale datasets will be released upon paper acceptance.***
