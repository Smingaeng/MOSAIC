# Mosaic: Graph Recommendation via Mixture-of-Experts with LLM-based Community-Aware Intent and Conformity

## Framework

<img width="1280" height="416" alt="Framework" src="https://github.com/user-attachments/assets/8839c239-ef39-4f83-aea3-c4cdd2082b53" />


## Model Specifications

| Component | Description |
|---|---|
| LLM Model | Qwen2.5-14B-Instruct |
| Semantic Encoder | text-embedding-3-large |
| Backbone | LightGCN |
| Routing | Top-k gated Mixture-of-Experts |
| Views | Global Intent, Global Conformity, Local Intent, Local Conformity |

## Dependencies

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate mosaic_env
```

### 2. Install Additional Python Packages

```bash
pip install -r requirements.txt
```

## Dataset

We use the following datasets:

- Yelp
- Amazon-Book
- Amazon-Movie

Dataset statistics are computed using `trn_mat.pkl`, `val_mat.pkl`, and `tst_mat.pkl`.

| Dataset | Users | Items | Interactions | Density |
|---|---:|---:|---:|---:|
| Yelp | 11,091 | 11,010 | 277,535 | 0.0023 |
| Amazon-Book | 11,000 | 9,332 | 200,860 | 0.0020 |
| Amazon-Movie | 7,351 | 7,002 | 95,174 | 0.0018 |

## Dataset Download

You can download the preprocessed interaction matrices and Mosaic prior files here:

- [Google Drive]({https://drive.google.com/drive/folders/12qwMbhaVnbY9TAG7AT-VTwUPRLkx-dDR?usp=sharing})

## Train & Evaluate

Run the following command to train and evaluate Mosaic:

```bash
python encoder/train_encoder.py --model mosaic --dataset {dataset} --cuda 0
```

Available dataset names:

```bash
yelp
amazon_book
amazon_movie
```

For example:

```bash
python encoder/train_encoder.py --model mosaic --dataset amazon_book --cuda 0
```

## Hyperparameters

Hyperparameters for Mosaic and baseline models are stored in:

```text
encoder/config/modelconf/
```

For Mosaic, see:

```text
encoder/config/modelconf/mosaic.yml
```

## Acknowledgement

For fair comparison and reproducibility, we reuse parts of the IRLLRec codebases, including training/evaluation routines and related utilities.

Many thanks to the authors of the following repository for providing useful training frameworks and open-source resource:

> [IRLLRec](https://github.com/wangyu0627/IRLLRec)
