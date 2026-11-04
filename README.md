# Zero-Shot Cross-Lingual Sentiment Transfer with mBERT

## Project Overview

This project studies zero-shot cross lingual transfer for sentiment analysis.
The core question is whether a model fine-tuned only on English reviews can
generalise to Spanish and Japanese without using any labelled target language
training data.

The task uses a Parquet version of the Multilingual Amazon Reviews Corpus
(MARC) from Hugging Face and is framed as binary sentiment classification:

- Positive (`1`): 4-5 stars
- Negative (`0`): 1-2 stars

Neutral 3-star reviews are removed for the binary polarity setup.

## Models and Experiments

The project compares:

- Majority class baseline
- TF-IDF + Multinomial Naive Bayes
- Translation-based TF-IDF + Multinomial Naive Bayes
- mBERT (`bert-base-multilingual-cased`) with layer-freezing ablations

Experiments are run with three seeds (`42`, `43`, `44`), and
results are reported as mean +/- standard deviation.

The neural ablation compares:

- `train_bottom_half`: train bottom half of encoder + classification head
- `train_top_half`: train top half of encoder + classification head

Additional analyses include:

- Lexical overlap masking on Spanish and Japanese test sets
- Token fertility analysis by tokenised sequence-length bins
- CLS latent alignment visualisation using 2D PCA
- Selected qualitative error analysis

## Repository Structure

```text
mbert-zero-shot-sentiment-transfer/
|-- SRC/
|   |-- dataset.py            # Data loading, preprocessing, and tokenisation
|   |-- baseline.py           # Majority and translation-based baselines
|   |-- ablation.py           # Neural ablation + lexical/fertility analyses
|   |-- reporting.py          # Result-table aggregation and plotting
|   `-- error_analysis.py     # Qualitative error-case extraction
|-- environment.yml           # Conda environment spec
|-- requirements.txt          # Pip dependency list
|-- main.py                   # Single entry point
`-- README.md                 # Project documentation
```

## Installation and Setup

1. Create the environment:

```bash
conda env create -f environment.yml
```

2. Activate the environment:

```bash
conda activate mbert-sentiment
```

No extra `pip install -r requirements.txt` step is needed, because
`environment.yml` already includes it.

3. Run the pipeline:

```bash
python main.py
```

## Outputs

Generated outputs are saved under:

- `outputs_csv/`
- `plots/`

Main aggregated result table:

- `outputs_csv/results.csv`

## Required Packages

Core dependencies:

- `torch`
- `transformers`
- `accelerate>=1.1.0`
- `datasets>=3,<4`
- `scikit-learn`
- `deep-translator`
- `matplotlib`
- `numpy`
- `pandas`
- `tqdm`

## Third-Party Code Disclosure

This project uses public third-party resources and libraries:

- Hugging Face model hub for pretrained mBERT
- Hugging Face datasets hub for MARC (`neonwatty/amazon_reviews_multi`)
- Standard ML/NLP libraries for baselines and evaluation
