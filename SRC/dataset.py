"""Dataset loading, preprocessing, and tokenisation helpers."""

import os

import numpy as np
from datasets import Dataset
from datasets import disable_progress_bar
from datasets import load_dataset


def map_stars_to_binary_label(row):
    """Map MARC stars to binary sentiment; keep neutral as -1 for filtering."""
    if row["stars"] <= 2:
        return {"label": 0}
    if row["stars"] >= 4:
        return {"label": 1}
    return {"label": -1}


def extract_texts(ds):
    """Build one model input string per row by merging review title and body."""
    # MARC rows must provide title+body; we merge them as one classifier input.
    required_cols = {"review_title", "review_body"}
    if not required_cols.issubset(set(ds.column_names)):
        raise ValueError("Dataset must contain review_title and review_body columns.")

    raw_texts = []
    for title, body in zip(ds["review_title"], ds["review_body"]):
        title_part = (title or "").strip()
        body_part = (body or "").strip()
        merged = f"{title_part} {body_part}".strip()
        raw_texts.append(merged)

    texts = []
    for value in raw_texts:
        if isinstance(value, str):
            texts.append(value)
        else:
            texts.append(str(value))
    return texts


def load_and_preprocess_data(dataset_name, seed, train_size, test_sizes, cache_dir):
    """Load EN train + EN/ES/JA test, then map/filter/sample fixed split sizes."""
    # Load EN train + EN/ES/JA test, map labels, drop neutral, and sample fixed sizes.
    disable_progress_bar()
    os.makedirs(cache_dir, exist_ok=True)

    train_data = load_dataset(dataset_name, "en", split="train", cache_dir=cache_dir)
    test_data = {}
    for lang in ["en", "es", "ja"]:
        test_data[lang] = load_dataset(
            dataset_name,
            lang,
            split="test",
            cache_dir=cache_dir,
        )

    train_data = train_data.map(map_stars_to_binary_label)
    train_data = train_data.filter(lambda row: row["label"] != -1)

    filtered_test = {}
    for lang, ds in test_data.items():
        mapped = ds.map(map_stars_to_binary_label)
        filtered = mapped.filter(lambda row: row["label"] != -1)
        filtered_test[lang] = filtered
    test_data = filtered_test

    train_data = train_data.shuffle(seed=seed).select(range(train_size))

    sampled_test = {}
    for lang, ds in test_data.items():
        sampled_test[lang] = ds.shuffle(seed=seed).select(range(test_sizes[lang]))
    test_data = sampled_test

    return train_data, test_data


def tokenize_datasets(tokenizer, train_data, test_data, max_len):
    """Tokenise train/test datasets and return torch-ready inputs + true labels."""
    # Build torch-ready tokenized datasets for train and per-language test splits.
    disable_progress_bar()
    train_ds = Dataset.from_dict(
        {
            "text": extract_texts(train_data),
            "label": train_data["label"],
        }
    )

    # We keep fixed max length so all batches have consistent tensor shapes.
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )

    train_tok = train_ds.map(tokenize_batch, batched=True)
    train_tok.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "label"],
    )

    test_tok = {}
    true_labels = {}
    for lang, ds in test_data.items():
        lang_ds = Dataset.from_dict({"text": extract_texts(ds), "label": ds["label"]})
        lang_tok = lang_ds.map(tokenize_batch, batched=True)
        lang_tok.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "token_type_ids", "label"],
        )
        test_tok[lang] = lang_tok
        true_labels[lang] = np.array(ds["label"])

    return train_tok, test_tok, true_labels
