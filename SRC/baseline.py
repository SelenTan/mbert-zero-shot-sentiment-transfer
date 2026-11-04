"""Classical baseline models and translation cache utilities."""

import hashlib
import json
import os

import numpy as np
from deep_translator import GoogleTranslator
from deep_translator.exceptions import RequestError
from deep_translator.exceptions import ServerException
from deep_translator.exceptions import TooManyRequests
from deep_translator.exceptions import TranslationNotFound
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm

from SRC.dataset import extract_texts


def translate_with_cache(ds, lang, cache_dir, bar_fmt, logger):
    """Translate one non-EN split to EN with stable cache keys and cache reuse."""
    # pylint: disable=too-many-locals,too-many-statements
    # Translate non-EN test text to EN and reuse disk cache for stable reruns.
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{lang}_to_en.json")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            cache = {str(key): str(value) for key, value in payload.items()}

    texts = extract_texts(ds)
    translator = GoogleTranslator(source=lang, target="en")
    translated_texts = []
    cache_hit = 0
    cache_miss = 0
    fallback = 0
    changed = False

    progress = tqdm(
        texts,
        desc=f"Translating {lang}",
        unit="text",
        bar_format=bar_fmt,
    )
    for idx, text in enumerate(progress):
        stripped = text.strip()
        key = ""

        # Prefer stable review_id keys to keep cache entries reusable across reruns.
        if "review_id" in ds.column_names:
            review_id = ds["review_id"][idx]
            if review_id is not None and str(review_id).strip():
                key = str(review_id)

        if not key:
            digest = hashlib.sha1(f"{lang}::{stripped}".encode("utf-8")).hexdigest()
            key = f"{lang}_{idx}_{digest}"

        if key in cache and cache[key].strip():
            translated_texts.append(cache[key])
            cache_hit += 1
            continue

        cache_miss += 1
        if not stripped:
            translated_texts.append("")
            cache[key] = ""
            changed = True
            continue

        # Keep one fallback path: any failed/empty translation reuses source text.
        try:
            translated = translator.translate(stripped)
            translated_str = str(translated).strip() if translated is not None else ""
        except (
            TranslationNotFound,
            RequestError,
            ServerException,
            TooManyRequests,
        ):
            translated_str = ""

        if not translated_str:
            translated_str = stripped
            fallback += 1

        translated_texts.append(translated_str)
        cache[key] = translated_str
        changed = True

    if changed:
        with open(cache_path, "w", encoding="utf-8") as file:
            json.dump(cache, file, ensure_ascii=False)

    logger.info(
        "[%s] cache h=%d m=%d fb=%d n=%d",
        lang,
        cache_hit,
        cache_miss,
        fallback,
        len(texts),
    )
    return translated_texts


def run_baseline_model(
    train_data,
    test_data,
    results,
    metric_cls,
    cache_dir,
    bar_fmt,
    logger,
):
    """Run majority and TF-IDF baselines, then append metrics into `results`."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    # Evaluate majority + TF-IDF baselines and append metrics for each language.
    logger.info("Running baseline models")
    # Baseline vocabulary is fit on EN train only to match zero-shot setup.
    train_texts = extract_texts(train_data)
    train_labels = np.array(train_data["label"])
    if len(train_labels) == 0:
        raise ValueError("Train labels are empty after preprocessing.")

    train_count = len(train_labels)
    majority_class = int(np.bincount(train_labels).argmax())
    vectorizer = TfidfVectorizer()
    train_features = vectorizer.fit_transform(train_texts)
    classifier = MultinomialNB()
    classifier.fit(train_features, train_labels)

    for lang, ds in test_data.items():
        true_labels = np.array(ds["label"])
        test_count = len(true_labels)

        # Majority baseline uses the dominant class in train labels.
        majority_pred = np.full(test_count, majority_class, dtype=int)
        majority_acc = accuracy_score(true_labels, majority_pred)
        majority_f1 = f1_score(true_labels, majority_pred, zero_division=0)
        majority_metric = metric_cls(
            "baseline",
            "majority_class",
            lang,
            float(majority_acc),
            float(majority_f1),
            train_count,
            test_count,
        )
        results.append(majority_metric)

        if lang == "en":
            eval_texts = extract_texts(ds)
            experiment_name = "tfidf_nb_en"
        else:
            # ES/JA are translated to EN before reusing the same EN-trained vectorizer.
            eval_texts = translate_with_cache(ds, lang, cache_dir, bar_fmt, logger)
            experiment_name = "translation_tfidf_nb"

        pred_labels = classifier.predict(vectorizer.transform(eval_texts))
        tfidf_acc = accuracy_score(true_labels, pred_labels)
        tfidf_f1 = f1_score(true_labels, pred_labels, zero_division=0)
        tfidf_metric = metric_cls(
            "baseline",
            experiment_name,
            lang,
            float(tfidf_acc),
            float(tfidf_f1),
            train_count,
            test_count,
        )
        results.append(tfidf_metric)
