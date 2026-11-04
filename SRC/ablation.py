"""Neural ablation pipeline and analysis helpers for cross-lingual transfer."""

import os
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments
from transformers.trainer_callback import ProgressCallback
from transformers.utils import logging as hf_logging
from transformers.utils.versions import require_version

from SRC.dataset import extract_texts
from SRC.dataset import tokenize_datasets


class QuietTrainer(Trainer):
    """Trainer subclass that keeps compact logging without verbose callbacks."""

    def log(self, logs, *_args, **_kwargs):
        # Keep trainer log history for report tables while avoiding verbose print callbacks.
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)


def _extract_cls_embeddings(model, tokenizer, texts, max_len, batch_size):
    """Extract [CLS] encoder embeddings for a batchable list of texts."""
    # Extract [CLS] embeddings from the encoder output for a list of texts.
    # We run inference in mini-batches to keep memory usage stable.
    if len(texts) == 0:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)

    model.eval()
    device = next(model.parameters()).device
    cls_chunks = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        encoded = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        # We use base_model to obtain token-level hidden states before classifier head.
        with torch.no_grad():
            outputs = model.base_model(**encoded)
            cls_tensor = outputs.last_hidden_state[:, 0, :]
        cls_chunks.append(cls_tensor.cpu().numpy())

    return np.concatenate(cls_chunks, axis=0).astype(np.float32)


def _tokenize_eval_dataset(tokenizer, data_split, max_len):
    """Tokenise one evaluation split and return torch-formatted dataset."""
    # Tokenize one held-out split for validation using the same max_length policy.
    texts = extract_texts(data_split)
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )
    data_map = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "token_type_ids": encoded["token_type_ids"],
        "label": data_split["label"],
    }
    eval_ds = Dataset.from_dict(data_map)
    eval_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "label"],
    )
    return eval_ds


def _resolve_validation_size(train_size):
    """Return validation size: 500 in full mode, adaptive in reduced mode."""
    if train_size <= 1:
        raise ValueError("Train split is too small to create train/validation sets.")
    if train_size > 500:
        return 500

    # For smaller training sets, use 20% validation with a floor of 50 samples.
    val_size = max(50, int(round(train_size * 0.2)))
    val_size = min(val_size, train_size - 1)
    return val_size


def _split_train_validation(train_data, seed):
    """Split EN train data into fit/validation subsets with deterministic seed."""
    # Use adaptive validation size for smaller training splits.
    val_size = _resolve_validation_size(len(train_data))
    split = train_data.train_test_split(
        test_size=val_size,
        seed=seed,
        shuffle=True,
    )
    return split["train"], split["test"]


def _dataset_signature(data_split):
    """Compute deterministic split hash for checkpoint compatibility checks."""
    # Build a stable hash for split contents to verify checkpoint compatibility.
    sha = hashlib.sha1()
    labels = data_split["label"]
    if "review_id" in data_split.column_names:
        keys = [str(value) for value in data_split["review_id"]]
    else:
        keys = extract_texts(data_split)

    for key, label in zip(keys, labels):
        sha.update(str(key).encode("utf-8"))
        sha.update(b"|")
        sha.update(str(int(label)).encode("utf-8"))
        sha.update(b"\n")
    return sha.hexdigest()


def _build_checkpoint_metadata(
    model_name,
    max_len,
    epochs,
    tr_bs,
    ev_bs,
    seed,
    train_signature,
    val_signature,
    eval_strategy,
    eval_steps,
    logging_steps,
):
    """Build checkpoint metadata used to validate reuse compatibility."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # Record settings that must match before reusing a cached checkpoint.
    return {
        "model_name": model_name,
        "max_len": int(max_len),
        "epochs": int(epochs),
        "train_batch_size": int(tr_bs),
        "eval_batch_size": int(ev_bs),
        "seed": int(seed),
        "train_signature": train_signature,
        "val_signature": val_signature,
        "eval_strategy": str(eval_strategy),
        "eval_steps": int(eval_steps),
        "logging_steps": int(logging_steps),
    }


def _load_checkpoint_if_compatible(checkpoint_dir, expected_meta, logger):
    """Load checkpoint only when metadata exactly matches current run settings."""
    # Load a saved model only when metadata exactly matches current run setup.
    meta_path = os.path.join(checkpoint_dir, "meta.json")
    if not os.path.isdir(checkpoint_dir):
        return None
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, "r", encoding="utf-8") as file:
        saved_meta = json.load(file)
    if saved_meta != expected_meta:
        logger.info("Checkpoint mismatch, retraining this mode.")
        return None

    logger.info("Reusing checkpoint: %s", checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    return model


def _save_checkpoint(model, tokenizer, checkpoint_dir, meta_payload):
    """Persist model/tokenizer plus metadata for deterministic future reuse."""
    # Save final model/tokenizer plus metadata for future no-retrain reuse.
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(
        checkpoint_dir,
        safe_serialization=False,
    )
    tokenizer.save_pretrained(checkpoint_dir)
    meta_path = os.path.join(checkpoint_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(meta_payload, file, ensure_ascii=False, indent=2)


def _collect_train_val_error_rows(log_history, run_name):
    """Flatten Trainer log_history into train/validation loss table rows."""
    # Convert Trainer log history into a flat table for report error curves.
    rows = []
    for item in log_history:
        train_loss = item.get("loss")
        val_loss = item.get("eval_loss")
        if train_loss is None and val_loss is None:
            continue

        rows.append(
            {
                "run_name": run_name,
                "step": int(item.get("step", -1)),
                "epoch": float(item.get("epoch", -1.0)),
                "training_error": (
                    float(train_loss)
                    if train_loss is not None
                    else np.nan
                ),
                "validation_error": (
                    float(val_loss)
                    if val_loss is not None
                    else np.nan
                ),
            }
        )
    return rows


def _load_logged_error_rows_from_checkpoint(run_dir, run_name):
    """Recover logged train/eval losses from latest HuggingFace checkpoint state."""
    # Recover step-level train/validation logs from the latest HF checkpoint.
    # This keeps loss-curve plotting available even when we reuse checkpoints
    # and skip fresh training in the current run.
    checkpoint_names = []
    for name in os.listdir(run_dir):
        if not name.startswith("checkpoint-"):
            continue
        suffix = name.split("-", maxsplit=1)[-1]
        if not suffix.isdigit():
            continue
        checkpoint_names.append((int(suffix), name))
    if len(checkpoint_names) == 0:
        return []

    checkpoint_names.sort(reverse=True)
    latest_name = checkpoint_names[0][1]
    state_path = os.path.join(run_dir, latest_name, "trainer_state.json")
    if not os.path.exists(state_path):
        return []

    with open(state_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    history = payload.get("log_history", [])
    if not isinstance(history, list):
        return []
    return _collect_train_val_error_rows(history, run_name)


def _plot_cls_alignment(model, tokenizer, test_data, max_len, plots_dir):
    """Project multilingual [CLS] embeddings to 2D PCA for alignment inspection."""
    # pylint: disable=too-many-locals
    # Build a 2D PCA view of [CLS] embeddings to inspect cross-lingual alignment.
    lang_order = []
    for lang in ["en", "es", "ja"]:
        if lang in test_data and len(test_data[lang]) > 0:
            lang_order.append(lang)
    if len(lang_order) == 0:
        return

    embedding_blocks = []
    lang_blocks = []
    for lang in lang_order:
        texts = extract_texts(test_data[lang])
        cls_emb = _extract_cls_embeddings(
            model,
            tokenizer,
            texts,
            max_len,
            batch_size=32,
        )
        embedding_blocks.append(cls_emb)
        lang_blocks.extend([lang] * len(cls_emb))

    all_embeddings = np.concatenate(embedding_blocks, axis=0)
    reducer = PCA(n_components=2)
    projected = reducer.fit_transform(all_embeddings)
    lang_array = np.array(lang_blocks)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"en": "#1f77b4", "es": "#8833e9", "ja": "#d627aa"}
    for lang in lang_order:
        idx = np.where(lang_array == lang)[0]
        ax.scatter(
            projected[idx, 0],
            projected[idx, 1],
            s=10,
            alpha=0.35,
            label=lang,
            c=colors.get(lang, "#333333"),
        )

    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    ax.set_title("mBERT [CLS] Latent Alignment")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path = os.path.join(plots_dir, "latent_alignment_cls_pca.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _predict_without_progress(trainer, dataset):
    """Run prediction without extra progress-bar noise from Trainer callbacks."""
    # Keep progress bars for training only; suppress bars during predict calls.
    prev_disable_tqdm = trainer.args.disable_tqdm
    had_progress_callback = any(
        isinstance(callback, ProgressCallback)
        for callback in trainer.callback_handler.callbacks
    )
    trainer.args.disable_tqdm = True
    if had_progress_callback:
        trainer.remove_callback(ProgressCallback)

    try:
        output = trainer.predict(dataset)
    finally:
        if had_progress_callback:
            trainer.add_callback(ProgressCallback)
        trainer.args.disable_tqdm = prev_disable_tqdm
    return output.predictions


def _build_training_args(
    out_dir,
    run_name,
    epochs,
    tr_bs,
    ev_bs,
    seed,
    enable_checkpoint,
):
    """Create TrainingArguments with version-compatible fallbacks."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # Keep trainer output minimal while preserving progress bars during training.
    save_strategy = "epoch" if enable_checkpoint else "no"
    base_kwargs = {
        "output_dir": os.path.join(out_dir, f"structural_semantic_{run_name}"),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": tr_bs,
        "per_device_eval_batch_size": ev_bs,
        "fp16": False,
        "save_strategy": save_strategy,
        "logging_strategy": "steps",
        "logging_steps": 50,
        "disable_tqdm": False,
        "report_to": [],
        "seed": seed,
        "eval_steps": 50,
    }
    attempt_kwargs = dict(base_kwargs)
    attempt_kwargs["eval_strategy"] = "steps"
    attempt_kwargs["save_safetensors"] = False

    try:
        return TrainingArguments(**attempt_kwargs)
    except TypeError:
        # Older transformers may not support save_safetensors/eval_strategy.
        attempt_kwargs.pop("save_safetensors", None)
        try:
            return TrainingArguments(**attempt_kwargs)
        except TypeError:
            attempt_kwargs.pop("eval_strategy", None)
            attempt_kwargs["evaluation_strategy"] = "steps"
            return TrainingArguments(**attempt_kwargs)


def _apply_freeze_mode(model, mode):
    """Freeze selected encoder half for top/bottom layer ablation modes."""
    # Freeze either top or bottom half of encoder layers for structural analysis.
    for param in model.parameters():
        param.requires_grad = True

    layers = model.base_model.encoder.layer
    midpoint = len(layers) // 2
    if mode == "freeze_top_half":
        target_layers = layers[midpoint:]
    else:
        target_layers = layers[:midpoint]

    for layer in target_layers:
        for param in layer.parameters():
            param.requires_grad = False


def _collect_english_subword_ids(train_data, tokenizer, max_len):
    """Collect EN subword vocabulary set used for lexical overlap masking."""
    # Collect EN train subword ids as the lexical overlap reference vocabulary.
    english_ids = set()
    for text in extract_texts(train_data):
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )["input_ids"]
        english_ids.update(token_ids)
    return english_ids


def _mask_tokens_for_one_row(input_ids, attn_mask, special_ids, english_ids, mask_id):
    """Mask EN-overlap tokens for one encoded row and return overlap stats."""
    replaced_ids = []
    shared_count = 0
    active_count = 0

    # Only non-padding and non-special tokens are counted for overlap ratio.
    for token_id, attn_value in zip(input_ids, attn_mask):
        if attn_value == 0:
            replaced_ids.append(token_id)
            continue
        if token_id in special_ids:
            replaced_ids.append(token_id)
            continue

        active_count += 1
        if token_id in english_ids:
            shared_count += 1
            replaced_ids.append(mask_id)
            continue
        replaced_ids.append(token_id)

    return replaced_ids, shared_count, active_count


def _build_masking_outputs(encoded, special_ids, english_ids, mask_id):
    """Build masked token ids and overlap statistics for a full encoded split."""
    masked_input_ids = []
    overlap_ratio = []
    shared_count_list = []

    for input_ids, attn_mask in zip(encoded["input_ids"], encoded["attention_mask"]):
        replaced_ids, shared_count, active_count = _mask_tokens_for_one_row(
            input_ids,
            attn_mask,
            special_ids,
            english_ids,
            mask_id,
        )
        masked_input_ids.append(replaced_ids)
        shared_count_list.append(shared_count)
        ratio = 0.0 if active_count == 0 else shared_count / active_count
        overlap_ratio.append(ratio)

    return (
        masked_input_ids,
        np.array(overlap_ratio, dtype=np.float32),
        np.array(shared_count_list, dtype=np.int32),
    )


def _build_masked_eval_dataset(texts, labels, tokenizer, max_len, english_ids):
    """Mask EN-overlap subwords in non-EN text and return torch eval dataset."""
    # Build masked eval data by replacing EN-overlap subwords with [MASK].
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer does not provide [MASK] token id.")

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )
    mask_id = tokenizer.mask_token_id
    special_ids = set(tokenizer.all_special_ids)
    masked_input_ids, overlap_ratio_arr, shared_count_arr = _build_masking_outputs(
        encoded,
        special_ids,
        english_ids,
        mask_id,
    )

    masked_ds = Dataset.from_dict(
        {
            "input_ids": masked_input_ids,
            "attention_mask": encoded["attention_mask"],
            "token_type_ids": encoded["token_type_ids"],
            "label": labels,
        }
    )
    masked_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "label"],
    )

    return masked_ds, overlap_ratio_arr, shared_count_arr


def _build_readable_bins(max_token_len, min_width=5, target_bin_count=10):
    """Build readable histogram bins for token-fertility plots."""
    # Build adaptive bin edges so the x-axis keeps a readable number of bars.
    if max_token_len <= 0:
        return np.array([0, min_width], dtype=np.int32)

    raw_width = int(np.ceil(max_token_len / target_bin_count))
    width = int(np.ceil(raw_width / min_width) * min_width)
    width = max(width, min_width)
    bins = np.arange(0, max_token_len + width + 1, width, dtype=np.int32)
    if len(bins) < 2:
        bins = np.array([0, width], dtype=np.int32)
    return bins


def _compute_binned_error_rates(token_len_arr, is_error_arr, bins):
    """Compute error-rate value for each token-length bin interval."""
    # Compute one mean error rate per bin and build compact interval labels.
    labels = []
    values = []
    for left, right in zip(bins[:-1], bins[1:]):
        in_bin = (token_len_arr >= left) & (token_len_arr < right)
        if np.any(in_bin):
            rate = float(np.mean(is_error_arr[in_bin]))
        else:
            rate = 0.0
        labels.append(f"{left}-{right - 1}")
        values.append(rate)
    return labels, values


def run_neural_experiments(
    train_data,
    test_data,
    results,
    metric_cls,
    model_name,
    max_len,
    epochs,
    tr_bs,
    ev_bs,
    seed,
    out_dir,
    plots_dir,
    logger,
    enable_checkpoint=True,
):
    """Run neural ablations and downstream analyses; append metrics in-place."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    # Train two structural settings and keep the best ES/JA transfer checkpoint.
    # Then run the report-required analyses from that single selected run.
    require_version(
        "accelerate>=1.1.0",
        "Please install accelerate>=1.1.0 to run Trainer.",
    )
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()

    logger.info("Running neural experiments")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    fit_train_data, val_data = _split_train_validation(train_data, seed)
    logger.info(
        "Neural split: train=%d validation=%d",
        len(fit_train_data),
        len(val_data),
    )

    with tqdm(
        total=2,
        desc="Neural split prep",
        unit="step",
        leave=False,
    ) as prep_bar:
        # Use split train for fitting, and keep 500 EN for validation monitoring.
        train_tok, test_tok, true_labels = tokenize_datasets(
            tokenizer,
            fit_train_data,
            test_data,
            max_len,
        )
        prep_bar.update(1)
        val_tok = _tokenize_eval_dataset(tokenizer, val_data, max_len)
        prep_bar.update(1)
    train_signature = _dataset_signature(fit_train_data)
    val_signature = _dataset_signature(val_data)

    modes = [
        ("freeze_top_half", "train_bottom_half", "Train bottom half + head"),
        ("freeze_bottom_half", "train_top_half", "Train top half + head"),
    ]
    eval_strategy = "steps"
    eval_steps = 50
    logging_steps = 50
    best_logits = {}
    best_name = ""
    best_score = -1.0
    best_trainer = None
    error_rows = []

    for mode, run_name, mode_label in modes:
        run_dir = os.path.join(out_dir, f"structural_semantic_{run_name}")
        checkpoint_dir = os.path.join(run_dir, "checkpoint_final")
        meta_payload = _build_checkpoint_metadata(
            model_name,
            max_len,
            epochs,
            tr_bs,
            ev_bs,
            seed,
            train_signature,
            val_signature,
            eval_strategy,
            eval_steps,
            logging_steps,
        )
        model = None
        if enable_checkpoint:
            model = _load_checkpoint_if_compatible(
                checkpoint_dir,
                meta_payload,
                logger,
            )
        train_args = _build_training_args(
            out_dir,
            run_name,
            epochs,
            tr_bs,
            ev_bs,
            seed,
            enable_checkpoint,
        )
        trained_now = False

        # Fallback 1: checkpoint missing/mismatch -> retrain this mode.
        if model is None:
            # Rebuild the model each run so freeze settings do not leak across modes.
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
            )
            _apply_freeze_mode(model, mode)
            trained_now = True

        trainer = QuietTrainer(
            model=model,
            args=train_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            processing_class=tokenizer,
        )
        if trained_now:
            trainer.train()
            if enable_checkpoint:
                _save_checkpoint(
                    trainer.model,
                    tokenizer,
                    checkpoint_dir,
                    meta_payload,
                )
            error_rows.extend(
                _collect_train_val_error_rows(
                    trainer.state.log_history,
                    run_name,
                )
            )
        else:
            if enable_checkpoint:
                # If we skip training, recover historical loss logs from checkpoint.
                error_rows.extend(
                    _load_logged_error_rows_from_checkpoint(
                        run_dir,
                        run_name,
                    )
                )
            # Add one fresh validation point so this run is always represented.
            val_loss = float(trainer.evaluate(val_tok)["eval_loss"])
            error_rows.append(
                {
                    "run_name": run_name,
                    "step": -1,
                    "epoch": float(epochs),
                    "training_error": np.nan,
                    "validation_error": val_loss,
                }
            )

        logits_map = {}
        acc_map = {}
        for lang, ds in test_tok.items():
            logits = _predict_without_progress(trainer, ds)
            logits_map[lang] = logits

            y_pred = np.argmax(logits, axis=1)
            y_true = true_labels[lang]
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            acc_map[lang] = float(acc)

            metric = metric_cls(
                "structural_semantic",
                run_name,
                lang,
                float(acc),
                float(f1),
                len(fit_train_data),
                len(y_true),
            )
            results.append(metric)

        es_acc = acc_map.get("es", 0.0)
        ja_acc = acc_map.get("ja", 0.0)
        avg_acc = float(np.mean([es_acc, ja_acc]))
        logger.info(
            "[%s]: en=%.4f es=%.4f ja=%.4f avg_es_ja=%.4f",
            mode_label,
            acc_map.get("en", 0.0),
            es_acc,
            ja_acc,
            avg_acc,
        )

        # Select the model that transfers best on ES/JA, not the EN-only winner.
        if avg_acc > best_score:
            best_score = avg_acc
            best_name = run_name
            best_logits = logits_map
            best_trainer = trainer

    if best_trainer is None:
        raise ValueError("Failed to obtain best trainer for downstream analyses.")

    _plot_cls_alignment(
        best_trainer.model,
        tokenizer,
        test_data,
        max_len,
        plots_dir,
    )
    logger.info("Running lexical overlap ablation for ES/JA")

    english_ids = _collect_english_subword_ids(train_data, tokenizer, max_len)
    for lang in ["es", "ja"]:
        texts = extract_texts(test_data[lang])
        y_true = np.array(test_data[lang]["label"])
        base_pred = np.argmax(best_logits[lang], axis=1)
        base_acc = accuracy_score(y_true, base_pred)

        masked_ds, _, _ = _build_masked_eval_dataset(
            texts,
            test_data[lang]["label"],
            tokenizer,
            max_len,
            english_ids,
        )
        masked_logits = _predict_without_progress(best_trainer, masked_ds)
        masked_pred = np.argmax(masked_logits, axis=1)
        masked_acc = accuracy_score(y_true, masked_pred)
        masked_f1 = f1_score(y_true, masked_pred, zero_division=0)
        acc_drop = float(base_acc - masked_acc)

        # Record masked performance into the shared summary table.
        metric = metric_cls(
            "lexical_overlap_ablation",
            f"{best_name}_mask_shared_subwords",
            lang,
            float(masked_acc),
            float(masked_f1),
            len(fit_train_data),
            len(y_true),
        )
        results.append(metric)

        logger.info(
            "[lexical_ablation][%s] base_acc=%.4f masked_acc=%.4f acc_drop=%.4f",
            lang,
            base_acc,
            masked_acc,
            acc_drop,
        )

    logger.info("Running token fertility analysis for ES/JA")

    # Token fertility checks whether longer tokenized samples correlate with errors.
    for lang in ["es", "ja"]:
        texts = extract_texts(test_data[lang])
        y_true = np.array(test_data[lang]["label"])
        y_pred = np.argmax(best_logits[lang], axis=1)

        token_len = []
        is_error = []
        for idx, text in enumerate(texts):
            token_ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_len,
            )["input_ids"]
            length = len(token_ids)
            error = int(y_pred[idx] != y_true[idx])

            token_len.append(length)
            is_error.append(error)

        # Build adaptive-width bins to keep fertility plots visually readable.
        token_len_arr = np.array(token_len, dtype=np.int32)
        is_error_arr = np.array(is_error, dtype=np.float32)
        max_len_lang = int(np.max(token_len_arr))
        bins = _build_readable_bins(max_len_lang)
        labels, values = _compute_binned_error_rates(
            token_len_arr,
            is_error_arr,
            bins,
        )

        mean_error = float(np.mean(is_error)) if len(is_error) > 0 else 0.0
        mean_tokens = float(np.mean(token_len)) if len(token_len) > 0 else 0.0
        logger.info(
            "[fertility][%s] mean_error=%.4f mean_tokens=%.2f n=%d",
            lang,
            mean_error,
            mean_tokens,
            len(token_len),
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(labels, values, color="#1f77b4")
        ax.set_xlabel("Subword Length Range")
        ax.set_ylabel("Error Rate")
        ax.set_title(f"Token Fertility vs Error Rate ({lang})")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(
            os.path.join(plots_dir, f"token_fertility_error_rate_{lang}.png"),
            dpi=300,
        )
        plt.close(fig)

    return best_logits, error_rows
