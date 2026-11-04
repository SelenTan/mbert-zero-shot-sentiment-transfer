"""Helpers for selecting and annotating qualitative misclassification examples."""

import numpy as np
import pandas as pd

from SRC.dataset import extract_texts


def _katakana_ratio(text):
    """Return katakana character ratio used as a lightweight JA sparsity cue."""
    # Compute the ratio of katakana-like characters as a sparsity proxy for JA.
    if len(text) == 0:
        return 0.0
    katakana_count = 0
    for char in text:
        code = ord(char)
        if 0x30A0 <= code <= 0x30FF:
            katakana_count += 1
        elif 0xFF66 <= code <= 0xFF9F:
            katakana_count += 1
    return katakana_count / len(text)


def _guess_error_source(text, true_label, pred_label, lang):
    """Return one heuristic short explanation for a given misclassification."""
    # pylint: disable=too-many-return-statements
    # Provide a concise, heuristic explanation for one misclassification.
    stripped = text.strip()
    lower = stripped.lower()

    if lang == "ja":
        if _katakana_ratio(stripped) >= 0.22:
            return "High katakana/loanword ratio causes sparse subword coverage."
        if any(
            token in stripped
            for token in ["けど", "しかし", "なのに", "一方で", "ただ"]
        ):
            return "Contrastive discourse flips polarity after local sentiment cues."
        if len(stripped) >= 120:
            return "Long review mixes multiple stances and hurts global polarity."
        return "Meaning depends on context or implied tone beyond surface words."

    if true_label == 0 and pred_label == 1:
        if any(token in lower for token in ["pero", "aunque", "sin embargo"]):
            return "Polarity flips after contrast; model overweights early praise."
        if any(token in lower for token in ["genial", "perfecto", "excelente"]):
            return "Possible sarcasm or contrast makes positive words misleading."
        return "Dense local positive terms hide a negative final conclusion."

    if any(token in lower for token in ["no ", "nunca", "jamás", "ninguno"]):
        return "Negation scope is complex and polarity propagation fails."
    return "Sentiment is implicit; explicit polarity words are too weak."


def _select_error_indices(y_true, y_pred, max_examples):
    """Select up to `max_examples` errors with balanced FP/FN preference."""
    # Prefer a balanced selection of FP/FN; fall back to earliest errors if needed.
    fp_idx = [i for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1]
    fn_idx = [i for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0]
    selected = []

    fp_take = min(len(fp_idx), max_examples // 2)
    fn_take = min(len(fn_idx), max_examples // 2)
    selected.extend(fp_idx[:fp_take])
    selected.extend(fn_idx[:fn_take])

    if len(selected) < max_examples:
        all_err = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
        for idx in all_err:
            if idx not in selected:
                selected.append(idx)
            if len(selected) == max_examples:
                break
    return selected


def build_error_analysis_examples(test_data, best_logits):
    """Build a compact ES/JA misclassification table for qualitative discussion."""
    # Extract 4 ES and 4 JA misclassified reviews with heuristic error reasons.
    rows = []
    for lang in ["es", "ja"]:
        texts = extract_texts(test_data[lang])
        y_true = np.array(test_data[lang]["label"], dtype=np.int32)
        y_pred = np.argmax(best_logits[lang], axis=1).astype(np.int32)
        chosen = _select_error_indices(y_true, y_pred, max_examples=4)

        for idx in chosen:
            text = str(texts[idx])
            rows.append(
                {
                    "language": lang,
                    "original_text": text,
                    "true_label": int(y_true[idx]),
                    "predicted_label": int(y_pred[idx]),
                    "error_source": _guess_error_source(
                        text,
                        int(y_true[idx]),
                        int(y_pred[idx]),
                        lang,
                    ),
                }
            )
    return pd.DataFrame(rows)
