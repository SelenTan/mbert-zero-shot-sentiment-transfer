"""Result aggregation and figure generation for report-ready outputs."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _metric_mean_std(values):
    """Return mean and sample std while ignoring NaN values."""
    # Compute mean and sample std, ignoring missing values.
    arr = np.array(values, dtype=np.float32)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return np.nan, np.nan
    if len(valid) == 1:
        return float(valid[0]), 0.0
    return float(np.mean(valid)), float(np.std(valid, ddof=1))


def _values_by_seed(
    summary_long,
    seeds,
    group_name,
    exp_name,
    lang,
    metric_name,
):
    """Collect one metric value per seed for one group/experiment/language key."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # Fetch one metric value per seed for a specific model-language setting.
    values = []
    for seed in seeds:
        matched = summary_long[
            (summary_long["seed"] == seed)
            & (summary_long["experiment_group"] == group_name)
            & (summary_long["experiment_name"] == exp_name)
            & (summary_long["language"] == lang)
        ]
        if len(matched) == 0:
            values.append(np.nan)
        else:
            values.append(float(matched.iloc[0][metric_name]))
    return values


def build_results_table(summary_long, seeds):
    """Aggregate per-seed rows into mean/std result columns."""
    # Build one report table with per-seed metrics and Mean +/- Std columns.
    group_cols = ["experiment_group", "experiment_name", "language"]
    rows = []
    grouped = summary_long.groupby(group_cols, dropna=False, sort=True)
    for keys, group_df in grouped:
        row = {
            "experiment_group": keys[0],
            "experiment_name": keys[1],
            "language": keys[2],
            "train_samples": int(group_df.iloc[0]["train_samples"]),
            "test_samples": int(group_df.iloc[0]["test_samples"]),
        }
        for metric_name in ["accuracy", "f1"]:
            metric_values = _values_by_seed(
                summary_long,
                seeds,
                keys[0],
                keys[1],
                keys[2],
                metric_name,
            )
            for idx, seed in enumerate(seeds):
                row[f"{metric_name}_seed_{seed}"] = metric_values[idx]
            mean_val, std_val = _metric_mean_std(metric_values)
            row[f"{metric_name}_mean"] = mean_val
            row[f"{metric_name}_std"] = std_val
            row[f"{metric_name}_mean_std"] = f"{mean_val:.4f} +/- {std_val:.4f}"
        rows.append(row)

    return pd.DataFrame(rows)


def build_results_export_table(summary_stats):
    """Drop redundant aggregate numeric columns before CSV export."""
    # Keep compact report columns and drop redundant numeric mean/std columns.
    drop_cols = [
        "accuracy_mean",
        "accuracy_std",
        "f1_mean",
        "f1_std",
    ]
    keep_cols = [col for col in summary_stats.columns if col not in drop_cols]
    return summary_stats[keep_cols].copy()


def build_validation_results_table(all_error_rows, seeds):
    """Aggregate final train/validation losses per structural run across seeds."""
    # pylint: disable=too-many-locals
    # Aggregate final train/validation errors per structural run across seeds.
    if len(all_error_rows) == 0:
        return pd.DataFrame()

    error_df = pd.DataFrame(all_error_rows)
    if len(error_df) == 0:
        return pd.DataFrame()

    rows = []
    run_names = sorted(error_df["run_name"].dropna().unique().tolist())
    for run_name in run_names:
        row = {"experiment_name": run_name}
        val_values = []
        train_values = []
        for seed in seeds:
            seed_df = error_df[
                (error_df["seed"] == seed)
                & (error_df["run_name"] == run_name)
            ].copy()
            if len(seed_df) == 0:
                val_values.append(np.nan)
                train_values.append(np.nan)
                continue

            seed_df = seed_df.sort_values(["step", "epoch"])
            val_rows = seed_df[~seed_df["validation_error"].isna()]
            train_rows = seed_df[~seed_df["training_error"].isna()]
            if len(val_rows) > 0:
                val_values.append(float(val_rows.iloc[-1]["validation_error"]))
            else:
                val_values.append(np.nan)
            if len(train_rows) > 0:
                train_values.append(float(train_rows.iloc[-1]["training_error"]))
            else:
                train_values.append(np.nan)

        for idx, seed in enumerate(seeds):
            row[f"validation_error_seed_{seed}"] = val_values[idx]
            row[f"training_error_seed_{seed}"] = train_values[idx]

        val_mean, val_std = _metric_mean_std(val_values)
        train_mean, train_std = _metric_mean_std(train_values)
        row["validation_error_mean_std"] = f"{val_mean:.4f} +/- {val_std:.4f}"
        row["training_error_mean_std"] = f"{train_mean:.4f} +/- {train_std:.4f}"
        rows.append(row)

    return pd.DataFrame(rows)


def _get_mean_std(summary_stats, group_name, exp_name, lang, metric_name):
    """Read one mean/std metric pair from the summary table with strict keys."""
    # Read mean/std values from the aggregated table with strict key matching.
    matched = summary_stats[
        (summary_stats["experiment_group"] == group_name)
        & (summary_stats["experiment_name"] == exp_name)
        & (summary_stats["language"] == lang)
    ]
    if len(matched) == 0:
        raise ValueError(
            f"Missing stats row for group={group_name}, "
            f"experiment={exp_name}, language={lang}."
        )
    mean_val = float(matched.iloc[0][f"{metric_name}_mean"])
    std_val = float(matched.iloc[0][f"{metric_name}_std"])
    return mean_val, std_val


def _best_structural_run(summary_long):
    """Select best structural run by average ES/JA accuracy across seeds."""
    # Select the structural run with the highest ES/JA mean accuracy across seeds.
    structural = summary_long[summary_long["experiment_group"] == "structural_semantic"]
    target_rows = structural[structural["language"].isin(["es", "ja"])]
    mean_by_run = target_rows.groupby(
        "experiment_name",
        dropna=False,
    )["accuracy"].mean()
    return str(mean_by_run.idxmax())


def _best_structural_run_for_seed(summary_long, seed):
    """Select best structural run by average ES/JA accuracy within one seed."""
    # Select the best structural run inside one seed by ES/JA mean accuracy.
    structural = summary_long[
        (summary_long["seed"] == seed)
        & (summary_long["experiment_group"] == "structural_semantic")
    ]
    target_rows = structural[structural["language"].isin(["es", "ja"])]
    mean_by_run = target_rows.groupby(
        "experiment_name",
        dropna=False,
    )["accuracy"].mean()
    return str(mean_by_run.idxmax())


def plot_main_result_comparison(summary_long, summary_stats, plots_dir):
    """Plot headline model comparison with mean/std accuracy error bars."""
    # pylint: disable=too-many-locals
    # Plot Mean +/- Std accuracy points for the three headline model families.
    models = ["Majority", "Translation baseline", "mBERT best"]
    langs = ["en", "es", "ja"]
    best_run = _best_structural_run(summary_long)

    means = np.zeros((len(models), len(langs)), dtype=np.float32)
    stds = np.zeros((len(models), len(langs)), dtype=np.float32)
    mappings = [
        [
            ("baseline", "majority_class", "en"),
            ("baseline", "majority_class", "es"),
            ("baseline", "majority_class", "ja"),
        ],
        [
            ("baseline", "tfidf_nb_en", "en"),
            ("baseline", "translation_tfidf_nb", "es"),
            ("baseline", "translation_tfidf_nb", "ja"),
        ],
        [
            ("structural_semantic", best_run, "en"),
            ("structural_semantic", best_run, "es"),
            ("structural_semantic", best_run, "ja"),
        ],
    ]

    for model_idx, config in enumerate(mappings):
        for lang_idx, (group_name, exp_name, lang) in enumerate(config):
            mean_val, std_val = _get_mean_std(
                summary_stats,
                group_name,
                exp_name,
                lang,
                "accuracy",
            )
            means[model_idx, lang_idx] = mean_val
            stds[model_idx, lang_idx] = std_val

    fig, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models), dtype=np.float32)
    offsets = np.array([-0.16, 0.0, 0.16], dtype=np.float32)
    colors = {"en": "#1f77b4", "es": "#2ca02c", "ja": "#d62728"}

    for lang_idx, lang in enumerate(langs):
        axis.errorbar(
            x + offsets[lang_idx],
            means[:, lang_idx],
            yerr=stds[:, lang_idx],
            fmt="o",
            capsize=4,
            elinewidth=1.5,
            markersize=7,
            label=lang.upper(),
            color=colors[lang],
        )

    axis.set_xticks(x)
    axis.set_xticklabels(models)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Accuracy (Mean +/- Std)")
    axis.set_title("Main Result Comparison")
    axis.legend(loc="best")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        os.path.join(plots_dir, "main_result_comparison_accuracy.png"),
        dpi=300,
    )
    plt.close(fig)


def plot_layer_freezing_ablation(summary_stats, plots_dir):
    """Plot top-half versus bottom-half training ablation with error bars."""
    # Plot Mean +/- Std accuracy points for top-half vs bottom-half freezing.
    langs = ["en", "es", "ja"]
    top_means = []
    top_stds = []
    bottom_means = []
    bottom_stds = []
    for lang in langs:
        top_mean, top_std = _get_mean_std(
            summary_stats,
            "structural_semantic",
            "train_top_half",
            lang,
            "accuracy",
        )
        bottom_mean, bottom_std = _get_mean_std(
            summary_stats,
            "structural_semantic",
            "train_bottom_half",
            lang,
            "accuracy",
        )
        top_means.append(top_mean)
        top_stds.append(top_std)
        bottom_means.append(bottom_mean)
        bottom_stds.append(bottom_std)

    fig, axis = plt.subplots(figsize=(8, 5))
    x = np.arange(len(langs), dtype=np.float32)
    axis.errorbar(
        x - 0.08,
        bottom_means,
        yerr=bottom_stds,
        fmt="o",
        capsize=4,
        elinewidth=1.5,
        markersize=7,
        label="train bottom half",
        color="#ff7f0e",
    )
    axis.errorbar(
        x + 0.08,
        top_means,
        yerr=top_stds,
        fmt="o",
        capsize=4,
        elinewidth=1.5,
        markersize=7,
        label="train top half",
        color="#1f77b4",
    )

    axis.set_xticks(x)
    axis.set_xticklabels([lang.upper() for lang in langs])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Accuracy (Mean +/- Std)")
    axis.set_title("Layer Freezing Ablation")
    axis.legend(loc="best")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        os.path.join(plots_dir, "layer_freezing_ablation_accuracy.png"),
        dpi=300,
    )
    plt.close(fig)


def plot_training_validation_loss(all_error_rows, summary_long, seeds, plots_dir):
    """Plot step-wise training and validation loss for a representative run."""
    # Plot one train/validation loss curve by step for a representative seed run.
    if len(all_error_rows) == 0:
        return

    seed = seeds[0]
    best_run = _best_structural_run_for_seed(summary_long, seed)
    error_df = pd.DataFrame(all_error_rows)
    run_df = error_df[
        (error_df["seed"] == seed)
        & (error_df["run_name"] == best_run)
    ].copy()
    if len(run_df) == 0:
        return
    run_df = run_df.sort_values(["step", "epoch"]).reset_index(drop=True)

    # If this run reused checkpoints, step logs may be missing (-1 only).
    if int(run_df["step"].max()) < 0:
        run_df["step"] = np.arange(1, len(run_df) + 1, dtype=np.int32)
    else:
        min_positive = int(run_df.loc[run_df["step"] >= 0, "step"].min())
        run_df.loc[run_df["step"] < 0, "step"] = min_positive

    train_df = run_df[~run_df["training_error"].isna()].copy()
    eval_df = run_df[~run_df["validation_error"].isna()].copy()
    train_df = train_df.sort_values("step")
    eval_df = eval_df.sort_values("step")
    eval_df = eval_df.drop_duplicates(subset=["step"], keep="last")

    fig, axis = plt.subplots(figsize=(8, 5))
    if len(train_df) > 0:
        axis.plot(
            train_df["step"].to_numpy(dtype=np.int32),
            train_df["training_error"].to_numpy(dtype=np.float32),
            label="Training Loss",
            color="#1f77b4",
            linewidth=2,
        )
    if len(eval_df) > 0:
        eval_steps = eval_df["step"].to_numpy(dtype=np.int32)
        eval_loss = eval_df["validation_error"].to_numpy(dtype=np.float32)
        if len(eval_df) <= 2:
            axis.scatter(
                eval_steps,
                eval_loss,
                label="Validation Loss",
                color="#d62728",
                s=32,
                zorder=3,
            )
        else:
            axis.plot(
                eval_steps,
                eval_loss,
                label="Validation Loss",
                color="#d62728",
                linewidth=2,
                marker="o",
                markersize=4,
            )

    axis.set_xlabel("Steps")
    axis.set_ylabel("Loss")
    axis.set_title("Training vs Validation Loss")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(
        os.path.join(plots_dir, "training_validation_loss_by_steps.png"),
        dpi=300,
    )
    plt.close(fig)
