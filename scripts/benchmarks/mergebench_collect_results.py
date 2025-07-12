#!/usr/bin/env python
# coding: utf-8

# %%

import logging
import os
from collections import defaultdict
from typing import Any, Dict, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from fusion_bench.utils import load_from_json
from fusion_bench.utils.rich_utils import setup_colorlogging

setup_colorlogging()
log = logging.getLogger(__name__)


# %%


MATH_TASKS = [("gsm8k_cot", "exact_match,flexible-extract")]
MULTILINGUAL_TASKS = [
    ("m_mmlu_fr", "acc,none"),
    ("arc_fr", "acc_norm,none"),
    ("hellaswag_fr", "acc_norm,none"),
    ("m_mmlu_es", "acc,none"),
    ("arc_es", "acc_norm,none"),
    ("hellaswag_es", "acc_norm,none"),
    ("m_mmlu_de", "acc,none"),
    ("arc_de", "acc_norm,none"),
    ("hellaswag_de", "acc_norm,none"),
    ("m_mmlu_ru", "acc,none"),
    ("arc_ru", "acc_norm,none"),
    ("hellaswag_ru", "acc_norm,none"),
]
INSTRUCTION_FOLLOWING_TASKS = [("ifeval", "inst_level_loose_acc,none")]
CODING_TASKS = [
    ("mbpp_plus", "pass_at_1,none"),
    ("humaneval_plus", "pass@1,create_test"),
]
SAFETY_TASKS = [
    ("truthfulqa", ("truthfulqa_mc2", "acc,none")),
    ("toxigen", "acc_norm,none"),
    ("winogender", ("winogender_all", "acc,none")),
]

TASK_CATEGORIES = {
    "math": MATH_TASKS,
    "multilingual": MULTILINGUAL_TASKS,
    "instruction_following": INSTRUCTION_FOLLOWING_TASKS,
    "coding": CODING_TASKS,
    "safety": SAFETY_TASKS,
}


# %%


def load_results(model):
    results = defaultdict(list)

    for task_category, tasks in TASK_CATEGORIES.items():
        for task_name, metric_name in tasks:
            report_dir = f"results/{task_name}/{model.replace('/', '__')}"
            if os.path.exists(report_dir) and len(os.listdir(report_dir)) > 0:
                if len(os.listdir(report_dir)) > 1:
                    log.warning(
                        f"Multiple reports found for {model} on {task_name} in directory {report_dir}"
                    )

                report_file = os.path.join(report_dir, os.listdir(report_dir)[0])
                report = load_from_json(report_file)
                if isinstance(metric_name, str):
                    metric = report["results"][task_name][metric_name]
                else:
                    metric = report["results"]
                    for i in metric_name:
                        metric = metric[i]

                # append to results
                results["model"].append(model)
                results["task_category"].append(task_category)
                results["task_name"].append(task_name)
                results["metric_name"].append(
                    metric_name
                    if isinstance(metric_name, str)
                    else ":".join(metric_name)
                )
                results["score"].append(metric)
    return results


def load_results_as_df(models: list[str]):
    results = defaultdict(list)
    for model in models:
        result = load_results(model)
        for k, v in result.items():
            results[k].extend(v)

    results_df = pd.DataFrame(results)
    return results_df


# %%


def plot_heatmap(results_df: pd.DataFrame):
    # Sort task_names by task_category, then by task_name for stability
    sorted_tasks = (
        results_df[["task_name", "task_category"]]
        .drop_duplicates()
        .sort_values(["task_category", "task_name"])
    )
    sorted_task_names = sorted_tasks["task_name"].tolist()

    # Pivot the DataFrame to have models as columns, task_names as rows, and scores as values
    pivot_df = results_df.pivot(index="task_name", columns="model", values="score")
    pivot_df = pivot_df.reindex(sorted_task_names)

    # Get task_name to task_category mapping for coloring
    task_to_category = results_df.set_index("task_name")["task_category"].to_dict()
    unique_categories = results_df["task_category"].unique()
    category_colors = sns.color_palette("Set2", n_colors=len(unique_categories))
    category_color_map = dict(zip(unique_categories, category_colors))

    # Prepare yticklabels and their colors
    yticklabels = pivot_df.index.tolist()
    ytick_colors = [category_color_map[task_to_category[task]] for task in yticklabels]

    # Prepare the figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(yticklabels) * 0.4)))

    # Use 'YlGn' colormap for each row
    cmap = plt.get_cmap("YlGn")
    for i, task in enumerate(pivot_df.index):
        row = pivot_df.loc[task].values.astype(float)
        # Mask NaNs for correct min/max
        valid = ~np.isnan(row)
        if valid.sum() == 0:
            continue
        row_min = np.nanmin(row)
        row_max = np.nanmax(row)
        norm = (
            Normalize(vmin=row_min, vmax=row_max)
            if row_max > row_min
            else Normalize(vmin=row_min - 1e-6, vmax=row_max + 1e-6)
        )
        for j, val in enumerate(row):
            rect_color = cmap(norm(val)) if not np.isnan(val) else (1, 1, 1, 1)
            ax.add_patch(
                plt.Rectangle((j, i), 1, 1, color=rect_color, ec="gray", lw=0.5)
            )
            if not np.isnan(val):
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    color="white" if norm(val) > 0.5 else "black",
                    fontsize=9,
                )

    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot_df.columns)) + 0.5)
    ax.set_xticklabels(pivot_df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot_df.index)) + 0.5)
    ax.set_yticklabels(pivot_df.index)

    # Color ytick labels by task_category
    for ytick, color in zip(ax.get_yticklabels(), ytick_colors):
        ytick.set_color(color)

    # Set axis labels
    ax.set_xlabel("Model")
    ax.set_ylabel("Task Name")

    # Create a legend for task_category colors
    from matplotlib.patches import Patch

    handles = [
        Patch(color=category_color_map[cat], label=cat) for cat in unique_categories
    ]
    ax.legend(
        handles=handles,
        title="Task Category",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    # Set limits and invert y-axis to match seaborn heatmap style
    ax.set_xlim(0, len(pivot_df.columns))
    ax.set_ylim(0, len(pivot_df.index))
    ax.invert_yaxis()

    plt.tight_layout()


# %%
models = [
    "google/gemma-2-2b",
    "MergeBench/gemma-2-2b_instruction",
    "MergeBench/gemma-2-2b_math",
    "MergeBench/gemma-2-2b_coding",
    "MergeBench/gemma-2-2b_multilingual",
    "MergeBench/gemma-2-2b_safety",
]
results_df = load_results_as_df(models)
results_df.to_csv("collected_results/gemma-2-2b.csv", index=False)
plot_heatmap(results_df)
plt.savefig("collected_results/gemma-2-2b.pdf", bbox_inches="tight")
plt.show()

# %%
models = [
    "google/gemma-2-2b-it",
    "MergeBench/gemma-2-2b-it_instruction",
    "MergeBench/gemma-2-2b-it_math",
    "MergeBench/gemma-2-2b-it_coding",
    "MergeBench/gemma-2-2b-it_multilingual",
    "MergeBench/gemma-2-2b-it_safety",
]

results_df = load_results_as_df(models)
results_df.to_csv("collected_results/gemma-2-2b-it.csv", index=False)
plot_heatmap(results_df)
plt.savefig("collected_results/gemma-2-2b-it.pdf", bbox_inches="tight")
plt.show()

# %%
models = [
    "google/gemma-2-9b",
    "MergeBench/gemma-2-9b_instruction",
    "MergeBench/gemma-2-9b_math",
    "MergeBench/gemma-2-9b_coding",
    "MergeBench/gemma-2-9b_multilingual",
    "MergeBench/gemma-2-9b_safety",
]

results_df = load_results_as_df(models)
results_df.to_csv("collected_results/gemma-2-9b.csv", index=False)
plot_heatmap(results_df)
plt.savefig("collected_results/gemma-2-9b.pdf", bbox_inches="tight")
plt.show()

# %%
models = [
    "google/gemma-2-9b-it",
    "MergeBench/gemma-2-9b-it_instruction",
    "MergeBench/gemma-2-9b-it_math",
    "MergeBench/gemma-2-9b-it_coding",
    "MergeBench/gemma-2-9b-it_multilingual",
    "MergeBench/gemma-2-9b-it_safety",
]

results_df = load_results_as_df(models)
results_df.to_csv("collected_results/gemma-2-9b-it.csv", index=False)
plot_heatmap(results_df)
plt.savefig("collected_results/gemma-2-9b-it.pdf", bbox_inches="tight")
plt.show()

# %%
models = [
    "meta-llama/Llama-3.2-3B",
    "MergeBench/Llama-3.2-3B_instruction",
    "MergeBench/Llama-3.2-3B_math",
    "MergeBench/Llama-3.2-3B_coding",
    "MergeBench/Llama-3.2-3B_multilingual",
    "MergeBench/Llama-3.2-3B_safety",
]

results_df = load_results_as_df(models)
results_df.to_csv("collected_results/Llama-3.2-3B.csv", index=False)
plot_heatmap(results_df)
plt.savefig("collected_results/Llama-3.2-3B.pdf", bbox_inches="tight")
plt.show()

# %%
models = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "MergeBench/Llama-3.2-3B-Instruct_instruction",
    "MergeBench/Llama-3.2-3B-Instruct_math",
    "MergeBench/Llama-3.2-3B-Instruct_coding",
    "MergeBench/Llama-3.2-3B-Instruct_multilingual",
    "MergeBench/Llama-3.2-3B-Instruct_safety",
]

results_df = load_results_as_df(models)
results_df.to_csv("collected_results/Llama-3.2-3B-Instruct.csv", index=False)
plot_heatmap(results_df)
plt.savefig("collected_results/Llama-3.2-3B-Instruct.pdf", bbox_inches="tight")
plt.show()

# %%
models = [
    "meta-llama/Llama-3.1-8B",
    "MergeBench/Llama-3.1-8B_instruction",
    "MergeBench/Llama-3.1-8B_math",
    "MergeBench/Llama-3.1-8B_coding",
    "MergeBench/Llama-3.1-8B_multilingual",
    "MergeBench/Llama-3.1-8B_safety",
]

results_df = load_results_as_df(models)
results_df.to_csv("collected_results/Llama-3.1-8B.csv", index=False)
plot_heatmap(results_df)
plt.savefig("collected_results/Llama-3.1-8B.pdf", bbox_inches="tight")
plt.show()

# %%
models = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "MergeBench/Llama-3.1-8B-Instruct_instruction",
    "MergeBench/Llama-3.1-8B-Instruct_math",
    "MergeBench/Llama-3.1-8B-Instruct_coding",
    "MergeBench/Llama-3.1-8B-Instruct_multilingual",
    "MergeBench/Llama-3.1-8B-Instruct_safety",
]

results_df = load_results_as_df(models)
results_df.to_csv("collected_results/Llama-3.1-8B-Instruct.csv", index=False)
plot_heatmap(results_df)
plt.savefig("collected_results/Llama-3.1-8B-Instruct.pdf", bbox_inches="tight")
plt.show()

# %%
