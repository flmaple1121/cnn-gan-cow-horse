from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_ROOT = ROOT / "docs"
DATA_ROOT = ROOT / "results"
FIGURE_SRC = DATA_ROOT / "figures"
TABLE_SRC = DATA_ROOT / "tables"
SELECTED_ROOT = ROOT / "selected"

OUT_ROOT = TEMPLATE_ROOT / "generated"
OUT_FIGURES = OUT_ROOT / "figures"
OUT_TABLES = OUT_ROOT / "tables"
for path in (OUT_ROOT, OUT_FIGURES, OUT_TABLES):
    path.mkdir(parents=True, exist_ok=True)


def percent(value: float) -> str:
    return f"{value * 100:.2f}\\%"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def save_side_by_side(
    left_path: Path,
    right_path: Path,
    output_path: Path,
    left_title: str,
    right_title: str,
    figsize=(10, 4.2),
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for axis, image_path, title in zip(axes, [left_path, right_path], [left_title, right_title]):
        axis.imshow(load_image(image_path))
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_vertical_pair(
    top_path: Path,
    bottom_path: Path,
    output_path: Path,
    top_title: str,
    bottom_title: str,
    figsize=(5.1, 7.4),
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    for axis, image_path, title in zip(axes, [top_path, bottom_path], [top_title, bottom_title]):
        axis.imshow(load_image(image_path))
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_four_panel(
    image_paths: Iterable[Path],
    titles: Iterable[str],
    output_path: Path,
    figsize=(10, 8),
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    for axis, image_path, title in zip(axes, image_paths, titles):
        axis.imshow(load_image(image_path))
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_selected_montage(selected_paths: list[Path], output_path: Path) -> None:
    cols = 4
    rows = int(np.ceil(len(selected_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8.4, 6.2))
    axes = np.array(axes).reshape(rows, cols)
    for axis in axes.flatten():
        axis.axis("off")

    for axis, image_path in zip(axes.flatten(), selected_paths):
        axis.imshow(load_image(image_path))
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_dataset_table() -> None:
    df = pd.read_csv(TABLE_SRC / "dataset_split_summary.csv")
    pivot = (
        df.pivot(index="split", columns="class_name", values="count")
        .reindex(["Train", "Test"])
        .fillna(0)
        .astype(int)
    )
    text = rf"""
\begin{{tabular}}{{lcc}}
\toprule
Split & Cow & Horse \\
\midrule
Train & {pivot.loc["Train", "Cow"]} & {pivot.loc["Train", "Horse"]} \\
Test & {pivot.loc["Test", "Cow"]} & {pivot.loc["Test", "Horse"]} \\
\bottomrule
\end{{tabular}}
""".strip()
    write_text(OUT_TABLES / "dataset_summary.tex", text + "\n")


def build_seed_table() -> None:
    df = pd.read_csv(TABLE_SRC / "task1_seed_results.csv")
    comparison_df = pd.read_csv(TABLE_SRC / "task1_task2_comparison.csv")
    task1_row = comparison_df[comparison_df["experiment"] == "Task 1 simple CNN ensemble"].iloc[0]
    lines = [
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"Run & Overall & Cow & Horse \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{int(row['seed'])} & "
            f"{percent(row['accuracy'])} & "
            f"{percent(row['cow_accuracy'])} & "
            f"{percent(row['horse_accuracy'])} \\\\"
        )
    lines.append(r"\midrule")
    lines.append(
        "Ensemble & "
        f"{percent(task1_row['overall_accuracy'])} & "
        f"{percent(task1_row['cow_accuracy'])} & "
        f"{percent(task1_row['horse_accuracy'])} \\\\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(OUT_TABLES / "task1_seed_results.tex", "\n".join(lines) + "\n")


def build_comparison_table() -> None:
    df = pd.read_csv(TABLE_SRC / "task1_task2_comparison.csv")
    method_labels = {
        "Task 1 simple CNN ensemble": "Task 1 CNN",
        "Task 2 manual image augmentation": "Task 2 + 12 horses",
    }
    lines = [
        r"\begin{tabular}{p{2.1cm}ccccc}",
        r"\toprule",
        r"Method & Overall & Cow & Horse & Sel. Cow & Sel. Horse \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{method_labels.get(row['experiment'], row['experiment'])} & "
            f"{percent(row['overall_accuracy'])} & "
            f"{percent(row['cow_accuracy'])} & "
            f"{percent(row['horse_accuracy'])} & "
            f"{int(row['selected_cows'])} & "
            f"{int(row['selected_horses'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(OUT_TABLES / "task1_task2_comparison.tex", "\n".join(lines) + "\n")


def build_figures() -> None:
    save_four_panel(
        [
            FIGURE_SRC / "task2_cow_generator_epoch_900.png",
            FIGURE_SRC / "task2_cow_generator_epoch_1000.png",
            FIGURE_SRC / "task2_horse_generator_epoch_900.png",
            FIGURE_SRC / "task2_horse_generator_epoch_1000.png",
        ],
        [
            "Cow generator at epoch 900",
            "Cow generator at epoch 1000",
            "Horse generator at epoch 900",
            "Horse generator at epoch 1000",
        ],
        OUT_FIGURES / "task2_epoch_grid.png",
        figsize=(9.4, 7.0),
    )

    save_vertical_pair(
        FIGURE_SRC / "task1_training_loss_across_three_fixed_seeds.png",
        FIGURE_SRC / "task1_ensemble_confusion_matrix.png",
        OUT_FIGURES / "task1_overview.png",
        "Task 1 training loss",
        "Task 1 confusion matrix",
        figsize=(5.2, 6.9),
    )

    selected_horse_paths = sorted((SELECTED_ROOT / "horse").glob("*.png"))
    save_selected_montage(selected_horse_paths, OUT_FIGURES / "task2_selected_horses.png")

    save_side_by_side(
        FIGURE_SRC / "pretrained_discriminator_failure.png",
        FIGURE_SRC / "task2_horse_generator_epoch_1000.png",
        OUT_FIGURES / "task2_resnet_comparison.png",
        "Pretrained discriminator failure",
        "From-scratch horse generator at epoch 1000",
        figsize=(10, 4.8),
    )

    save_vertical_pair(
        OUT_FIGURES / "task2_selected_horses.png",
        OUT_FIGURES / "task2_resnet_comparison.png",
        OUT_FIGURES / "task2_manual_overview.png",
        "The 12 manually selected horse images",
        "Why the pretrained discriminator was rejected",
        figsize=(8.8, 7.8),
    )

    save_vertical_pair(
        FIGURE_SRC / "task3_translations_training_images.png",
        FIGURE_SRC / "task3_translations_unseen_test_images.png",
        OUT_FIGURES / "task3_overview.png",
        "Task 3 translations on training images",
        "Task 3 translations on unseen test images",
        figsize=(5.2, 7.4),
    )


def main() -> None:
    build_dataset_table()
    build_seed_table()
    build_comparison_table()
    build_figures()
    print(f"Generated report assets under: {OUT_ROOT}")


if __name__ == "__main__":
    main()
