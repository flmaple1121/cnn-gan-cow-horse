# Generated from notebooks/cnn-and-gan.ipynb by scripts/export_code.py.
# Running this script trains all three tasks; use the notebook for selective execution.

# %%
import copy
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
import torchvision.utils as vutils


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {0: "Cow", 1: "Horse"}
TRAIN_LABEL_MAP = {"Cow": 0, "Horse": 1}
TEST_LABEL_MAP = {"Shape1": 0, "Shape2": 1}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

CONFIG = {
    "seed": 17,
    "ensemble_seeds": [11, 21, 31],
    "image_size": 256,
    "gan_image_size": 128,
    "batch_size": 16,
    "gan_batch_size": 8,
    "num_workers": 0,
    "task1_epochs": 15,
    "task2_classifier_threshold": 0.58,
    "task2_realism_floor": 0.05,
    "task2_epochs": 1000,

    "task3_epochs": 50,
    "task1_lr": 1e-4,
    "task1_weight_decay": 1e-4,
    "task1_class_weights": [1.0, 5.0],
    "latent_dim": 100,
    "gan_lr": 2e-4,
    "gan_betas": (0.5, 0.999),
    "synthetic_sample_weight": 1.0,
    "task2_candidate_count": 80,
    "task2_checkpoint_keep": 3,
    "task2_manual_selected_root": None,
    "task2_sharpness_factor": 0.80,

    "task2_preview_every": 100,
    "task2_eval_every": 50,
    "cycle_lr": 2e-4,
    "cycle_betas": (0.5, 0.999),
    "cycle_lambda_cycle": 10.0,
    "cycle_lambda_identity": 5.0,
}

PROJECT_ROOT = next((base for base in [Path.cwd(), *Path.cwd().parents] if (base / "Newdata").is_dir()), Path.cwd())
OUTPUT_ROOT = (Path("/kaggle/working") if Path("/kaggle/working").exists() else PROJECT_ROOT) / "report_outputs"
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
VALID_DIR = OUTPUT_ROOT / "valid_candidates"
SELECTED_DIR = OUTPUT_ROOT / "selected_candidates"
for artifact_dir in (FIGURE_DIR, TABLE_DIR, VALID_DIR, SELECTED_DIR):
    artifact_dir.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    """Build filesystem-safe names for exported report artifacts."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def save_figure(name: str, fig=None, dpi: int = 300):
    """Save a matplotlib figure so it can be reused directly in the report."""
    fig = fig or plt.gcf()
    path = FIGURE_DIR / f"{slugify(name)}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to {path}")
    return path


def save_dataframe_artifacts(dataframe: pd.DataFrame, name: str):
    """Save each important table both as CSV and as a PNG snapshot for the report."""
    stem = slugify(name)
    csv_path = TABLE_DIR / f"{stem}.csv"
    png_path = TABLE_DIR / f"{stem}.png"
    dataframe.to_csv(csv_path, index=False)

    if len(dataframe.columns) == 0:
        fig, axis = plt.subplots(figsize=(8, 2))
        axis.axis("off")
        axis.text(0.5, 0.5, "No rows available", ha="center", va="center", fontsize=12)
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved table CSV to {csv_path}")
        print(f"Saved table image to {png_path}")
        return {"csv": csv_path, "png": png_path}

    fig_width = max(8, 1.6 * len(dataframe.columns))
    fig_height = max(1.8, 0.5 * (len(dataframe) + 1))
    fig, axis = plt.subplots(figsize=(fig_width, fig_height))
    axis.axis("off")
    table = axis.table(
        cellText=dataframe.values,
        colLabels=dataframe.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved table CSV to {csv_path}")
    print(f"Saved table image to {png_path}")
    return {"csv": csv_path, "png": png_path}


def save_tensor_image(tensor: torch.Tensor, path: Path):
    """Save one normalized tensor image to the requested output path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = torch.clamp((tensor.detach().cpu() + 1.0) / 2.0, 0.0, 1.0)
    vutils.save_image(image, path)
    print(f"Saved generated image to {path}")
    return path


print(f"Report artifacts will be saved under: {OUTPUT_ROOT}")


def set_seed(seed: int):
    """Set all random seeds so repeated runs are easier to compare."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_dataset_root() -> Path:
    """Find the dataset locally first, then fall back to Kaggle input mounts."""
    local_root = PROJECT_ROOT / "Newdata"
    if local_root.exists():
        return local_root

    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        for candidate in kaggle_root.glob("**/Newdata"):
            if (candidate / "Train").exists() and (candidate / "Test").exists():
                return candidate

    raise FileNotFoundError("Could not find the dataset root. Expected a Newdata folder.")



def resolve_manual_selected_root(manual_root=None) -> Path | None:
    """Locate the folder that contains the manually accepted synthetic images."""

    def is_selected_dir(path: Path) -> bool:
        return path.exists() and ((path / "cow").exists() or (path / "horse").exists())

    if manual_root is not None:
        manual_root = Path(manual_root)
        for candidate in (manual_root, manual_root / "selected"):
            if is_selected_dir(candidate):
                return candidate
        raise FileNotFoundError(f"Could not find a selected folder under: {manual_root}")

    search_paths = [
        PROJECT_ROOT / "selected",
        Path("valid_candidates") / "selected",
        Path("image_selection_outputs") / "valid_candidates" / "selected",
        OUTPUT_ROOT / "valid_candidates" / "selected",
    ]

    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        search_paths.extend(kaggle_root.glob("**/valid_candidates/selected"))
        search_paths.extend(kaggle_root.glob("**/selected"))

    seen = set()
    for candidate in search_paths:
        candidate = Path(candidate)
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if is_selected_dir(candidate):
            return candidate

    return None


def parse_view_id(filename: str) -> str:
    """The last two dash-separated fields identify the shared viewpoint grid."""
    parts = Path(filename).stem.split("-")
    if len(parts) >= 3:
        return "-".join(parts[-2:])
    return Path(filename).stem


def load_records(dataset_root: Path):
    """Build a reusable record list for all tasks."""
    records = {"train": [], "test": []}
    for split_name, label_map in [("Train", TRAIN_LABEL_MAP), ("Test", TEST_LABEL_MAP)]:
        for folder_name, label in label_map.items():
            folder = dataset_root / split_name / folder_name
            assert folder.exists(), f"Missing folder: {folder}"
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                records[split_name.lower()].append(
                    {
                        "path": path,
                        "filename": path.name,
                        "label": label,
                        "class_name": CLASS_NAMES[label],
                        "folder": folder_name,
                        "split": split_name,
                        "view_id": parse_view_id(path.name),
                        "is_synthetic": False,
                    }
                )
    return records["train"], records["test"]


def records_to_dataframe(records):
    """Create a small inspection table."""
    return pd.DataFrame(
        {
            "split": [record["split"] for record in records],
            "folder": [record["folder"] for record in records],
            "class_name": [record["class_name"] for record in records],
            "label": [record["label"] for record in records],
            "filename": [record["filename"] for record in records],
            "view_id": [record["view_id"] for record in records],
            "is_synthetic": [record["is_synthetic"] for record in records],
        }
    )


def compute_sobel_tensor(images: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge magnitude because shape cues are very important here."""
    if images.ndim == 3:
        images = images.unsqueeze(0)
    grayscale = images.mean(dim=1, keepdim=True)
    kernel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=images.device,
    ).unsqueeze(0)
    kernel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=images.device,
    ).unsqueeze(0)
    grad_x = F.conv2d(grayscale, kernel_x, padding=1)
    grad_y = F.conv2d(grayscale, kernel_y, padding=1)
    magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
    return magnitude


def tensor_to_display_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised tensor to a displayable RGB image."""
    image = tensor.detach().cpu().clamp(-1, 1)
    image = (image + 1.0) / 2.0
    return image.permute(1, 2, 0).numpy()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalised tensor back to PIL so the real and fake images share one transform path."""
    image = tensor_to_display_image(tensor)
    image = (image * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(image)


set_seed(CONFIG["seed"])
DATA_ROOT = resolve_dataset_root()
train_records, test_records = load_records(DATA_ROOT)

assert TRAIN_LABEL_MAP["Cow"] == 0
assert TRAIN_LABEL_MAP["Horse"] == 1
assert TEST_LABEL_MAP["Shape1"] == 0
assert TEST_LABEL_MAP["Shape2"] == 1

print(f"Device: {DEVICE}")
print(f"Dataset root: {DATA_ROOT}")
print(f"Training images: {len(train_records)}")
print(f"Test images: {len(test_records)}")

# %%
all_records_df = records_to_dataframe(train_records + test_records)
display(all_records_df.head())

summary_df = (
    all_records_df.groupby(["split", "class_name"])
    .size()
    .reset_index(name="count")
    .sort_values(["split", "class_name"])
)
display(summary_df)
save_dataframe_artifacts(summary_df, "dataset_split_summary")

# %%


def show_sample_grid(records, title: str, max_items: int = 6, save_name: str | None = None):
    """Display a few images so the viewpoint structure is visible."""
    selected = records[:max_items]
    fig, axes = plt.subplots(1, len(selected), figsize=(3 * len(selected), 3))
    if len(selected) == 1:
        axes = [axes]
    for axis, record in zip(axes, selected):
        image = Image.open(record["path"]).convert("RGB")
        axis.imshow(image)
        axis.set_title(f"{record['class_name']}\n{record['view_id']}", fontsize=9)
        axis.axis("off")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_name is not None:
        save_figure(save_name, fig)
    plt.show()


show_sample_grid([record for record in train_records if record["label"] == 0], "Training cows", save_name="training_cows_sample_grid")
show_sample_grid([record for record in train_records if record["label"] == 1], "Training horses", save_name="training_horses_sample_grid")
show_sample_grid([record for record in test_records if record["label"] == 0], "Test cows (Shape1)", save_name="test_cows_sample_grid")
show_sample_grid([record for record in test_records if record["label"] == 1], "Test horses (Shape2)", save_name="test_horses_sample_grid")

# %%
def build_transforms(image_size: int, train: bool):
    """Keep augmentation light so the model sees small variations without drifting away from the real data."""
    transform_steps = [T.Resize((image_size, image_size))]
    # I keep augmentation light because the dataset is tiny and aggressive transforms could change the class evidence.
    if train:
        transform_steps.extend(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(8),
            ]
        )
    transform_steps.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return T.Compose(transform_steps)


class RecordDataset(Dataset):
    """Load real images from the record list with one shared transform pipeline."""

    def __init__(self, records, transform):
        self.records = list(records)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["path"]).convert("RGB")
        return self.transform(image), record["label"], record["filename"]


class MixedDataset(Dataset):
    """Mix real records and synthetic tensors while keeping the same augmentation interface."""

    def __init__(self, real_records, synthetic_items, transform):
        self.real_records = list(real_records)
        self.synthetic_items = list(synthetic_items)
        self.transform = transform

    def __len__(self):
        return len(self.real_records) + len(self.synthetic_items)

    def __getitem__(self, index):
        if index < len(self.real_records):
            record = self.real_records[index]
            image = Image.open(record["path"]).convert("RGB")
            return self.transform(image), record["label"], record["filename"]

        item = self.synthetic_items[index - len(self.real_records)]
        if "path" in item:
            image = Image.open(item["path"]).convert("RGB")
        else:
            image = tensor_to_pil(item["tensor"])
        return self.transform(image), item["label"], item["filename"]


def build_train_loader(records, synthetic_items=None):
    """Use weighted sampling so horse examples are never drowned out during training."""
    synthetic_items = synthetic_items or []
    dataset = MixedDataset(records, synthetic_items, build_transforms(CONFIG["image_size"], train=True))

    labels = [record["label"] for record in records] + [item["label"] for item in synthetic_items]
    is_synthetic = [False] * len(records) + [True] * len(synthetic_items)

    class_counts = [max(1, labels.count(0)), max(1, labels.count(1))]
    sample_weights = []
    # Inverse class frequencies balance class sampling; multiplier 1.0 gives synthetic and real images the same within-class weight.
    for label, synth_flag in zip(labels, is_synthetic):
        weight = 1.0 / class_counts[label]
        if synth_flag:
            weight *= CONFIG["synthetic_sample_weight"]
        sample_weights.append(weight)

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,
        num_workers=CONFIG["num_workers"],
    )
    return loader, dataset



def build_eval_loader(records):
    """The evaluation loader stays deterministic and augmentation-free."""
    dataset = RecordDataset(records, build_transforms(CONFIG["image_size"], train=False))
    return DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

# %%
def weights_init_kaiming(module):
    """Kaiming initialisation keeps the original small CNN trainable from scratch."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class SimpleClassifier(nn.Module):
    """A compact CNN is appropriate here because the dataset is tiny and overfitting is the main risk."""

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Three conv blocks are enough to capture coarse shape and texture cues
        # without making the network large relative to the dataset size.
        self.block1 = nn.Sequential(
        # The first block captures low-level edges and the global silhouette of the animal.
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
        # The second block combines those edges into more informative local texture and contour patterns.
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
        # The third block produces higher-level class cues before the global pooling stage.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global average pooling reduces the chance of memorising exact pixel locations.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # The classifier head stays small because the feature extractor already compresses the image heavily.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        return self.classifier(x)

# %%
def confusion_from_predictions(y_true, y_pred):
    """Build a 2x2 confusion matrix without depending on external libraries."""
    matrix = np.zeros((2, 2), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def metrics_from_predictions(y_true, y_pred):
    """Return overall, balanced, and class-wise accuracy."""
    matrix = confusion_from_predictions(y_true, y_pred)
    overall = float((matrix[0, 0] + matrix[1, 1]) / max(1, matrix.sum()))
    cow_acc = float(matrix[0, 0] / max(1, matrix[0].sum()))
    horse_acc = float(matrix[1, 1] / max(1, matrix[1].sum()))
    balanced = 0.5 * (cow_acc + horse_acc)
    prediction_counts = {"Cow": int((np.array(y_pred) == 0).sum()), "Horse": int((np.array(y_pred) == 1).sum())}
    return {
        "accuracy": overall,
        "balanced_accuracy": balanced,
        "classwise_accuracy": {"Cow": cow_acc, "Horse": horse_acc},
        "confusion_matrix": matrix,
        "prediction_counts": prediction_counts,
    }


def show_confusion_matrix(confusion_matrix, title: str, save_name: str | None = None):
    """Plot the confusion matrix because one-class collapse is the main failure mode."""
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(confusion_matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=["Cow", "Horse"])
    axis.set_yticks([0, 1], labels=["Cow", "Horse"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    for row in range(2):
        for col in range(2):
            axis.text(col, row, int(confusion_matrix[row, col]), ha="center", va="center", color="black")
    fig.colorbar(image)
    plt.tight_layout()
    if save_name is not None:
        save_figure(save_name, fig)
    plt.show()


def train_simple_cnn(train_records, test_records, seed: int, synthetic_items=None):
    """Train one original-style CNN end to end and evaluate it on the test set."""
    set_seed(seed)
    train_loader, train_dataset = build_train_loader(train_records, synthetic_items=synthetic_items)
    test_loader = build_eval_loader(test_records)

    model = SimpleClassifier().to(DEVICE)
    model.apply(weights_init_kaiming)

    class_weights = torch.tensor(CONFIG["task1_class_weights"], dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # I use a horse-sensitive loss because the horse class is harder in practice and tends to be under-predicted.
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["task1_lr"], weight_decay=CONFIG["task1_weight_decay"])

    history = {"train_loss": []}
    for _ in range(CONFIG["task1_epochs"]):
    # The training loop stays simple: no early stopping is used, and stability comes from fixed seeds plus ensembling.
        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        history["train_loss"].append(running_loss / max(1, len(train_loader)))

    metrics = evaluate_model(model, test_loader)
    metrics["seed"] = seed
    metrics["history"] = history
    metrics["model"] = model
    metrics["train_size"] = len(train_dataset)
    return metrics


def evaluate_model(model, loader):
    """Evaluate one model and keep raw logits for later ensemble averaging."""
    model.eval()
    y_true, y_pred, filenames = [], [], []
    logits_list = []
    with torch.no_grad():
        for images, labels, batch_filenames in loader:
            images = images.to(DEVICE)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            logits_list.append(logits.cpu())
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predictions.tolist())
            filenames.extend(list(batch_filenames))

    metrics = metrics_from_predictions(y_true, y_pred)
    metrics["logits"] = torch.cat(logits_list, dim=0)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["filenames"] = filenames
    return metrics


def evaluate_ensemble(models, records):
    """Average logits across several small CNNs to reduce lucky and unlucky single runs."""
    loader = build_eval_loader(records)
    logits_sum = None
    y_true, filenames = None, None
    for model in models:
        metrics = evaluate_model(model, loader)
        if logits_sum is None:
            logits_sum = metrics["logits"].clone()
            y_true = metrics["y_true"]
            filenames = metrics["filenames"]
        else:
            logits_sum += metrics["logits"]
    mean_logits = logits_sum / len(models)
    y_pred = torch.argmax(mean_logits, dim=1).numpy().tolist()
    metrics = metrics_from_predictions(y_true, y_pred)
    metrics["mean_logits"] = mean_logits
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["filenames"] = filenames
    return metrics

def score_candidate_with_task1_ensemble(fake_tensor, models):
    """Use the Task 1 ensemble as a semantic filter for GAN candidates."""
    image = tensor_to_pil(fake_tensor)
    transform = build_transforms(CONFIG["image_size"], train=False)
    x = transform(image).unsqueeze(0).to(DEVICE)

    prob_sum = None
    with torch.no_grad():
        for model in models:
            model.eval()
            probs = torch.softmax(model(x), dim=1)
            prob_sum = probs if prob_sum is None else prob_sum + probs

    prob_mean = (prob_sum / len(models)).squeeze(0).cpu().numpy()
    return prob_mean

# %%
seed_results = []
for seed in CONFIG["ensemble_seeds"]:
    seed_result = train_simple_cnn(train_records, test_records, seed=seed)
    seed_results.append(seed_result)
    print(
        f"Seed {seed}: accuracy={seed_result['accuracy']:.4f}, "
        f"cow={seed_result['classwise_accuracy']['Cow']:.4f}, "
        f"horse={seed_result['classwise_accuracy']['Horse']:.4f}"
    )

task1_models = [result["model"] for result in seed_results]
task1_metrics = evaluate_ensemble(task1_models, test_records)

task1_seed_table = pd.DataFrame(
    [
        {
            "seed": result["seed"],
            "accuracy": result["accuracy"],
            "balanced_accuracy": result["balanced_accuracy"],
            "cow_accuracy": result["classwise_accuracy"]["Cow"],
            "horse_accuracy": result["classwise_accuracy"]["Horse"],
            "train_size": result["train_size"],
        }
        for result in seed_results
    ]
)
display(task1_seed_table)
save_dataframe_artifacts(task1_seed_table, "task1_seed_results")

print(f"Task 1 ensemble accuracy: {task1_metrics['accuracy']:.4f}")
print(f"Task 1 ensemble balanced accuracy: {task1_metrics['balanced_accuracy']:.4f}")
print(f"Task 1 class-wise accuracy: {task1_metrics['classwise_accuracy']}")
print(f"Task 1 prediction counts: {task1_metrics['prediction_counts']}")
show_confusion_matrix(task1_metrics["confusion_matrix"], "Task 1 ensemble confusion matrix", save_name="task1_ensemble_confusion_matrix")

plt.figure(figsize=(8, 4))
for result in seed_results:
    epochs = np.arange(1, len(result["history"]["train_loss"]) + 1)
    plt.plot(epochs, result["history"]["train_loss"], marker="o", label=f"Seed {result['seed']}")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("Task 1 training loss across the three fixed seeds")
plt.legend()
plt.tight_layout()
save_figure("task1_training_loss_across_three_fixed_seeds")
plt.show()

print(
    "Task 1 interpretation: the ensemble is still a simple CNN, but averaging three fixed runs "
    "reduces the chance that one unlucky seed collapses to a nearly single-class decision rule."
)

# %%
class SingleClassTensorDataset(Dataset):
    """Convert one class of real images into normalised tensors for GAN training."""

    def __init__(self, records, image_size: int):
        self.records = list(records)
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["path"]).convert("RGB")
        tensor = self.transform(image)
        return tensor, record["label"], record["filename"]



class StableGenerator(nn.Module):
    """A pure-noise generator that upsamples gradually and adds one refinement block before output."""

    def __init__(self, latent_dim: int = 100, img_channels: int = 3):
        super().__init__()
        self.init_size = CONFIG["gan_image_size"] // 16
        self.l1 = nn.Sequential(
        # The latent vector is first projected into a compact spatial feature map so the network can grow the image gradually.
            nn.Linear(latent_dim, 256 * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True),
        )

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv_blocks = nn.Sequential(
        # Progressive upsampling lets the generator add spatial detail step by step instead of hallucinating the full image at once.
            conv_block(256, 128),
            conv_block(128, 64),
            conv_block(64, 32),
            conv_block(32, 16),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.size(0), 256, self.init_size, self.init_size)
        return self.conv_blocks(out)


class StableDiscriminator(nn.Module):
    """A small discriminator is enough because the generated images are also small and class-specific."""

    def __init__(self, img_channels: int = 3):
        super().__init__()

        def block(in_channels, out_channels, use_bn=True):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
        # The discriminator compresses the image with stride-2 convolutions so it can focus on increasingly abstract realism cues.
            block(img_channels, 16, use_bn=False),
            block(16, 32),
            block(32, 64),
            block(64, 128),
        )
        final_spatial = CONFIG["gan_image_size"] // 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * final_spatial * final_spatial, 1),
            nn.Sigmoid(),
        )

    def forward(self, images):
        return self.classifier(self.features(images))

# %%
def make_gan_loader(records):
    """Each class-specific GAN only sees one domain at a time."""
    dataset = SingleClassTensorDataset(records, image_size=CONFIG["gan_image_size"])
    return DataLoader(dataset, batch_size=CONFIG["gan_batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])


def compute_real_sharpness_median(records):
    """Use the real class sharpness as a simple quality baseline for filtering."""
    dataset = SingleClassTensorDataset(records, image_size=CONFIG["gan_image_size"])
    values = []
    for tensor, _, _ in DataLoader(dataset, batch_size=1, shuffle=False):
        values.append(float(compute_sobel_tensor(tensor).mean().item()))
    return float(np.median(values))


def score_noise_candidate(fake_image, discriminator, class_label: int, task1_models, real_sharpness_median: float):
    """Combine realism, classifier confidence, and edge-energy alignment into one candidate score."""
    with torch.no_grad():
        realism = float(discriminator(fake_image.unsqueeze(0).to(DEVICE)).mean().item())

    sharpness = float(compute_sobel_tensor(fake_image.unsqueeze(0)).mean().item())
    class_probs = score_candidate_with_task1_ensemble(fake_image, task1_models)
    classifier_conf = float(class_probs[class_label])

    sharpness_alignment = max(
        0.0,
        1.0 - abs(sharpness - real_sharpness_median) / max(real_sharpness_median, 1e-6),
    )

    score = 0.70 * classifier_conf + 0.10 * realism + 0.20 * sharpness_alignment
    return {
        "realism": realism,
        "sharpness": sharpness,
        "classifier_conf": classifier_conf,
        "sharpness_alignment": sharpness_alignment,
        "score": score,
    }


def select_diverse_noise_candidates(candidates, keep_count: int, min_distance: float = 0.08):
    """Avoid selecting near-duplicate GAN samples when scoring a checkpoint."""
    if keep_count <= 0:
        return []

    selected = []
    for item in candidates:
        too_close = False
        for chosen in selected:
            dist = float(torch.mean(torch.abs(item["tensor"] - chosen["tensor"])).item())
            if dist < min_distance:
                too_close = True
                break
        if not too_close:
            selected.append(item)
        if len(selected) == keep_count:
            break

    return selected


def evaluate_noise_checkpoint(
    generator,
    discriminator,
    class_label: int,
    class_name: str,
    task1_models,
    real_sharpness_median: float,
    noise_bank,
):
    """Evaluate one checkpoint on a fixed noise bank so preview and scoring remain consistent."""
    candidates = []
    debug = {"sharpness_fail": 0, "classifier_fail": 0, "realism_fail": 0}
    sharpness_threshold = CONFIG["task2_sharpness_factor"] * real_sharpness_median

    with torch.no_grad():
        fake_images = generator(noise_bank).detach().cpu()

    for idx, fake_image in enumerate(fake_images):
        metrics = score_noise_candidate(
            fake_image,
            discriminator,
            class_label,
            task1_models,
            real_sharpness_median,
        )

        if metrics["sharpness"] < sharpness_threshold:
            debug["sharpness_fail"] += 1
            continue
        if metrics["realism"] < CONFIG["task2_realism_floor"]:
            debug["realism_fail"] += 1
            continue
        if metrics["classifier_conf"] < CONFIG["task2_classifier_threshold"]:
            debug["classifier_fail"] += 1
            continue

        candidates.append(
            {
                "label": class_label,
                "class_name": class_name,
                "filename": f"{class_name.lower()}_noise_{idx:03d}.png",
                "tensor": fake_image,
                **metrics,
            }
        )

    candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
    selected = select_diverse_noise_candidates(candidates, CONFIG["task2_checkpoint_keep"])
    mean_selected_score = float(np.mean([item["score"] for item in selected])) if selected else 0.0
    checkpoint_score = len(selected) + mean_selected_score

    return {
        "checkpoint_score": checkpoint_score,
        "valid_count": len(candidates),
        "selected_count": len(selected),
        "mean_selected_score": mean_selected_score,
        "all_candidates": candidates,
        "selected": selected,
        "debug": debug,
    }


def train_noise_gan(records, class_name: str, class_label: int, task1_models, seed: int):
    """Train a pure-noise GAN and keep the checkpoint that yields the best fixed-bank candidates."""
    set_seed(seed)
    loader = make_gan_loader(records)
    generator = StableGenerator(latent_dim=CONFIG["latent_dim"]).to(DEVICE)
    discriminator = StableDiscriminator().to(DEVICE)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=CONFIG["gan_lr"], betas=CONFIG["gan_betas"])
    optimizer_d = optim.Adam(discriminator.parameters(), lr=CONFIG["gan_lr"], betas=CONFIG["gan_betas"])

    history = defaultdict(list)
    preview_grids = []
    real_sharpness_median = compute_real_sharpness_median(records)

    # One fixed noise bank is reused for checkpoint comparison and for the exported valid candidate bank.
    selection_noise_bank = torch.randn(CONFIG["task2_candidate_count"], CONFIG["latent_dim"], device=DEVICE)
    preview_noise = selection_noise_bank[:16]

    best_score = -1e9
    best_epoch = 0
    best_generator_state = copy.deepcopy(generator.state_dict())
    best_discriminator_state = copy.deepcopy(discriminator.state_dict())
    best_all_candidates = []
    best_debug = None

    for epoch in range(1, CONFIG["task2_epochs"] + 1):
        running_d = 0.0
        running_g = 0.0
        batch_count = 0

        for real_images, _, _ in loader:
            real_images = real_images.to(DEVICE)
            batch_size = real_images.size(0)

            real_targets = torch.ones(batch_size, 1, device=DEVICE) * 0.9
            fake_targets = torch.zeros(batch_size, 1, device=DEVICE)

            optimizer_d.zero_grad(set_to_none=True)
            real_loss = criterion(discriminator(real_images), real_targets)

            noise = torch.randn(batch_size, CONFIG["latent_dim"], device=DEVICE)
            fake_images = generator(noise)
            fake_loss = criterion(discriminator(fake_images.detach()), fake_targets)
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            optimizer_d.step()

            g_loss_value = 0.0
            for _ in range(2):
                optimizer_g.zero_grad(set_to_none=True)
                fresh_noise = torch.randn(batch_size, CONFIG["latent_dim"], device=DEVICE)
                fresh_fake = generator(fresh_noise)
                g_loss = criterion(discriminator(fresh_fake), torch.ones(batch_size, 1, device=DEVICE))
                g_loss.backward()
                optimizer_g.step()
                g_loss_value = g_loss.item()

            running_d += d_loss.item()
            running_g += g_loss_value
            batch_count += 1

        history["d_loss"].append(running_d / max(1, batch_count))
        history["g_loss"].append(running_g / max(1, batch_count))

        if epoch % CONFIG["task2_preview_every"] == 0 or epoch == CONFIG["task2_epochs"]:
            with torch.no_grad():
                preview = generator(preview_noise).detach().cpu()
            preview_grids.append((epoch, vutils.make_grid(preview, nrow=4, normalize=True)))

        if epoch % CONFIG["task2_eval_every"] == 0 or epoch == CONFIG["task2_epochs"]:
            checkpoint_metrics = evaluate_noise_checkpoint(
                generator,
                discriminator,
                class_label,
                class_name,
                task1_models,
                real_sharpness_median,
                selection_noise_bank,
            )

            print(
                f"{class_name} epoch {epoch}: valid={checkpoint_metrics['valid_count']}, "
                f"selected={checkpoint_metrics['selected_count']}, "
                f"mean_selected_score={checkpoint_metrics['mean_selected_score']:.4f}, "
                f"checkpoint_score={checkpoint_metrics['checkpoint_score']:.4f}, "
                f"debug={checkpoint_metrics['debug']}"
            )

            score_margin = 0.03
            if (
                checkpoint_metrics["checkpoint_score"] > best_score + score_margin
                or (
                    abs(checkpoint_metrics["checkpoint_score"] - best_score) <= score_margin
                    and epoch > best_epoch
                )
            ):
                best_score = checkpoint_metrics["checkpoint_score"]
                best_epoch = epoch
                best_generator_state = copy.deepcopy(generator.state_dict())
                best_discriminator_state = copy.deepcopy(discriminator.state_dict())
                best_all_candidates = copy.deepcopy(checkpoint_metrics["all_candidates"])
                best_debug = copy.deepcopy(checkpoint_metrics["debug"])

    generator.load_state_dict(best_generator_state)
    discriminator.load_state_dict(best_discriminator_state)

    print(f"{class_name} best GAN checkpoint: epoch {best_epoch}, score={best_score:.4f}")

    return {
        "generator": generator,
        "discriminator": discriminator,
        "history": dict(history),
        "preview_grids": preview_grids,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "all_candidates": best_all_candidates,
        "debug": best_debug,
    }


def save_valid_candidate_bank(class_name: str, candidates, best_epoch: int):
    """Export every valid candidate from the best checkpoint so it can be inspected manually later."""
    class_dir = VALID_DIR / class_name.lower()
    if class_dir.exists():
        shutil.rmtree(class_dir)
    class_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in candidates:
        saved_path = save_tensor_image(item["tensor"], class_dir / item["filename"])
        rows.append(
            {
                "class_name": item["class_name"],
                "label": item["label"],
                "filename": item["filename"],
                "saved_path": str(saved_path),
                "best_epoch": best_epoch,
                "realism": item["realism"],
                "sharpness": item["sharpness"],
                "classifier_conf": item["classifier_conf"],
                "sharpness_alignment": item["sharpness_alignment"],
                "score": item["score"],
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "class_name",
            "label",
            "filename",
            "saved_path",
            "best_epoch",
            "realism",
            "sharpness",
            "classifier_conf",
            "sharpness_alignment",
            "score",
        ],
    )


def load_manual_selected_items(selected_root: Path):
    """Copy the manually accepted images into the notebook outputs and return their metadata."""
    for class_dir in (SELECTED_DIR / "cow", SELECTED_DIR / "horse"):
        if class_dir.exists():
            shutil.rmtree(class_dir)

    rows = []
    for class_name in ["Cow", "Horse"]:
        label = 0 if class_name == "Cow" else 1
        source_dir = Path(selected_root) / class_name.lower()
        target_dir = SELECTED_DIR / class_name.lower()

        if not source_dir.exists():
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        for source_path in sorted(source_dir.iterdir()):
            if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            target_path = target_dir / source_path.name
            shutil.copy2(source_path, target_path)
            rows.append(
                {
                    "class_name": class_name,
                    "label": label,
                    "filename": source_path.name,
                    "source_path": str(source_path),
                    "selected_path": str(target_path),
                }
            )

    return pd.DataFrame(
        rows,
        columns=["class_name", "label", "filename", "source_path", "selected_path"],
    )


def compare_task2_results(task1_metrics, task2_metrics, selected_counts):
    """Summarise the Task 1 baseline and the Task 2 manually curated augmentation result."""
    rows = [
        {
            "experiment": "Task 1 simple CNN ensemble",
            "overall_accuracy": task1_metrics["accuracy"],
            "balanced_accuracy": task1_metrics["balanced_accuracy"],
            "cow_accuracy": task1_metrics["classwise_accuracy"]["Cow"],
            "horse_accuracy": task1_metrics["classwise_accuracy"]["Horse"],
            "selected_cows": 0,
            "selected_horses": 0,
        }
    ]
    if task2_metrics is not None:
        rows.append(
            {
                "experiment": "Task 2 manual image augmentation",
                "overall_accuracy": task2_metrics["accuracy"],
                "balanced_accuracy": task2_metrics["balanced_accuracy"],
                "cow_accuracy": task2_metrics["classwise_accuracy"]["Cow"],
                "horse_accuracy": task2_metrics["classwise_accuracy"]["Horse"],
                "selected_cows": selected_counts["cows"],
                "selected_horses": selected_counts["horses"],
            }
        )
    comparison = pd.DataFrame(rows)
    display(comparison)
    save_dataframe_artifacts(comparison, "task1_task2_comparison")
    return comparison


def plot_preview_history(preview_grids, title_prefix: str, save_prefix: str | None = None):
    """Show preview grids so the GAN quality can be inspected instead of assumed."""
    for epoch, grid in preview_grids:
        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.title(f"{title_prefix} - epoch {epoch}")
        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
        if save_prefix is not None:
            save_figure(f"{save_prefix}_epoch_{epoch}")
        plt.show()


def train_task1_ensemble(train_records, test_records, synthetic_items=None):
    """Reuse the exact Task 1 recipe so Task 2 remains a fair comparison."""
    results = []
    for seed in CONFIG["ensemble_seeds"]:
        results.append(train_simple_cnn(train_records, test_records, seed=seed, synthetic_items=synthetic_items))
    models = [result["model"] for result in results]
    metrics = evaluate_ensemble(models, test_records)
    return metrics, results

# %%
cow_train_records = [record for record in train_records if record["label"] == 0]
horse_train_records = [record for record in train_records if record["label"] == 1]

task2_cow = train_noise_gan(
    cow_train_records,
    class_name="Cow",
    class_label=0,
    task1_models=task1_models,
    seed=CONFIG["seed"] + 100,
)
task2_horse = train_noise_gan(
    horse_train_records,
    class_name="Horse",
    class_label=1,
    task1_models=task1_models,
    seed=CONFIG["seed"] + 200,
)

plot_preview_history(task2_cow["preview_grids"], "Task 2 cow generator", save_prefix="task2_cow_generator")
plot_preview_history(task2_horse["preview_grids"], "Task 2 horse generator", save_prefix="task2_horse_generator")

# I export every valid image from the best checkpoint so the final manual curation happens outside the GAN loop.
task2_valid_df = pd.concat(
    [
        save_valid_candidate_bank("Cow", task2_cow["all_candidates"], task2_cow["best_epoch"]),
        save_valid_candidate_bank("Horse", task2_horse["all_candidates"], task2_horse["best_epoch"]),
    ],
    ignore_index=True,
)

print(f"Task 2 valid cows: {int((task2_valid_df['class_name'] == 'Cow').sum())}")
print(f"Task 2 valid horses: {int((task2_valid_df['class_name'] == 'Horse').sum())}")
display(task2_valid_df.head(20))
save_dataframe_artifacts(task2_valid_df, "task2_valid_candidates")

# %%
selected_columns = ["class_name", "label", "filename", "source_path", "selected_path"]
selected_task2_table = pd.DataFrame(columns=selected_columns)
task2_metrics, task2_seed_results = None, None

# Final Task 2 retraining uses manually selected files rather than automatic top-score images.
selected_root = resolve_manual_selected_root(CONFIG["task2_manual_selected_root"])
if selected_root is None:
    print(
        "No selected folder was found. The valid candidate bank was exported under "
        f"{VALID_DIR}. After manual inspection, place the accepted files inside a "
        "`selected/cow` and/or `selected/horse` folder, then rerun this cell."
    )
    save_dataframe_artifacts(selected_task2_table, "task2_manual_selection")
else:
    print(f"Using manual Task 2 selection from: {selected_root}")
    selected_task2_table = load_manual_selected_items(selected_root)
    display(selected_task2_table)
    save_dataframe_artifacts(selected_task2_table, "task2_manual_selection")

    synthetic_items = [
        {
            "path": Path(row["selected_path"]),
            "label": int(row["label"]),
            "filename": Path(row["selected_path"]).name,
        }
        for _, row in selected_task2_table.iterrows()
    ]

    if len(synthetic_items) == 0:
        print("The selected folder is empty. Task 2 retraining was skipped.")
    else:
        task2_metrics, task2_seed_results = train_task1_ensemble(
            train_records,
            test_records,
            synthetic_items=synthetic_items,
        )

selected_counts = {
    "cows": int((selected_task2_table["class_name"] == "Cow").sum()) if "class_name" in selected_task2_table else 0,
    "horses": int((selected_task2_table["class_name"] == "Horse").sum()) if "class_name" in selected_task2_table else 0,
}

comparison_table = compare_task2_results(task1_metrics, task2_metrics, selected_counts)

# %%
class CycleResidualBlock(nn.Module):
    """Residual blocks help style transfer because they refine features without discarding the content structure."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
        # A residual block refines features while preserving the input content pathway, which is useful for style transfer.
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorCycleGAN(nn.Module):
    """A small ResNet-style generator is enough for qualitative unpaired style transfer."""

    def __init__(self, img_channels: int = 3, num_residual_blocks: int = 4):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(img_channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        in_features = 64
        out_features = 128
        for _ in range(2):
        # The encoder progressively reduces spatial size so the residual blocks operate on a richer and more compact representation.
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features *= 2

        for _ in range(num_residual_blocks):
            model += [CycleResidualBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features //= 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, img_channels, kernel_size=7),
            nn.Tanh(),
        ]
        # PatchGAN judges local patches rather than the whole image, which is appropriate for style and texture realism.
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class DiscriminatorCycleGAN(nn.Module):
    """PatchGAN looks at local patches, which is a natural fit for texture and style realism."""

    def __init__(self, img_channels: int = 3):
        super().__init__()

        def block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(img_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.model(x)

# %%
class UnpairedDataset(Dataset):
    """Each item pairs one cow image and one horse image without requiring exact correspondence."""

    def __init__(self, records_a, records_b, image_size: int):
        self.records_a = list(records_a)
        self.records_b = list(records_b)
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return max(len(self.records_a), len(self.records_b))

    def __getitem__(self, index):
        record_a = self.records_a[index % len(self.records_a)]
        record_b = self.records_b[index % len(self.records_b)]
        image_a = Image.open(record_a["path"]).convert("RGB")
        image_b = Image.open(record_b["path"]).convert("RGB")
        return {"A": self.transform(image_a), "B": self.transform(image_b)}


class ReplayBuffer:
    """A replay buffer slows discriminator oscillation by mixing fresh and slightly older fake images."""

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, batch):
        returned = []
        for item in batch.detach():
            item = item.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(item)
                returned.append(item)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    returned.append(self.data[idx].clone())
                    self.data[idx] = item
                else:
                    returned.append(item)
        return torch.cat(returned, dim=0)

def train_cyclegan(records_a, records_b, seed: int):
    """Train a basic CycleGAN using adversarial, cycle, and identity losses only."""
    set_seed(seed)
    dataset = UnpairedDataset(records_a, records_b, image_size=CONFIG["gan_image_size"])
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=CONFIG["num_workers"])

    generator_ab = GeneratorCycleGAN().to(DEVICE)
    generator_ba = GeneratorCycleGAN().to(DEVICE)
    discriminator_a = DiscriminatorCycleGAN().to(DEVICE)
    discriminator_b = DiscriminatorCycleGAN().to(DEVICE)

    optimizer_g = optim.Adam(
        list(generator_ab.parameters()) + list(generator_ba.parameters()),
        lr=CONFIG["cycle_lr"],
        betas=CONFIG["cycle_betas"],
    )
    optimizer_d_a = optim.Adam(discriminator_a.parameters(), lr=CONFIG["cycle_lr"], betas=CONFIG["cycle_betas"])
    optimizer_d_b = optim.Adam(discriminator_b.parameters(), lr=CONFIG["cycle_lr"], betas=CONFIG["cycle_betas"])

    criterion_gan = nn.MSELoss()
    # The training objective combines adversarial, cycle-consistency, and identity losses so style changes do not destroy content structure.
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    fake_a_buffer = ReplayBuffer(50)
    # The replay buffers slow down discriminator oscillation by mixing fresh fake images with slightly older ones.
    fake_b_buffer = ReplayBuffer(50)
    history = defaultdict(list)

    for _ in range(CONFIG["task3_epochs"]):
        running_g = 0.0
        running_d_a = 0.0
        running_d_b = 0.0
        batch_count = 0
        for batch in loader:
            real_a = batch["A"].to(DEVICE)
            real_b = batch["B"].to(DEVICE)

            optimizer_g.zero_grad(set_to_none=True)
            same_a = generator_ba(real_a)
            same_b = generator_ab(real_b)
            loss_identity = (
                criterion_identity(same_a, real_a) + criterion_identity(same_b, real_b)
            ) * CONFIG["cycle_lambda_identity"]

            fake_b = generator_ab(real_a)
            fake_a = generator_ba(real_b)
            pred_fake_b = discriminator_b(fake_b)
            pred_fake_a = discriminator_a(fake_a)
            loss_gan = criterion_gan(pred_fake_b, torch.ones_like(pred_fake_b)) + criterion_gan(
                pred_fake_a, torch.ones_like(pred_fake_a)
            )

            recovered_a = generator_ba(fake_b)
            recovered_b = generator_ab(fake_a)
            loss_cycle = (
                criterion_cycle(recovered_a, real_a) + criterion_cycle(recovered_b, real_b)
            ) * CONFIG["cycle_lambda_cycle"]

            loss_g = loss_identity + loss_gan + loss_cycle
            loss_g.backward()
            optimizer_g.step()

            optimizer_d_a.zero_grad(set_to_none=True)
            pred_real_a = discriminator_a(real_a)
            loss_real_a = criterion_gan(pred_real_a, torch.ones_like(pred_real_a))
            buffered_fake_a = fake_a_buffer.push_and_pop(fake_a)
            pred_fake_a = discriminator_a(buffered_fake_a.detach())
            loss_fake_a = criterion_gan(pred_fake_a, torch.zeros_like(pred_fake_a))
            loss_d_a = 0.5 * (loss_real_a + loss_fake_a)
            loss_d_a.backward()
            optimizer_d_a.step()

            optimizer_d_b.zero_grad(set_to_none=True)
            pred_real_b = discriminator_b(real_b)
            loss_real_b = criterion_gan(pred_real_b, torch.ones_like(pred_real_b))
            buffered_fake_b = fake_b_buffer.push_and_pop(fake_b)
            pred_fake_b = discriminator_b(buffered_fake_b.detach())
            loss_fake_b = criterion_gan(pred_fake_b, torch.zeros_like(pred_fake_b))
            loss_d_b = 0.5 * (loss_real_b + loss_fake_b)
            loss_d_b.backward()
            optimizer_d_b.step()

            running_g += loss_g.item()
            running_d_a += loss_d_a.item()
            running_d_b += loss_d_b.item()
            batch_count += 1

        history["g_loss"].append(running_g / max(1, batch_count))
        history["d_a_loss"].append(running_d_a / max(1, batch_count))
        history["d_b_loss"].append(running_d_b / max(1, batch_count))

    return {"generator_ab": generator_ab, "generator_ba": generator_ba, "history": dict(history)}

# %%
def visualize_translations(cycle_result, source_cows, source_horses, title: str, max_items: int = 4, save_name: str | None = None):
    """Show both translation directions on a few examples."""
    transform = T.Compose(
        [
            T.Resize((CONFIG["gan_image_size"], CONFIG["gan_image_size"])),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    generator_ab = cycle_result["generator_ab"].eval()
    generator_ba = cycle_result["generator_ba"].eval()

    cow_batch = torch.stack(
        [transform(Image.open(record["path"]).convert("RGB")) for record in source_cows[:max_items]],
        dim=0,
    ).to(DEVICE)
    horse_batch = torch.stack(
        [transform(Image.open(record["path"]).convert("RGB")) for record in source_horses[:max_items]],
        dim=0,
    ).to(DEVICE)

    with torch.no_grad():
        fake_horses = generator_ab(cow_batch).detach().cpu()
        fake_cows = generator_ba(horse_batch).detach().cpu()

    fig, axes = plt.subplots(4, max_items, figsize=(3 * max_items, 12))
    axes = np.array(axes).reshape(4, max_items)
    for idx in range(max_items):
        axes[0, idx].imshow(tensor_to_display_image(cow_batch[idx].cpu()))
        axes[0, idx].set_title("Cow source")
        axes[0, idx].axis("off")
        axes[1, idx].imshow(tensor_to_display_image(fake_horses[idx]))
        axes[1, idx].set_title("Cow to Horse")
        axes[1, idx].axis("off")
        axes[2, idx].imshow(tensor_to_display_image(horse_batch[idx].cpu()))
        axes[2, idx].set_title("Horse source")
        axes[2, idx].axis("off")
        axes[3, idx].imshow(tensor_to_display_image(fake_cows[idx]))
        axes[3, idx].set_title("Horse to Cow")
        axes[3, idx].axis("off")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_name is not None:
        save_figure(save_name, fig)
    plt.show()


cycle_train_cows = [record for record in train_records if record["label"] == 0]
cycle_train_horses = [record for record in train_records if record["label"] == 1]
cycle_test_cows = [record for record in test_records if record["label"] == 0]
cycle_test_horses = [record for record in test_records if record["label"] == 1]

cycle_result = train_cyclegan(cycle_train_cows, cycle_train_horses, seed=CONFIG["seed"] + 500)

plt.figure(figsize=(10, 4))
epochs = np.arange(1, len(cycle_result["history"]["g_loss"]) + 1)
plt.plot(epochs, cycle_result["history"]["g_loss"], label="Generator loss")
plt.plot(epochs, cycle_result["history"]["d_a_loss"], label="Discriminator A loss")
plt.plot(epochs, cycle_result["history"]["d_b_loss"], label="Discriminator B loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Task 3 CycleGAN training losses")
plt.legend()
plt.tight_layout()
save_figure("task3_cyclegan_training_losses")
plt.show()

visualize_translations(cycle_result, cycle_train_cows, cycle_train_horses, "Task 3 translations on training images", save_name="task3_translations_training_images")
visualize_translations(cycle_result, cycle_test_cows, cycle_test_horses, "Task 3 translations on unseen test images", save_name="task3_translations_unseen_test_images")
