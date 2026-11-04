# Publication validation

Checked on 2026-11-04. These are packaging and CPU smoke checks, not a repeat of the original GPU training experiment.

- Notebook schema and all code cells/exported source compile.
- Dataset counts, labels and SHA-256 manifests match all 164 original input images and 12 final synthetic horses.
- Repository path discovery works when the working directory is `notebooks/`.
- The mixed training dataset contains 94 images; inverse-class-frequency sampling balances 41 cows and 53 horses, with the same within-class weight for real and synthetic horses.
- CNN, pure-noise GAN, and residual CycleGAN/PatchGAN pass CPU forward/backward checks with finite gradients.
- Recorded final accuracies correspond to 54/82 and 55/82 correct predictions.
- All 332 checked files in the source project remain unchanged.

## Environment used for CPU checks

- Python: 3.12.1
- torch: 2.3.0
- torchvision: 0.18.0
- numpy: 1.26.4
- pandas: 3.0.1
- matplotlib: 3.8.4
- Pillow: 10.3.0
- nbformat: 5.10.4
- IPython: 9.6.0

This environment is recorded for the publication checks; it is not a recovered lockfile for the original Kaggle run.
