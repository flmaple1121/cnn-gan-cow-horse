# Saved experiment artifacts

- `tables/task1_task2_comparison.csv`: authoritative final baseline-versus-manual-augmentation comparison.
- `tables/task1_seed_results.csv`: individual CNN seeds; the final ensemble averages logits rather than accuracies.
- `tables/dataset_split_summary.csv`: the 41-per-class, per-split input counts.
- `figures/`: saved high-resolution dataset grids, CNN loss/confusion plots, GAN epoch previews, the failed pretrained-discriminator experiment, and CycleGAN translations.
- `notebook_outputs/`: figures externalized from the original executed notebook for GitHub browsing. Some repeat plots in `figures/` at the notebook's original resolution.
- `valid_candidates/unselected/`: the additional candidate images retained in the saved project files.
- `tables/task2_valid_candidates.csv`: historical candidate scores, including the original Kaggle output paths. This historical table is not a manifest of every committed candidate image.
- `archive/task2_selected_synthetic_samples.csv`: an earlier automatic selection (four cows and four horses). It is **not** the final selection. The final 12-horse selection is recorded in `../selected/selection.csv`.

All numeric results are saved historical outputs. Publication did not rerun full training. There are no model checkpoint files in the supplied project. New notebook runs write to the ignored `report_outputs/` directory.
