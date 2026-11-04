# Presentation Q\&A Prep

## Dataset and Evaluation Protocol

### 1. Why is this dataset difficult even though it is balanced?
The dataset is balanced at the class level, but it is not rich at the instance level. In training, there is only one cow instance and one horse instance, each observed from multiple viewpoints. In testing, the viewpoints are similar, but the object instances are different. This means the real challenge is instance-level generalization, not class imbalance.

### 2. Why do you report both overall accuracy and class-wise accuracy?
Overall accuracy alone hides where the model fails. In this project, the main weakness is horse recognition, while cow recognition is consistently easier. If I only reported the global accuracy, I could miss the fact that most of the remaining errors come from one class. Class-wise accuracy makes the impact of the design choices much clearer.

## Task 1: CNN Design

### 3. Why is the CNN so small?
The dataset is extremely small, so overfitting is the dominant risk. A compact CNN with three convolutional blocks is large enough to learn edges, local patterns, and higher-level class cues, but still small enough to remain trainable from scratch. I also reduce parameters further with global average pooling instead of a large fully connected stack. In this setting, a smaller model is a better engineering choice than a deeper one.

### 4. Why use the class weight \([1,5]\)?
Horse is the weaker class in practice, even though the dataset is balanced numerically. Without extra emphasis, the CNN tends to learn cow features more easily and leaves horse performance behind. The \([1,5]\) loss weighting makes horse mistakes more expensive during optimization. This was one of the most important choices for making the baseline stable enough to be useful.

### 5. Why average logits instead of accuracies?
The final ensemble decision must be made at the image level, not at the metric level. Averaging accuracies would only combine three scalar summaries after all image-level information has already been lost. By averaging logits, I combine the seed-specific confidence values for each test image before making the final prediction. This is a proper ensemble rule, whereas averaging accuracies is not.

### 6. Why did you use weighted random sampling as well as class weighting?
These two mechanisms act at different stages. Weighted random sampling changes how often each class is seen during training batches, while class weighting changes how strongly mistakes are penalized in the loss. In this project, both help because the horse class is harder to learn, even though the dataset is balanced in count. Using both made the training less fragile across seeds.

## Task 2: GAN and Asymmetric Augmentation

### 7. Why train cow and horse GANs separately?
The two classes do not behave the same way during adversarial training. In practice, cow images became cleaner earlier, while horse images remained the difficult side. Training them separately lets me monitor checkpoint quality independently and export class-specific valid candidate banks. This is more controllable than forcing one GAN to model both classes together on such a small dataset.

### 8. Why is the latent vector dimension 100?
I chose a standard moderate latent size that is expressive enough for a small GAN but not unnecessarily large for this dataset. A much smaller latent space would restrict variation too much, while a much larger one would increase instability without clear benefit. Since the dataset is tiny, the bottleneck should remain simple. Dimension 100 is a reasonable compromise between expressiveness and trainability.

### 9. Why not rely only on automatic final GAN scoring?
The automatic scores were useful for building a valid candidate bank, but they were not sufficient for the final subset. In particular, global animal shape is easier to judge visually than with a single scalar based on realism, confidence, and sharpness. Some images scored well numerically while still looking structurally weak. Manual curation was therefore the more defensible final step.

### 10. Why is asymmetric augmentation justified?
Because the final error pattern is asymmetric. Cow accuracy was already strong and stable, while horse accuracy remained the bottleneck. Adding more synthetic cows tended to reinforce the easy class without helping the weak one. The successful result came from directly targeting the horse class, which is exactly where the classifier still needed support.

### 11. Why was `synthetic_sample_weight = 1.0` the correct final choice?
Earlier in the project, lower synthetic weights were safer because the automatically selected images were less reliable. In the final setup, I manually curated 12 horse images that were visually acceptable and specifically chosen to support the weak class. Once that manual filtering was done, reducing their sampling weight made them too weak to matter. Setting the weight to 1.0 ensured that the selected synthetic horses had enough influence during retraining.

### 12. Why did the pretrained discriminator fail?
On this dataset, the pretrained discriminator became too strong relative to the generator. The result was not better horse structure, but visually failed local artifacts and unstable blob-like outputs. This is a common problem when the discriminator can memorize tiny datasets too easily. In practice, the simpler from-scratch discriminator gave a more learnable adversarial signal.

## Task 3: Style Transfer

### 13. Why is Task 3 evaluated qualitatively?
Task 3 is not a classification task, so there is no direct scalar accuracy target that captures what good translation means. What matters is whether the translation changes domain appearance while preserving the main content structure, such as silhouette and viewpoint. That is why qualitative inspection is the right primary evaluation here. I still show training curves so the optimization behavior is visible, not hidden.

### 14. Why use CycleGAN-style residual generators and PatchGAN discriminators?
Residual generators are a natural choice when the goal is to change appearance without destroying structure. The residual bottleneck lets the network refine the representation instead of rebuilding the whole image from scratch. PatchGAN is also appropriate because local texture and style realism matter more than perfect global synthesis in this task. Together, these two choices are a good fit for small-data unpaired translation.

## Limitations and Future Work

### 15. What would you try next if you had more time?
I would first test a more systematic manual-selection protocol, for example by comparing several curated horse subsets instead of one final subset. Second, I would explore stronger but still from-scratch generators that preserve global structure better under the pure-noise constraint. Third, I would test whether a slightly larger CNN or a different regularization strategy can improve horse recognition without sacrificing stability. The next step would still be targeted and data-driven rather than simply making every model bigger.
