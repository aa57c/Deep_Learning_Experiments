# Fine-Tuning SegFormer for Semantic Segmentation

This project demonstrates fine-tuning the **SegFormer** model (`SegFormerForSemanticSegmentation`) on a **custom semantic segmentation dataset**. The goal is to label each pixel in an image with one of 150 semantic classes.

---

## Dataset

- **Toy ADE20k subset**: 10 training + 10 validation images with corresponding segmentation maps.
- Used for **proof-of-concept and model validation**.
- Each pixel belongs to one of **150 semantic classes**.

---

## Approach

1. **Load Pre-trained SegFormer Encoder**
   - Initialized with **ImageNet-1k pre-trained weights**.
   - Decoder head initialized randomly.
   
2. **Prepare Custom Dataset**
   - Defined a PyTorch `Dataset` class to load images and annotations.
   - Applied label reduction (`reduce_labels=True`) to match SegFormer’s expected label range.

3. **Define Dataloaders**
   - Training batch size: 2
   - Validation batch size: 2

4. **Model Setup**
   - SegFormer variant: `nvidia/mit-b0`
   - Number of labels: 150
   - Set `id2label` and `label2id` mappings from ADE20k dataset.

5. **Training**
   - Optimizer: `AdamW` with learning rate `6e-5`
   - Device: GPU if available
   - Metric: `mean_iou` (from HuggingFace `evaluate`)
   - Overfitted the toy dataset to verify model correctness.

---

## Inference

- Model outputs per-pixel class logits.
- Logits upsampled to original image size.
- Segmentation map visualized by mapping each class to a color palette.
- Compared predicted segmentation to ground truth.

---

## Results

- Successfully trained the model to overfit the toy dataset.
- Generated **pixel-level predictions** for test images.
- Computed metrics:
  - **Mean IoU**: Evaluates overlap between predicted and ground truth segmentation.
  - **Per-category accuracy**: Evaluates accuracy for each semantic class.
- Visualized predictions show **good alignment with ground truth** despite tiny dataset.

---

## Key Takeaways

- Fine-tuning a semantic segmentation model on a small dataset is feasible for proof-of-concept.
- HuggingFace `transformers` + `datasets` libraries enable **fast prototyping**.
- SegFormer architecture is flexible and can adapt to custom datasets with minimal modifications.

---

## Files

- `Fine_tune_Segformer.ipynb` — Training script
- `ADE20k_toy_dataset/` — Sample dataset (this has to be downloaded)
- `README.md` — This file
