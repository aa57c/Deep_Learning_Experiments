# Experiment 2 – Transfer Learning & Semantic Segmentation

This folder contains Jupyter notebooks for Experiment 2, covering:

- Part A: Transfer learning with a custom topology on CIFAR100 & Tiny-ImageNet.
- Part A (Analysis): Visualization of results for Custom, ResNet, and VGG models using datasets CIFAR10, CIFAR100, & Tiny-ImageNet
- Part B: Semantic segmentation using HuggingFace SegFormer on ADE20k toy dataset.


## 📂 Notebooks
### 1. experiment2_A_custom_topology.ipynb
- Loads a pretrained custom model.
- Prepares datasets (CIFAR100, Tiny-ImageNet).
- Applies data augmentation (ImageDataGenerator).
- Rebuilds model with adjusted output layers for new class counts.
- Trains with early stopping and learning rate scheduling.
- Saves training history (CSV) and plots.
  
### Pipeline:
```mermaid
flowchart TD
    A[Load Pretrained Model] --> B[Modify Final Layer for New Classes]
    B --> C[Prepare Dataset CIFAR100 / Tiny-ImageNet]
    C --> D[Data Augmentation]
    D --> E[Train with LR Scheduler + Early Stopping]
    E --> F[Save Training History & Results]
```
### 2. experiment2_A_csv_to_plot.ipynb

- Reads CSV logs from Part A & baselines (ResNet, VGG19).
- Generates side-by-side plots:
  - Training vs Validation Accuracy
  - Training vs Validation Loss
  - Saves plots into corresponding result directories.

### Pipeline:
```mermaid
flowchart TD
    A[Load Training Logs CSVs] --> B[Parse Histories for Custom, ResNet, and VGG]
    B --> C[Plot Accuracy and Loss Curves]
    C --> D[Save Plots to Results Folder]
```
### 3. experiment2_B.ipynb
- Downloads and extracts ADE20k toy dataset.
- Defines custom PyTorch Dataset for segmentation.
- Fine-tunes SegFormer (nvidia/mit-b0).
- Logs metrics: Loss, Mean IoU, Mean Accuracy.
- Produces overlays of predictions with segmentation masks.
- Generates per-class metrics and heatmaps for Accuracy & IoU.

### Pipeline:
```mermaid
flowchart TD
    A[Download ADE20k Toy Dataset] --> B[Define Dataset Class]
    B --> C[Load SegFormer Model]
    C --> D[Train on ADE20k with AdamW Optimizer]
    D --> E[Log Metrics: Loss, mIoU, Accuracy]
    E --> F[Save CSV Results]
    F --> G[Visualize Predictions with Masks]
    G --> H[Compute Per-Class Metrics + Heatmaps]
```
## 🚀 How to Run
### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- Install dependencies:
  ```bash
  pip install tensorflow keras torch torchvision transformers datasets evaluate imageio pandas matplotlib seaborn
  ```
### Running
1. Open JupyterLab/Notebook:
  ```bash
  jupyter notebook
  ```
2. Navigate to this folder (experiment2/).
3. Run each notebook cell sequentially.

# 📊 Outputs
## Custom Topology
<table>
  <tr>
    <th>Dataset</th>
    <th>Training Accuracy Plot</th>
  </tr>
  <tr>
    <td align="center"><b>CIFAR100</b></td>
    <td align="center">
      <img src="https://github.com/aa57c/Deep_Learning_Experiments/blob/master/Experiment2/Custom/CIFAR100/plot.png?raw=true" width="500"><br>
      <em>Figure 1: Training Accuracy Plot for Custom Topology with CIFAR100 Dataset</em>
    </td>
  </tr>
  <tr>
    <td align="center"><b>TinyImageNet</b></td>
    <td align="center">
      <img src="https://github.com/aa57c/Deep_Learning_Experiments/blob/master/Experiment2/Custom/ImageNet/plot.png?raw=true" width="500"><br>
      <em>Figure 2: Training Accuracy Plot for Custom Topology with TinyImageNet Dataset</em>
    </td>
  </tr>
</table>

## ResNet Model
<table>
  <tr>
    <th>Dataset</th>
    <th>Training Accuracy Plot</th>
  </tr>
  <tr>
    <td align="center"><b>CIFAR10</b></td>
    <td align="center">
      <img src="https://github.com/aa57c/Deep_Learning_Experiments/blob/master/Experiment2/ResNet/CIFAR10/plot.png?raw=true" width="500"><br>
      <em>Figure 3: Training Accuracy Plot for ResNet Model with CIFAR10 Dataset</em>
    </td>
  </tr>
  <tr>
    <td align="center"><b>CIFAR100</b></td>
    <td align="center">
      <img src="https://github.com/aa57c/Deep_Learning_Experiments/blob/master/Experiment2/ResNet/CIFAR100/plot.png?raw=true" width="500"><br>
      <em>Figure 4: Training Accuracy Plot for ResNet Model with CIFAR100 Dataset</em>
    </td>
  </tr>
  <tr>
    <td align="center"><b>TinyImageNet</b></td>
    <td align="center">
      <img src="https://github.com/aa57c/Deep_Learning_Experiments/blob/master/Experiment2/ResNet/ImageNet/plot.png?raw=true" width="500"><br>
      <em>Figure 5: Training Accuracy Plot for ResNet Model with TinyImageNet Dataset</em>
    </td>
  </tr>
</table>

## VGG19 Model
<table>
  <tr>
    <th>Dataset</th>
    <th>Training Accuracy Plot</th>
  </tr>
  <tr>
    <td align="center"><b>CIFAR10</b></td>
    <td align="center">
      <img src="https://github.com/aa57c/Deep_Learning_Experiments/blob/master/Experiment2/VGG19/CIFAR10/plot.png?raw=true" width="500"><br>
      <em>Figure 6: Training Accuracy Plot for VGG19 Model Topology with CIFAR10 Dataset</em>
    </td>
  </tr>
  <tr>
    <td align="center"><b>CIFAR100</b></td>
    <td align="center">
      <img src="https://github.com/aa57c/Deep_Learning_Experiments/blob/master/Experiment2/VGG19/CIFAR100/plot.png?raw=true" width="500"><br>
      <em>Figure 7: Training Accuracy Plot for VGG19 Model Topology with CIFAR100 Dataset</em>
    </td>
  </tr>
  <tr>
    <td align="center"><b>TinyImageNet</b></td>
    <td align="center">
      <img src="https://github.com/aa57c/Deep_Learning_Experiments/blob/master/Experiment2/VGG19/ImageNet/plot.png?raw=true" width="500"><br>
      <em>Figure 8: Training Accuracy Plot for VGG19 Model Topology with TinyImageNet Dataset</em>
    </td>
  </tr>
</table>




### ⚡ Note: You don’t need to rerun all training to see results. Pre-generated CSV logs, plots, and visualizations are available in the Releases tab

## Demonstrates:

1. Transfer Learning for Image Classification
   - How a pretrained model (custom, ResNet, or VGG) can be fine-tuned on new datasets (CIFAR100, Tiny-ImageNet).
   - The effect of modifying final layers to match new class counts.
   - Comparison of training performance across different architectures.
   - Visualization of training/validation accuracy and loss to evaluate convergence and overfitting.
2. Semantic Segmentation with Pretrained Transformers
   - Fine-tuning a SegFormer model on ADE20k toy dataset.
   - Generating pixel-level predictions and overlays to visualize segmentation.
   - Computing per-class metrics, including Mean IoU and Accuracy.
   - Understanding how transfer learning accelerates convergence even on small datasets.
3. Key Takeaways:
   - Pretrained networks significantly reduce training time and improve performance when adapting to new tasks.
   - Visualization of metrics and overlays helps interpret model behavior and segmentation quality.
   - Provides a modular workflow that can be extended to larger datasets or other pretrained models.
