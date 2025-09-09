# Experiment 3: Tiny ImageNet Transfer Learning

**Objective:** Adapt a pre-trained 10-class CNN (`exp1_best_model.h5`) to classify Tiny ImageNet’s 200 classes. This experiment demonstrates practical skills in transfer learning and deep learning model adaptation for portfolio purposes.

## Dataset

* Tiny ImageNet (64x64 RGB images)
* 200 classes, 500 training images per class
* Validation images annotated for evaluation

## Method

1. Load pre-trained CNN and remove last three layers.
2. Add new dense layers: Dense(1024, ReLU) → Dropout(0.5) → Dense(200, Softmax)
3. Train with:

   * Adam optimizer, lr=0.001
   * Categorical crossentropy
   * Batch size 64, max 50 epochs
   * EarlyStopping (patience=10)
   * LearningRateScheduler
   * Data augmentation: rotation, shift, horizontal flip

## Results

* Training halted early due to stagnant validation loss
* Final validation accuracy: \~0.9%
* Trained model and logs are available under the **[Releases](../../releases)** tab

## Portfolio Notes

This experiment showcases the ability to:

* Perform transfer learning on large datasets
* Adapt pre-trained CNNs to new classification tasks
* Implement training optimizations and callbacks
* Document experimental workflows and results clearly for professional presentation.
