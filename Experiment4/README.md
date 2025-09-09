# Experiment 4: Transfer Learning from Tiny ImageNet to CIFAR-10

**Objective:** Leverage a pre-trained Tiny ImageNet model (`trained_tiny_imagenet_model.h5`) for CIFAR-10 classification, demonstrating transfer learning, model adaptation, and portfolio-ready deep learning skills.

## Dataset

* CIFAR-10 (32x32 RGB images)
* 10 classes, 50,000 training images, 10,000 test images
* Images normalized and one-hot encoded for model training

## Method

1. Load the pre-trained Tiny ImageNet model and remove its last three layers.
2. Adapt the network for 32x32 CIFAR-10 images.
3. Add new dense layers:

   * Dense(1024, ReLU) → Dropout(0.5) → Dense(10, Softmax)
4. Compile with Adam optimizer (lr=0.0001) and categorical crossentropy loss.
5. Train with:

   * Data augmentation: rotation, shift, zoom, horizontal flip
   * EarlyStopping (patience=10)
   * ModelCheckpoint to save best model (`exp4_model.keras`)
   * LearningRateScheduler
   * Up to 200 epochs (training stopped early at epoch 37)

## Results

* Final Test Accuracy: 67.08%
* Training halted early due to EarlyStopping
* Trained model and logs are available under the **[Releases](../../releases)** tab

## Portfolio Notes

This experiment highlights the ability to:

* Apply transfer learning across datasets of different sizes and resolutions
* Adapt large pre-trained CNNs to new tasks
* Implement robust training strategies including data augmentation and callbacks
* Evaluate and document model performance for professional portfolio presentation
