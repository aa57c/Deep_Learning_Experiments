# Deep Learning Portfolio

This repository contains two comprehensive deep learning projects demonstrating expertise in computer vision, object detection, and neural network architecture optimization.

## 🚀 Projects Overview

### 1. Object Detection Evaluation System
**File:** `experiment1_B_yolo.py`

A comprehensive evaluation framework for comparing YOLO model predictions against ground truth annotations using the Microsoft COCO dataset.

### 2. CNN Architecture Comparison Study
**File:** `experiment1.ipynb`

An extensive comparative analysis of 9 different CNN architectures on the CIFAR-10 dataset, exploring various optimization techniques and architectural improvements.

---

## 📊 Project 1: Object Detection Evaluation System

### Overview
This project implements a robust evaluation pipeline for object detection models, specifically focusing on YOLOv8 performance analysis against ground truth bounding boxes.

### Key Features
- **Dataset Processing**: Automated handling of Microsoft COCO dataset annotations
- **YOLO Integration**: Seamless integration with Ultralytics YOLOv8 for inference
- **IoU Calculation**: Custom implementation of Intersection over Union metrics
- **Visualization**: Automatic bounding box visualization with ground truth overlays
- **Performance Metrics**: Comprehensive evaluation results exported to CSV

### Technical Implementation

#### Core Components
- **Data Loading & Preprocessing**: Efficient CSV parsing and image loading using OpenCV
- **Ground Truth Extraction**: Automated bounding box extraction and class mapping
- **Model Inference**: YOLOv8 model integration with configurable confidence thresholds
- **Evaluation Metrics**: IoU calculation for predicted vs. ground truth comparisons

#### Code Highlights
```python
def calculate_iou(box1, box2):
    """Calculates Intersection over Union for bounding box evaluation"""
    # Efficient IoU implementation with edge case handling
```

### Results & Output
- **IoU Analysis**: Detailed comparison results saved to `iou_results.csv`
- **Visual Validation**: Bounding box visualizations for qualitative assessment
- **Performance Metrics**: Quantitative evaluation of model accuracy

---

## 🧠 Project 2: CNN Architecture Comparison Study

### Overview
A systematic study comparing 9 different CNN architectures on CIFAR-10, implementing various optimization strategies and analyzing their impact on model performance.

### Architecture Configurations Tested

| Model | Key Features | Learning Rate | Epochs | Innovations |
|-------|-------------|---------------|---------|-------------|
| **Config 1** | Baseline CNN | 0.0001 | 50 | Standard architecture |
| **Config 2** | Increased Depth | 0.0001 | 50 | 48/96 filters, 1024 dense |
| **Config 3** | Batch Normalization | 0.0001 | 50 | BN after each conv layer |
| **Config 4** | ELU Activation | 0.001 | 50 | ELU instead of ReLU |
| **Config 5** | Variable Dropout | 0.0001 | 50 | Progressive dropout rates |
| **Config 6** | Higher Learning Rate | 0.001 | 50 | Optimized learning rate |
| **Config 7** | Data Augmentation | 0.001 | 100 | ImageDataGenerator |
| **Config 8** | Aggressive Learning | 0.01 | 50 | High learning rate |
| **Config 9** | Full Optimization | 0.001 | 100 | BN + L2 + LR Scheduler |

### Advanced Techniques Implemented

#### 1. **Batch Normalization**
```python
Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001))
BatchNormalization()
```

#### 2. **Data Augmentation**
- Rotation (±10°)
- Width/Height shifts (±5%)
- Horizontal flipping
- Zoom variations (±10%)

#### 3. **Learning Rate Scheduling**
```python
def scheduler(epoch, lr):
    if epoch < 10:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.1))
```

#### 4. **Regularization Strategies**
- L2 weight regularization
- Progressive dropout rates
- Early stopping with patience

### Key Technical Features
- **Automated Training Pipeline**: Systematic training of all 9 configurations
- **Performance Tracking**: Comprehensive accuracy logging and CSV export
- **Confusion Matrix Generation**: Detailed classification analysis
- **Model Checkpointing**: Best model preservation for each configuration
- **Early Stopping**: Intelligent training termination to prevent overfitting

---

## 🛠️ Technical Stack

### Libraries & Frameworks
- **Deep Learning**: TensorFlow/Keras, Ultralytics YOLO
- **Computer Vision**: OpenCV, PIL
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn

### Development Environment
- **Platform**: Google Colab with GPU acceleration (A100)
- **Python Version**: 3.12.2
- **Hardware**: High-memory runtime for large model training

---

## 📈 Results & Insights

### Object Detection Performance
- Successfully implemented IoU-based evaluation system
- Achieved comprehensive performance analysis across multiple test images
- Generated exportable results for further analysis

### CNN Architecture Study
- Systematic comparison of 9 different architectural approaches
- Identification of optimal hyperparameter combinations
- Documentation of performance improvements through architectural innovations

---

## 🎯 Key Accomplishments

### Technical Excellence
- **Modular Code Design**: Clean, reusable functions with clear documentation
- **Comprehensive Evaluation**: Multiple metrics and visualization approaches
- **Production-Ready Code**: Error handling and robust data processing
- **Scalable Architecture**: Easily extensible for additional models or datasets

### Research Methodology
- **Systematic Approach**: Controlled experiments with single-variable changes
- **Quantitative Analysis**: Detailed performance metrics and comparisons
- **Visual Validation**: Comprehensive visualization for result interpretation
- **Reproducible Results**: Well-documented hyperparameters and configurations

---

## 💼 Professional Skills Demonstrated

- **Computer Vision**: Object detection, image classification, bounding box analysis
- **Deep Learning**: CNN architecture design, hyperparameter optimization
- **Model Evaluation**: IoU calculations, confusion matrices, performance metrics
- **Data Engineering**: Efficient data loading, preprocessing pipelines
- **Research Methodology**: Systematic experimentation and analysis
- **Code Quality**: Clean, documented, production-ready implementations

---

## 🔧 Setup & Usage

### Requirements Installation
```bash
pip install ultralytics tensorflow opencv-python pandas numpy matplotlib seaborn scikit-learn
```

### Running the Projects
1. **Object Detection Evaluation**: Execute the Python script with your COCO dataset path
2. **CNN Comparison Study**: Run the Jupyter notebook in a GPU-enabled environment

---

## 📝 Future Enhancements

- **Model Expansion**: Integration of additional YOLO versions and CNN architectures
- **Advanced Metrics**: Implementation of mAP, precision-recall curves
- **Automated Hyperparameter Tuning**: Bayesian optimization integration
- **Real-time Inference**: Deployment-ready model serving capabilities

---

## 📞 Contact

This portfolio demonstrates practical expertise in deep learning, computer vision, and systematic machine learning research. The projects showcase both theoretical understanding and practical implementation skills essential for modern AI development roles.

*Ready to contribute to cutting-edge AI projects and drive innovation through data-driven solutions.*
