# Deep Learning Experiments

This folder contains two comprehensive deep learning projects demonstrating expertise in object detection evaluation and convolutional neural network optimization.

## 📁 Project Files

### 1. `experiment1_B_yolo.ipynb`
YOLO model evaluation system using Microsoft COCO dataset

### 2. `experiment1.ipynb` 
CNN architecture comparison study on CIFAR-10 dataset

---

## 🎯 Project 1: Object Detection Evaluation System

### Overview
A comprehensive evaluation framework comparing YOLOv8 predictions against ground truth annotations using IoU metrics.

### Key Features
- **Automated COCO Dataset Processing**: Seamless annotation parsing and image loading
- **YOLO Integration**: Direct YOLOv8 model inference with configurable thresholds
- **Custom IoU Implementation**: Efficient intersection over union calculations
- **Visual Validation**: Bounding box visualization with ground truth overlays
- **Performance Export**: Detailed CSV results for analysis

### Technical Implementation

#### Core Pipeline
```python
# Ground truth extraction
boxes = {}
images = {}
for class_id in classes:
    first_row = df[df['class'] == class_id].iloc[0]
    images[class_id] = cv2.imread(base_path + first_row['filename'])
    boxes[class_id] = [first_row['xmin'], first_row['xmax'], first_row['ymin'], first_row['ymax']]

# YOLO prediction
model = YOLO("yolov8m.pt")
results = model.predict(source=image_path, conf=0.2, iou=0.5)
```

#### IoU Calculation Engine
```python
def calculate_iou(box1, box2):
    """Optimized IoU calculation with edge case handling"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    # ... intersection and union calculations
```

### Results
- **Quantitative Metrics**: IoU scores exported to `iou_results.csv`
- **Visual Analysis**: Bounding box overlay comparisons
- **Class-wise Performance**: Individual object class evaluation

---

## 🧠 Project 2: CNN Architecture Comparison Study

### Overview
Systematic comparison of 9 CNN architectures on CIFAR-10, exploring optimization techniques and performance improvements.

### Architecture Configurations

| Model | Key Innovation | Learning Rate | Special Features |
|-------|----------------|---------------|------------------|
| **Config 1** | Baseline CNN | 0.0001 | Standard 32/64 filters |
| **Config 2** | Increased Depth | 0.0001 | 48/96 filters, 1024 dense |
| **Config 3** | Batch Normalization | 0.0001 | BN after conv layers |
| **Config 4** | ELU Activation | 0.001 | ELU instead of ReLU |
| **Config 5** | Variable Dropout | 0.0001 | Progressive dropout (0.2→0.45) |
| **Config 6** | Optimized LR | 0.001 | Higher learning rate |
| **Config 7** | Data Augmentation | 0.001 | ImageDataGenerator |
| **Config 8** | Aggressive Training | 0.01 | Very high learning rate |
| **Config 9** | Full Optimization | 0.001 | BN + L2 + LR Scheduler |

### Advanced Techniques Implemented

#### 1. **Batch Normalization Integration**
```python
model3 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    # ... additional layers
])
```

#### 2. **Data Augmentation Pipeline**
```python
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    zoom_range=0.1
)
```

#### 3. **Learning Rate Scheduling**
```python
def scheduler(epoch, lr):
    if epoch < 10:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.1))
```

#### 4. **Comprehensive Training Pipeline**
```python
# Automated training for all 9 configurations
for model in models:
    model_num = models.index(model) + 1
    # Dynamic learning rate assignment
    # Model compilation and training
    # Performance evaluation and export
```

### Key Technical Features
- **Automated Pipeline**: Systematic training of all configurations
- **Performance Tracking**: Accuracy logging and CSV export
- **Model Persistence**: Best model checkpointing
- **Early Stopping**: Intelligent overfitting prevention
- **Confusion Matrix Analysis**: Detailed classification breakdown

---

## 🛠️ Technical Stack

### Core Libraries
- **Deep Learning**: TensorFlow/Keras, Ultralytics YOLO
- **Computer Vision**: OpenCV, PIL  
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Evaluation**: Scikit-learn

### Environment
- **Platform**: Google Colab with GPU acceleration
- **Hardware**: A100 GPU with high-memory runtime
- **Python**: 3.12.2

---

## 📊 Results & Performance

### Object Detection Evaluation
- **IoU Analysis**: Comprehensive prediction vs. ground truth comparison
- **Multi-class Support**: Evaluation across different object categories
- **Visual Validation**: Qualitative assessment through bounding box visualization

### CNN Architecture Study  
- **Systematic Comparison**: Performance metrics across 9 different configurations
- **Optimization Insights**: Impact analysis of various techniques
- **Best Practices**: Identification of optimal hyperparameter combinations

## Sample Results (Models 1, 5, and 7)
### Note: the rest can be found in Releases under version 1.0.0
![Model 1's Confusion Matrix](https://github.com/aa57c/Deep_Learning_Experiments/blob/master/Experiment1/confusion_matrix_model1.png?raw=true)


---

## 💡 Key Technical Achievements

### 1. **Production-Ready Evaluation Pipeline**
- Robust error handling and data validation
- Scalable architecture for large datasets
- Comprehensive metric calculation and export

### 2. **Systematic Experimentation**
- Controlled variable testing across architectures
- Automated training and evaluation workflows  
- Reproducible results with documented configurations

### 3. **Advanced Optimization Techniques**
- Multiple regularization strategies
- Dynamic learning rate scheduling
- Comprehensive data augmentation

---

## 🚀 Usage Instructions

### Setup
```bash
# Install required packages
pip install ultralytics tensorflow opencv-python pandas numpy matplotlib seaborn scikit-learn

# For Jupyter notebook
pip install jupyter ipykernel
```

### Execution

#### Object Detection Evaluation
```bash
python object_detection_evaluation.py
```

#### CNN Experiment
```bash
# Launch Jupyter and run experiment1.ipynb
jupyter notebook experiment1.ipynb
```


---

## 🎯 Professional Skills Demonstrated

- **Computer Vision**: Object detection, image classification, bounding box analysis
- **Deep Learning Architecture**: CNN design and optimization strategies
- **Model Evaluation**: IoU calculations, confusion matrices, systematic performance analysis
- **Research Methodology**: Controlled experimentation and quantitative analysis
- **Data Engineering**: Efficient preprocessing and augmentation pipelines
- **Production Code**: Clean, documented, scalable implementations



---

## 🔮 Future Development

### Object Detection Enhancements
- **mAP Implementation**: Mean Average Precision metrics
- **Multi-model Comparison**: Support for different YOLO versions
- **Real-time Inference**: Performance optimization for deployment

### CNN Study Extensions  
- **Neural Architecture Search**: Automated architecture optimization
- **Transfer Learning**: Pre-trained model fine-tuning analysis
- **Advanced Regularization**: Cutout, mixup, and other modern techniques

---

*These projects demonstrate comprehensive expertise in modern deep learning practices, from research methodology to production-ready implementations, essential for roles in computer vision and machine learning engineering.*
