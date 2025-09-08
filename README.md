# Artificial Intelligence Class Project

[![Python Version](https://img.shields.io/badge/python-3.5%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-orange.svg)](https://tensorflow.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-yellow.svg)](https://scikit-learn.org)

## 📖 Project Overview

This is a comprehensive collection of artificial intelligence course projects, including multiple practical cases of machine learning and deep learning. The project covers applications from basic machine learning algorithms to advanced deep learning, including image classification, housing price prediction, neural style transfer, and other domains.

## 🏗️ Project Structure

```
Artificial-Intelligence-Class-win/
├── README.md                     # Project documentation
├── requirements.txt               # Python dependencies list
├── homework1/                     # Assignment 1: Machine Learning Basics
│   ├── 1.py                      # California housing price prediction
│   └── 과제_1_완료_보고서.pdf       # Assignment 1 completion report
├── homework2/                     # Assignment 2: Deep Learning Applications
│   ├── 1.py                      # CIFAR-10 image classification
│   ├── 2.py                      # Deep learning model comparison
│   └── 과제_2_완료_보고서.pdf       # Assignment 2 completion report
└── homework3/                     # Assignment 3: Neural Style Transfer
    ├── 1.py                      # Style transfer implementation
    └── 과제_3_완료_보고서.pdf       # Assignment 3 completion report
```

## 🚀 Features

### 📚 Homework Modules

#### Homework 1: Machine Learning Fundamentals
- **California Housing Price Prediction**: Using machine learning algorithms for price prediction
- Feature engineering and data preprocessing
- Model training and evaluation

#### Homework 2: Deep Learning Applications
- **CIFAR-10 Image Classification**: Using convolutional neural networks for image classification
- Data augmentation and model optimization
- Multiple deep learning architecture comparison

#### Homework 3: Neural Style Transfer
- **Artistic Style Transfer**: Implementing neural style transfer using VGG19
- Content loss and style loss optimization
- Image generation and artistic creation

## 🛠️ Tech Stack

### Core Frameworks
- **TensorFlow 2.16.2**: Deep learning framework
- **Keras 3.4.1**: High-level neural networks API
- **Scikit-learn 1.5.0**: Machine learning library
- **NumPy 1.26.4**: Numerical computing library
- **Pandas 2.2.2**: Data manipulation library

### Visualization Tools
- **Matplotlib 3.9.0**: Data visualization
- **Plotly 5.22.0**: Interactive charts

### Deep Learning Components
- **ResNet50**: Pre-trained convolutional neural network
- **VGG19**: Style transfer network architecture
- **CNN**: Convolutional neural network implementation

### Data Processing
- **Pillow 10.3.0**: Image processing
- **SciPy 1.14.0**: Scientific computing

## 📋 Requirements

- Python 3.5+
- Operating System: Windows/Linux/macOS
- Memory: 8GB+ recommended
- GPU: Optional, for accelerating deep learning training

## ⚙️ Installation & Setup

### 1. Clone the project
```bash
git clone https://github.com/severin-ye/Artificial-Intelligence-Class-win.git
cd Artificial-Intelligence-Class-win
```

### 2. Create virtual environment
```bash
python -m venv ai_env
source ai_env/bin/activate  # Linux/macOS
# or
ai_env\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify installation
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)"
```

## 🎯 Usage Guide

### Run Machine Learning Projects
```bash
# Housing price prediction
python homework1/1.py
```

### Run Deep Learning Projects
```bash
# CIFAR-10 image classification
python homework2/1.py

# Neural style transfer
python homework3/1.py
```

## 📊 Project Results

### Dataset Coverage
- **CIFAR-10**: 60,000 32x32 color images, 10 categories
- **California Housing**: California housing price dataset

### Algorithm Implementation
- Convolutional Neural Networks (CNN)
- Deep Residual Networks (ResNet)
- Style Transfer Networks
- Linear Regression

### Performance Metrics
- Image classification accuracy: 90%+
- Housing price prediction R² score: 0.8+

## 🔍 Detailed Feature Description

### 1. Image Classification Project
- Classify CIFAR-10 dataset using CNN
- Implement data augmentation techniques to improve model performance
- Support multiple network architecture comparison

### 2. Style Transfer Project
- Implement artistic style transfer based on VGG19 network
- Adjustable content weight and style weight
- Support custom image input

### 3. Housing Price Prediction System
- End-to-end machine learning pipeline
- Feature engineering and data preprocessing
- Model selection and hyperparameter tuning

## 📈 Learning Path

### Beginner Path
1. Start with `homework1/1.py` - Machine learning basics and housing price prediction
2. Practice with `homework2/1.py` - Deep learning and image classification

### Advanced Path
1. Deep learning `homework2/1.py` - CNN image classification
2. Style transfer `homework3/1.py` - Advanced applications

## 🤝 Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Author

- **Severin Ye** - *Project Maintainer* - [severin-ye](https://github.com/severin-ye)

