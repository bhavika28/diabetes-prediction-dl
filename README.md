# Diabetes Prediction using Deep Learning

## 🎯 Business Problem
Predicting diabetes onset using patient health metrics to enable early intervention and improve healthcare outcomes. This project compares TensorFlow and PyTorch implementations for optimal model performance.

## 📊 Dataset & Approach

- Data Source: Pima Indians Diabetes Database (768 samples, 8 features)
- Target: Binary classification (diabetic vs non-diabetic)
- Key Features: Glucose levels, BMI, insulin, age, pregnancies, blood pressure
- Split: 70% training, 30% testing
- Validation: Cross-validation and hyperparameter optimization

## 🔑 Key Results

- Best Accuracy: 85.2% using optimized neural network
- Framework Comparison: TensorFlow vs PyTorch performance analysis
- Model Optimization: Systematic architecture experimentation showing 12% improvement over baseline
- Clinical Relevance: High precision (0.82) minimizes false positives for medical applications

## 🛠️ Technologies Used
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

## 🚀 Quick Start
Prerequisites
bashpip install tensorflow torch pandas scikit-learn matplotlib seaborn
Run the Analysis
bash# Clone the repository
git clone https://github.com/bhavika28/diabetes-prediction-dl.git
cd diabetes-prediction-dl

## Run TensorFlow implementation
python tensorflow_model.py

## Run PyTorch implementation
python pytorch_model.py

## Compare results
python compare_models.py
📁 Project Structure
diabetes-prediction-dl/
├── data/
│   └── diabetes.csv           # Raw dataset
├── notebooks/
│   ├── data_exploration.ipynb # EDA and visualization
│   ├── tensorflow_model.ipynb # TF implementation
│   └── pytorch_model.ipynb    # PyTorch implementation
├── src/
│   ├── data_preprocessing.py  # Data cleaning utilities
│   ├── model_tensorflow.py    # TF model architecture
│   ├── model_pytorch.py       # PyTorch model architecture
│   └── evaluation.py          # Model comparison metrics
├── results/
│   ├── model_comparison.png   # Performance visualization
│   └── confusion_matrix.png   # Classification results
└── README.md

## 🧪 Model Experiments
### Baseline Model

Simple 3-layer neural network
ReLU activation, dropout regularization
Result: 73.1% accuracy

### Optimized Architecture

Hypothesis: Deeper network with batch normalization will improve generalization
5-layer network with BatchNorm and adaptive learning rate
Result: 85.2% accuracy (+12% improvement)

### Framework Comparison
MetricTensorFlowPyTorchWinnerAccuracy85.2%84.8%TensorFlowTraining Time45s52sTensorFlowMemory Usage2.1GB2.3GBTensorFlowEase of Use⭐⭐⭐⭐⭐⭐⭐⭐⭐PyTorch
📈 Key Insights

### Feature Importance: Glucose levels and BMI are strongest predictors
Model Architecture: Batch normalization significantly improved convergence
Clinical Application: High precision (82%) reduces false positive diagnoses
Framework Choice: TensorFlow slightly outperforms PyTorch for this tabular data task

## 🔬 Technical Highlights

- Data Preprocessing: Handled missing values, feature scaling, outlier detection
- Model Validation: K-fold cross-validation with stratified sampling
- Hyperparameter Tuning: Grid search with early stopping
- Interpretability: SHAP values for feature importance analysis

## 📊 Visualizations

- Correlation matrix and feature distributions
- Training/validation loss curves
- ROC curves and precision-recall analysis
- Confusion matrices for both frameworks

## 🎯 Business Impact
This model could be integrated into:

- Electronic Health Record (EHR) systems for screening
- Mobile health apps for at-risk population monitoring
- Clinical decision support tools for healthcare providers

## 📝 Citation
If you use this work, please cite:
Prasannakumar, B. (2024). Diabetes Prediction using Deep Learning: TensorFlow vs PyTorch Comparison. 
GitHub repository: https://github.com/bhavika28/diabetes-prediction-dl
