# 🌸 Iris Flower Classification

A simple machine learning project to classify Iris flowers using the **Decision Tree** algorithm from `scikit-learn`.

## 📊 Project Overview
The goal of this project is to build a machine learning model that predicts the type of an Iris flower based on 4 features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The flower types are:
- Setosa
- Versicolor
- Virginica

## 🛠️ Tools & Libraries
- Python
- scikit-learn
- matplotlib
- seaborn
- numpy
- pandas

## 🧠 Models Tested
Several models were tested for comparison:
- Decision Tree 🌳
- Random Forest 🌲
- Support Vector Machine (SVM) 🧮
- K-Nearest Neighbors (KNN) 👥

## ✅ Results
We used 5-fold **cross-validation** to compare models.  
The best performing model was: **`{{ Best Model Name }}`**  
Accuracy: **`{{ Accuracy % }}`**

## 📈 Visualization
Confusion matrices were plotted to better understand each model’s performance.

## 🚀 How to Run
```bash
pip install -r requirements.txt
python iris_classifier.py
