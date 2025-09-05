# Titanic - Kaggle Challenge 🚢

This repository contains my solutions for the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) competition on Kaggle.

## 📂 Project Structure
```bash
dataset/ # Training and test data
train_LGBM.py # LightGBM model training script # score 0.79186
train_RF.py # Random Forest model training script # score 0.77751
pyproject.toml # Project dependencies
README.md # Project documentation
```

## ⚙️ Setup
Clone the repository and create a virtual environment:

```bash
git clone https://github.com/YOUR_USERNAME/titanic-kaggle.git
cd titanic-kaggle
```

Install dependencies:
```
pip install -r requirements.txt
```
## 🚀 Usage

Train and generate submission file with LightGBM:
```
python train_LGBM.py
```

Train and generate submission file with Random Forest:
```
python train_RF.py
```
Submission files will be saved as:
```
LightGBM_submission.csv

RF_submission.csv
```
## 📊 Results

LightGBM: 0.79186  (Kaggle public score)

Random Forest: 0.77751 

## 📝 Notes

Data files (train.csv, test.csv) are not included in this repo due to Kaggle’s policy.

You can download them from the Titanic dataset page
.

## 👤 Author

Kou-Wei Chen (陳國威)
Master’s student, Department of Artificial Intelligence, Chang Gung University