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
Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/titanic-kaggle.git
cd titanic-kaggle
```

Create and sync environment with uv:
```
uv sync
```
This will automatically create a virtual environment and install dependencies defined in pyproject.toml and locked in uv.lock.

Activate the environment:
```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows cmd
.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate
```
## 🚀 Usage

Train and generate submission file with LightGBM:
```
uv run python train_LGBM.py
```

Train and generate submission file with Random Forest:
```
uv run python train_RF.py
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