# Student Performance Prediction

A machine learning project that predicts whether a student is likely to **Pass or Fail** based on academic and lifestyle-related features.

## Project Overview

This project uses Python and Scikit-learn to build classification models for predicting student performance.

The model uses:

- Study hours
- Attendance
- Previous score
- Assignment score
- Sleep hours

The target variable is:

- `0` → Fail
- `1` → Pass

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Machine Learning Workflow

1. Load the dataset using Pandas
2. Explore the dataset using basic statistics
3. Perform exploratory data analysis (EDA)
4. Separate features and target
5. Split the data into training and testing sets
6. Train multiple classification models
7. Evaluate the models using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
8. Analyze the confusion matrix
9. Perform 5-fold cross-validation
10. Test a new student's data

## Models Used

### Logistic Regression

Used as the primary classification model.

### Decision Tree

Used to compare a tree-based classification approach.

### Random Forest

Used as an ensemble-based classification approach.

## Results

On the current dataset:

| Model | Test Accuracy | Mean CV Accuracy |
|---|---:|---:|
| Logistic Regression | 100% | 100% |
| Decision Tree | 83.3% | 96.7% |
| Random Forest | 83.3% | 96.7% |

Logistic Regression performed best on this dataset.

## Example Prediction

For a new student with:

- Study hours: 5
- Attendance: 82%
- Previous score: 68
- Assignment score: 72
- Sleep hours: 7

The model predicted:

**Pass**

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/harini911ch/student-performance-prediction.git
cd student-performance-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib
```

### 5. Run the project

```bash
python train_model.py
```

## Limitations

The current dataset is small and synthetic, so the results should not be considered representative of real-world student performance.

The 100% Logistic Regression accuracy does not mean the model will achieve 100% accuracy on unseen real-world data.

A larger and more representative dataset would be required for a production-level model.

## Future Improvements

- Use a larger real-world dataset
- Add more relevant student features
- Handle missing values and outliers
- Perform more extensive hyperparameter tuning
- Test the model on external data
- Deploy the trained model as an API
- Add a user interface for predictions