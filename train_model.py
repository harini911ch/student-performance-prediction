import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("students.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset information:")
data.info()

print("\nBasic statistics:")
print(data.describe())


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================

# Study hours vs result
plt.figure(figsize=(8, 5))
plt.scatter(data["study_hours"], data["result"])
plt.xlabel("Study Hours")
plt.ylabel("Result (0 = Fail, 1 = Pass)")
plt.title("Study Hours vs Student Result")
plt.show()


# Attendance vs result
plt.figure(figsize=(8, 5))
plt.scatter(data["attendance"], data["result"])
plt.xlabel("Attendance (%)")
plt.ylabel("Result (0 = Fail, 1 = Pass)")
plt.title("Attendance vs Student Result")
plt.show()


# Pass vs Fail distribution
result_counts = data["result"].value_counts()

plt.figure(figsize=(7, 5))
plt.bar(
    ["Fail", "Pass"],
    result_counts.reindex([0, 1])
)
plt.xlabel("Result")
plt.ylabel("Number of Students")
plt.title("Pass vs Fail Distribution")
plt.show()


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop("result", axis=1)
y = data["result"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)


# ============================================================
# 5. CREATE AND TRAIN MODELS
# ============================================================

logistic_model = LogisticRegression()

decision_tree_model = DecisionTreeClassifier(
    random_state=42
)

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

logistic_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

print("\nMultiple models trained successfully.")


# ============================================================
# 6. MAKE PREDICTIONS
# ============================================================

logistic_pred = logistic_model.predict(X_test)
decision_tree_pred = decision_tree_model.predict(X_test)
random_forest_pred = random_forest_model.predict(X_test)

print("\nLogistic Regression Predictions:")
print(logistic_pred)

print("\nDecision Tree Predictions:")
print(decision_tree_pred)

print("\nRandom Forest Predictions:")
print(random_forest_pred)

print("\nActual Values:")
print(y_test.values)


# ============================================================
# 7. COMPARE MODEL PERFORMANCE
# ============================================================

models = {
    "Logistic Regression": logistic_pred,
    "Decision Tree": decision_tree_pred,
    "Random Forest": random_forest_pred
}

print("\nModel Performance:")

for model_name, predictions in models.items():

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\n" + model_name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


# ============================================================
# 8. CONFUSION MATRIX FOR LOGISTIC REGRESSION
# ============================================================

cm = confusion_matrix(y_test, logistic_pred)

print("\nLogistic Regression Confusion Matrix:")
print(cm)


# ============================================================
# 9. CROSS-VALIDATION
# ============================================================

print("\nCross-Validation Results:")

logistic_cv = cross_val_score(
    logistic_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

decision_tree_cv = cross_val_score(
    decision_tree_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

random_forest_cv = cross_val_score(
    random_forest_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("Logistic Regression CV Accuracy:", logistic_cv)
print("Logistic Regression Mean:", logistic_cv.mean())

print("Decision Tree CV Accuracy:", decision_tree_cv)
print("Decision Tree Mean:", decision_tree_cv.mean())

print("Random Forest CV Accuracy:", random_forest_cv)
print("Random Forest Mean:", random_forest_cv.mean())


# ============================================================
# 10. SCALED LOGISTIC REGRESSION
# ============================================================

scaled_logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression())
])

scaled_logistic_model.fit(X_train, y_train)

scaled_pred = scaled_logistic_model.predict(X_test)

print("\nScaled Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, scaled_pred))
print("Precision:", precision_score(y_test, scaled_pred))
print("Recall:", recall_score(y_test, scaled_pred))
print("F1 Score:", f1_score(y_test, scaled_pred))


# ============================================================
# 11. NEW STUDENT PREDICTION
# ============================================================

new_student = pd.DataFrame([{
    "study_hours": 5,
    "attendance": 82,
    "previous_score": 68,
    "assignment_score": 72,
    "sleep_hours": 7
}])

prediction = logistic_model.predict(new_student)

print("\nNew Student Prediction:")

if prediction[0] == 1:
    print("Predicted Result: Pass")
else:
    print("Predicted Result: Fail")