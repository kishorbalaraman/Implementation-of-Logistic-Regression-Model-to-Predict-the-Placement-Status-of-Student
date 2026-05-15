# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and create/load the dataset for student placement details.

2. Split the dataset into training and testing data using input features and target output.

3. Train the Logistic Regression model and predict the placement status for test data.

4. Evaluate the model using Accuracy Score, Confusion Matrix, and Classification Report.


## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: KISHOR B
RegisterNumber: 212225230141
```


```

# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Step 2: Create Dataset Manually
data = {
    'CGPA': [6.5, 7.0, 8.2, 5.8, 7.5, 9.1, 6.8, 8.5, 5.5, 7.8],
    'IQ': [110, 120, 135, 100, 125, 140, 115, 130, 95, 128],
    'Placement': [0, 1, 1, 0, 1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Step 3: Select Features and Target
X = df[['CGPA', 'IQ']]
y = df['Placement']

# Step 4: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Create Logistic Regression Model
model = LogisticRegression()

# Step 6: Train Model
model.fit(X_train, y_train)

# Step 7: Predict Test Data
y_pred = model.predict(X_test)

# Step 8: Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy of Model:")
print(accuracy)

# Step 9: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Step 10: Classification Report
report = classification_report(y_test, y_pred)

print("\nClassification Report:")
print(report)

# Step 11: Predict New Student
cgpa = float(input("\nEnter CGPA: "))
iq = int(input("Enter IQ: "))

new_student = [[cgpa, iq]]

result = model.predict(new_student)

if result[0] == 1:
    print("Student is Placed")
else:
    print("Student is Not Placed")


```

## Output:

<img width="511" height="542" alt="image" src="https://github.com/user-attachments/assets/dadd06d5-0512-4dfc-b543-9d8da4a5a1c5" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

