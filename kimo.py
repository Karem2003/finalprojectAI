import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
data=pd.read_csv("heart.csv")

print("First 5 rows:") 
print(data.head()) 
print("\nDataset Info:") 
print(data.info()) 
print("\nMissing values:") 
print(data.isnull().sum())



# Check if there are any object (string) columns
print("\nColumns with object dtype:")
print(data.select_dtypes(include=['object']).columns)

# Convert categorical columns to numeric if needed
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
#---------------------------
# Correlation heatmap
plt.figure(figsize=(10,8)) 
sns.heatmap(data.corr(), annot=True, cmap='coolwarm') 
plt.title("Feature Correlation") 
plt.show()

#---------------------------
# 4. Feature selection and preprocessing

X = data.drop('target', axis=1) 
y = data['target']

scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)
#---------------------------
# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


#---------------------------
# 6. Model training using Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


#---------------------------
# 7. Evaluation

y_pred = model.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
#---------------------------
# 8. Feature importance

feature_importances = pd.Series(model.feature_importances_, index=data.columns[:-1])
feature_importances.sort_values(ascending=False).plot(kind='bar')
plt.title("Feature Importance")
plt.show()