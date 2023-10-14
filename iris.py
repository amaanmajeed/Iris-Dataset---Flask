import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the iris dataset
df = pd.read_csv('iris.data')
X = np.array(df.iloc[:, 0:4])
y = np.array(df.iloc[:, 4:])

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y.reshape(-1))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the SVM model
sv = SVC(kernel='linear').fit(X_train, y_train)

# Evaluate the performance of the model on the test set
accuracy = sv.score(X_test, y_test)
print("Accuracy:", accuracy)

pickle.dump(sv, open('iris.pkl', 'wb'))