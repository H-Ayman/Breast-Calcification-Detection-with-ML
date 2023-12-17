import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import joblib

# Read the CSV file into a pandas DataFrame
path = 'mamo1.csv'
df = pd.read_csv(path)

# Encode the 'class' column using LabelEncoder
encoder = LabelEncoder()
df['class'] = encoder.fit_transform(df['class'])

# Visualization of class distribution
x = df['class'].value_counts().to_list()
labels = ['Non-Microcalcification', 'Microcalcification']

# Select features (X) and target variable (y)
X = df[['area of object', 'average gray level', 'gradient strengh',
        'root mean square noise', 'contrast', 'low order']].values

y = df['class'].values

# Standardize features
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Initialize KNeighborsClassifier with k=1
k = 1
knn = KNeighborsClassifier(n_neighbors=k)

# Define hyperparameter grid for GridSearchCV
grid_params = {'n_neighbors': [1, 2, 3, 5, 7, 9, 11, 13, 21],
               'weights': ['uniform', 'distance'],
               'metric': ['minkowski', 'euclidean', 'manhattan']}

# Perform GridSearchCV to find the best hyperparameters
gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=1, cv=3, n_jobs=-1)
g_res = gs.fit(X_train, y_train)

# Output the best hyperparameters
print(g_res.best_params_)

# Train the model with the best hyperparameters obtained from GridSearchCV
neigh = KNeighborsClassifier(**g_res.best_params_).fit(X_train, y_train)

# Make predictions on the test set
yhat = neigh.predict(X_test)

# Evaluate the model accuracy
accuracy = np.mean(yhat == y_test)
print(f'The best accuracy was with {accuracy} with k={g_res.best_params_["n_neighbors"]}')

# Save the trained model to a file
joblib.dump(neigh, 'knn_model.joblib')