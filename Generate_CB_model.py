# Written by Rohit Shukla
# This script will generate the CatBoost based model for the network features.
# Please refer https://github.com/shuklarohit815/AlzGenPred for detailed tutorial.

import pandas as pd
from catboost import CatBoostClassifier
import pickle
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Open the file. If you have some other file name so please replace the file name with your file name.
data = pd.read_csv('selected_4_features_final.csv')

# The CatBoost model is proposed by using these four features.
x = data[["AverageShortestPathLength","ClosenessCentrality","NeighborhoodConnectivity","TopologicalCoefficient"]] # Add the index here. It will be good for future use.

# Save the value in the array
array = data.values
y =data.Label
print (x.shape, y.shape)

# Split the dataset in to train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1)
print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)

# These are hyperparameters which was obtained by using the GridSearchCV.
cb = CatBoostClassifier(depth = 8, iterations = 250, l2_leaf_reg = 3, learning_rate = 0.3)

# Train the model with the test set.
scores = cross_val_score(cb, x_train, y_train, cv=10, scoring='accuracy')
print(scores)
print("The training set accuracy is:", scores.mean())
cb.fit(x_train,y_train)

# Predict the value of test dataset
test_pred_class = cb.predict(x_test)
print("The test set accuracy is:", metrics.accuracy_score(y_test, test_pred_class))

#For saving the model
with open("catboost_model.pkl", "wb") as f:
  pickle.dump(cb,f)
