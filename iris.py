import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("Iris.csv")
df.drop("Id",axis=1)
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

X = np.array(df.iloc[:, 1:-1].values)
y = np.array(df.iloc[:, -1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model_load = pickle.load(open("Iris_data_train.m", "rb"))

sample = np.array([[4.9,3.0,1.4,0.2]])
result = model_load.predict(sample)
print(result)
