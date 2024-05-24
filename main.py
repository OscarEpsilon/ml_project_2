#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

#get file and clean
d = pd.read_csv("dummy_data.csv")
df = d[["time_spent", "age", "gender", "interests", "location", "demographics", "profession", "income", "indebt", "isHomeOwner", "Owns_Car"]]
df["gender"] = preprocessing.LabelEncoder().fit(df["gender"].unique()).transform(df["gender"])
df["interests"] = preprocessing.LabelEncoder().fit(df["gender"].unique()).transform(df["gender"])
df["location"] = preprocessing.LabelEncoder().fit(df["gender"].unique()).transform(df["gender"])
df["demographics"] = preprocessing.LabelEncoder().fit(df["gender"].unique()).transform(df["gender"])
df["profession"] = preprocessing.LabelEncoder().fit(df["gender"].unique()).transform(df["gender"])
df["indebt"] = preprocessing.LabelEncoder().fit(df["gender"].unique()).transform(df["gender"])
df["isHomeOwner"] = preprocessing.LabelEncoder().fit(df["gender"].unique()).transform(df["gender"])
df["Owns_Car"] = preprocessing.LabelEncoder().fit(df["gender"].unique()).transform(df["gender"])

X = df[["time_spent", "age", "gender", "interests", "location", "demographics", "profession", "income", "indebt", "isHomeOwner", "Owns_Car"]].values.astype(float) 
print(X[0:5])
