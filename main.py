#imports
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from itertools import combinations #will told me about this

#create file
d = pd.read_csv("dummy_data.csv")
df = d[["time_spent", "age", "gender", "interests", "location", "demographics", "profession", "income", "indebt", "isHomeOwner", "Owns_Car"]]

#clean data
df["gender"] = preprocessing.LabelEncoder().fit(["male", "female", "non-binary"]).transform(df["gender"])
df["interests"] = preprocessing.LabelEncoder().fit(["Sports", "Travel", "Lifestlye"]).transform(df["interests"])
df["location"] = preprocessing.LabelEncoder().fit(["United States", "United Kingdom", "Australia"]).transform(df["location"])
df["demographics"] = preprocessing.LabelEncoder().fit(["Urban", "Sub_Urban", "Rural"]).transform(df["demographics"])
df["profession"] = preprocessing.LabelEncoder().fit(["Student", "Software Engineer", "Marketer Manager"]).transform(df["profession"]) #these throw harmless warnings.
df["indebt"] = preprocessing.LabelEncoder().fit(["False", "True"]).transform(df["indebt"])
df["isHomeOwner"] = preprocessing.LabelEncoder().fit(["False", "True"]).transform(df["isHomeOwner"])
df["Owns_Car"] = preprocessing.LabelEncoder().fit(["False", "True"]).transform(df["Owns_Car"])
X = df[["time_spent", "age", "gender", "interests", "location", "demographics", "profession", "income", "indebt", "isHomeOwner", "Owns_Car"]].values.astype(float) 

#make variable arrays
features = ["age", "gender", "interests", "demographics", "profession", "income", "indebt", "isHomeOwner", "Owns_Car"]
ind = df[features]
deps = df[["time_spent"]]

#normalize data
ind = pd.DataFrame(preprocessing.StandardScaler().fit(ind).transform(ind))
ind.columns = features

#inspect data
ind.head()

feature_combs = []
for length in range(1, len(features) + 1):
    feature_combs.extend(list(combinations(features, length)))

acc_list = []

for comb in feature_combs:
    comb_ind = ind[list(comb)]
    train_d, test_d, train_i, test_i = train_test_split(deps, comb_ind, test_size=0.2, random_state=4) #split data
    for k in range(20, 200): #this loop finds the the accuracy for 100<k<300
        m = KNeighborsClassifier(n_neighbors = k, n_jobs = -1).fit(train_i, np.ravel(train_d)) #fits a model for k = k
        yhat = m.predict(test_i)  #generates predictions for model m on test set
        acc = metrics.accuracy_score(test_d, yhat) #gets accuracy score on test set
        acc_list.append([acc, k])

max(acc_list)