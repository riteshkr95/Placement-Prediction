import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
warnings.filterwarnings("ignore")

# load the place dataset

df = pd.read_csv("/Users/riteshkumar/Desktop/Notes_CampusX/datasets/placement.csv")

# Missing values
df.isnull().sum()
# Drop duplicates values
df.drop_duplicates(inplace=True)

# Variables X and y
X = df.drop(columns="placed")
y = df["placed"]

# train ,test and split

X_train ,X_test ,y_train ,y_test =  train_test_split(X ,y ,test_size=0.2 ,random_state=0)


# model
model = LogisticRegression()

model.fit(X_train ,y_train)

print(model.score(X_test ,y_test))


joblib.dump(model ,"log_reg.pkl")




