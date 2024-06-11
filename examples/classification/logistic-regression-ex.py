import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from classification.logistic import (
    GDLogRegress,
    scipyLogRegress,
)

df = pd.read_csv("data.csv")
data = df[["Study Hours", "Test Score", "Passed Exam"]]

scaler = MinMaxScaler()
# transform data
scaled = scaler.fit_transform(data)
X = scaled[:, :-1]
y = scaled[:, -1]

for model in [
    GDLogRegress(),
    scipyLogRegress(),
]:
    model.fit(X, y)
    ypred = model.predict(np.array([[5, 58]]))
    print(f"{model} = {ypred}")

skmodel = LogisticRegression()
skmodel.fit(X, y)
print(f"{skmodel} = {skmodel.predict(np.array([[5, 58]]))}")