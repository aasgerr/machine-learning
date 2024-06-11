import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from regression.linear import (
    GDLinRegress,
    NormLinRegress,
    QRLinRegress,
    SVDLinRegress,
    scipyLinRegress,
)

df = pd.read_csv("data.csv")
data = df[["Weight", "Volume", "CO2"]]

scaler = MinMaxScaler()
# transform data
scaled = scaler.fit_transform(data)
X = scaled[:, :-1]
y = scaled[:, -1]

for model in [
    GDLinRegress(),
    NormLinRegress(),
    QRLinRegress(),
    SVDLinRegress(),
    scipyLinRegress(),
]:
    model.fit(X, y)
    ypred = model.predict(np.array([[2300, 1300]]))
    print(f"{model} = {ypred}")

skmodel = LinearRegression()
skmodel.fit(X, y)
print(f"{skmodel} = {skmodel.predict(np.array([[2300, 1300]]))}")