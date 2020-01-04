from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series
from pandas.core.generic import NDFrame
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

veri = pd.read_csv("tcmb-altin-fiyatlari-2007:06.csv")

# print(veri)
x = np.arange(124).reshape(124,1)

y = veri["AltinFiyat"]

y = y.values.reshape(124,1)
# y = y.reshape(152,1)

plt.scatter(x,y)

tahminlineer = LinearRegression()
# tahminlineer.fit(x, y)
# tahminlineer.predict(x)

xYeni = PolynomialFeatures(degree=8).fit_transform(x)

polinom_model = tahminlineer.fit(xYeni, y)

plt.plot(x, polinom_model.predict(xYeni), color="red")
plt.show()

