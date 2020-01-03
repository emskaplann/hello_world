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
x = veri["AltinFiyat"]
y = range(1,152)

plt.scatter(y,x)
# plt.show()

tahminlineer = LinearRegression()
tahminlineer.fit(y, x)
tahminlineer.predict(x)

plt.plot(x, tahminlineer.predict(x), color="red")

