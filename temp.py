import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datasetsforecast.m3 import M3

Y_df, _, _ = M3.load(directory="./data/monthly", group="Monthly")
unique_ids = Y_df["unique_id"].unique()[:30]

print(Y_df.to_latex())

print(Y_df[Y_df["unique_id"] == 'M1'])

m1 = Y_df[Y_df["unique_id"] == 'M1']
print(len(m1.y[:-18]))
print(len(m1.y[-18:]))

print(len(m1.y))

fig, ax = plt.subplots()
ax.plot(m1.ds, m1.y)
plt.show()

print(Y_df[Y_df["unique_id"] == 'M10'])