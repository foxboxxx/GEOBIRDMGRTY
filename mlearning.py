import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pygmt
import matplotlib.pyplot as plt

goldenPlover = pd.read_csv(r"plovercsv.csv")
goldenPlover = goldenPlover[['decimalLongitude', 'decimalLatitude', 'individualCount', 'year']]
goldenPlover['decimalLongitude'] = goldenPlover['decimalLongitude'].fillna(0)
goldenPlover['decimalLatitude'] = goldenPlover['decimalLatitude'].fillna(0)
goldenPlover['decimalLongitude'] = goldenPlover['decimalLongitude'].astype(int)
goldenPlover['decimalLatitude'] = goldenPlover['decimalLatitude'].astype(int)
goldenPlover['individualCount'] = goldenPlover['individualCount'].fillna(1)
goldenPlover['individualCount'] = goldenPlover['individualCount'].astype(int)
goldenPlover = goldenPlover.dropna(subset=['year'])
goldenPlover['year'] = goldenPlover['year'].astype(int)
goldenPlover = goldenPlover.query('5 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')

earth2000 = np.zeros(shape=(180, 360))
earth2020 = np.zeros(shape=(180, 360))

goldenPlover2000 = goldenPlover.query('2000 == year')
for x in np.arange(0, goldenPlover2000.shape[0]):
        earth2000[goldenPlover2000.iloc[x]['decimalLatitude'] + 90, goldenPlover2000.iloc[x]['decimalLongitude'] + 180,] = earth2000[goldenPlover2000.iloc[x]['decimalLatitude'] + 90, goldenPlover2000.iloc[x]['decimalLongitude'] + 180,] + goldenPlover2000.iloc[x]['individualCount']
        print(x, " -> ", earth2000[goldenPlover2000.iloc[x]['decimalLatitude'] + 90, goldenPlover2000.iloc[x]['decimalLongitude'] + 180,])

goldenPlover2020 = goldenPlover.query('2020 == year')
for x in np.arange(0, goldenPlover2020.shape[0]):
        earth2020[goldenPlover2020.iloc[x]['decimalLatitude'] + 90, goldenPlover2020.iloc[x]['decimalLongitude'] + 180,] = earth2020[goldenPlover2000.iloc[x]['decimalLatitude'] + 90, goldenPlover2020.iloc[x]['decimalLongitude'] + 180,] + goldenPlover2020.iloc[x]['individualCount']
        print(x, " -> ", earth2020[goldenPlover2020.iloc[x]['decimalLatitude'] + 90, goldenPlover2020.iloc[x]['decimalLongitude'] + 180,])

tensor2000 = torch.from_numpy(earth2000)
tensor2020 = torch.from_numpy(earth2020)

print(torch.lerp(tensor2000, tensor2020, 0.5))
npinterpolate = torch.lerp(tensor2000, tensor2020, 0.5).numpy()
np.savetxt("tensor1.csv", npinterpolate, delimiter=",")

# plt.imsave("tensor.png",tensor2020, )


