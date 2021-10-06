#%%

import pandas as pd
import numpy as np
import pygmt 

#from IPython.display import Image
#import matplotlib.pyplot as plt

#Test commands
#pygmt.show_versions()
#pygmt.test()

testdf = pd.DataFrame({ 'A': [2, 3, 4, 5, 6] , 'B': [4, 9, 16, 25, 36]})
print(testdf)
print(testdf.loc[1].A)

fig = pygmt.Figure()
fig.basemap(region=[-90, -70, 0, 20], projection="M8i", frame=True)
fig.coast(shorelines=True)
fig.show()
fig.savefig("central-america-shorelines.png")

data = pygmt.datasets.load_japan_quakes()

# Set the region for the plot to be slightly larger than the data bounds.
region = [
    data.longitude.min() - 1,
    data.longitude.max() + 1,
    data.latitude.min() - 1,
    data.latitude.max() + 1,
]

print(region)
print(data.head())
fig = pygmt.Figure()
fig.basemap(region=region, projection="M8i", frame=True)
fig.coast(land="black", water="skyblue")
fig.plot(
    x=data.longitude,
    y=data.latitude,
    sizes=0.2 * (2 ** data.magnitude),
    style="cc",
    color="white",
    pen="black",
)
fig.savefig("test.png")



print("test test test")
# %%
