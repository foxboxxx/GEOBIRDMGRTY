import pandas as pd
import pygmt

df = pd.DataFrame({'Animal': [50, 50, 50, 51, 51], 'Max Speed': [42, 42, 43, 65, 65], 'INv': [100,100,100,100,100]})

print(df)

d2 = df.groupby(['Animal','Max Speed']).sum()

print(df)
print(d2)

print(d2.iloc[1])

fig = pygmt.Figure()
pygmt.makecpt(cmap="etopo1", reverse = True)
fig.grdimage("@earth_relief_10m", projection="H10c", frame=True)
fig.colorbar(frame=True)

fig.shift_origin(yshift="10c")

pygmt.makecpt(cmap="earth")
fig.grdimage("@earth_relief_10m", projection="H10c", frame=True)
fig.colorbar(frame=True)
fig.savefig("map.jpg")