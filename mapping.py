import pandas as pd
import numpy as np
import pygmt
import igrf

# mag = igrf.igrf('2010-07-12', glat=65, glon=-148, alt_km=100)
pygmt.show_versions()


# igrfDataFull = pd.read_csv(r'igrfTestNoaaWeb.csv')

worldRegion = [-170,180,-60,80]

# mag = igrf.igrf('2010-07-12', glat=65, glon=-148, alt_km=100)
# spacing = "111km"

grid = pd.read_csv('gridIgrf1980.csv')
print(grid)
grid2 = pygmt.xyz2grd(data = grid, region = worldRegion, spacing = "222km")
# grd = pygmt.xyz2grd(data = igrfDataFull, x = igrfDataFull.long, y = igrfDataFull.lat, z = igrfDataFull.intensity)
fig = pygmt.Figure()
fig.grdimage(grid=grid2, projection = "R12c",)
print(grid)
fig.show()

fig = pygmt.Figure()
fig.grdcontour(grid=grid2)
fig.show()


# grd = pygmt.xyz2grd(data = igrfDataFull, region = worldRegion, spacing = spacing)


# fig = pygmt.Figure()
# fig.grdimage(
#     grid=grd, 
#     region = worldRegion,
#     cmap="batlow",
#     projection="R12c"
#     )
# fig.plot(
#     x=igrfDataFull.lat, y=igrfDataFull.long, style="c0.3c", color=igrfDataFull.intensity, pen="1p,black"
# )

# fig.savefig("gridOne.png")

# fig.show()


# igrfData = pd.read_csv(r'igrf.csv')
# grid = igrfData['DGRF.7']
# print(grid.iloc[1:])     
# fig = pygmt.Figure()
# fig.grdimage(grid=igrfData.iloc[1:, :])
# fig.show()