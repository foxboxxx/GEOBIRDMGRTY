import pandas as pd
import numpy as np
import pygmt
import pyIGRF
from pyIGRF import calculate


# mag = igrf.igrf('2010-07-12', glat=65, glon=-148, alt_km=100)
pygmt.show_versions()
print(pyIGRF.igrf_value(0, 0, 0, 2020.0)[6])
'''
The response is 7 float number about magnetic filed which is:

    D: declination (+ve east)
    I: inclination (+ve down)
    H: horizontal intensity
    X: north component
    Y: east component
    Z: vertical component (+ve down)
    F: total intensity
    unit: degree or nT
'''
# igrfDataFull = pd.read_csv(r'igrfTestNoaaWeb.csv')
worldRegion = [-180,180,-90,90]

# mag = igrf.igrf('2010-07-12', glat=65, glon=-148, alt_km=100)
# spacing = "111km"
# grid = pd.read_csv('gridIgrf1980.csv')

for i in np.arange(2020.5, 2025.5, 5.0):
    templateFrame = pd.DataFrame(columns = ["long", "lat", "intensity"])
    for x in np.arange(-180, 180):
        for y in np.arange(90, -90, -1):
            templateFrame = templateFrame.append({"long": x,"lat": y,"intensity": pyIGRF.igrf_value(y, x, 0, i)[6]}, ignore_index=True)

    print(templateFrame)
    # newDf = templateFrame(index = False)
    templateFrame.to_csv('template.csv', index = False)
    reading = pd.read_csv('template.csv')
    print(reading)
    frameGrid = pygmt.xyz2grd(data = reading, region = worldRegion, spacing = "222km")
    fig = pygmt.Figure()
    fig.basemap(region=worldRegion, projection="R12c", frame=True)
    fig.coast(land="burlywood", water="lightblue", shorelines = True)
    fig.grdimage(grid=frameGrid, transparency=25, projection = 'R12c')  
    fig.contour(
        pen="0.2p",

        x=reading['long'],
        y=reading['lat'],
        z=reading['intensity'],
        #contour interval
        levels=1000,
        #contour interval marking
        annotation=5000,
    )
    fig.contour(
        pen="0.2p,red",
        x=reading['long'],
        y=reading['lat'],
        z=reading['intensity'],
        #contour interval
        levels=5000,
    )
    fig.savefig("EMF_Field{}.png".format(i))


grid = pd.read_csv('gridIgrf1980.csv')
print(grid)
grid2 = pygmt.xyz2grd(data = grid, region = worldRegion, spacing = "222km")
# grd = pygmt.xyz2grd(data = igrfDataFull, x = igrfDataFull.long, y = igrfDataFull.lat, z = igrfDataFull.intensity)
fig = pygmt.Figure()

#just gradient for 1980
fig.grdimage(grid=grid2, projection = 'H15c')
print(grid)
fig.show()

#just contour lines for 1980
fig = pygmt.Figure()
fig.grdcontour(grid=grid2)
fig.show()

#full map put together with previous elements
fig = pygmt.Figure()
fig.basemap(region=worldRegion, projection="H15c", frame=True)
fig.coast(land="#666666", water="lightblue", shorelines = True)
fig.grdimage(grid=grid2, transparency=25, projection = 'H15c')
# initial contour attempt
# fig.grdcontour(grid=grid2, projection = 'H15c', interval = 2500)
fig.contour(
    pen="0.1p",
    # pass the data as 3 1d data columns
    x=grid['long'],
    y=grid['lat'],
    z=grid['intensity'],
    # set the contours z values intervals to 1000
    levels=1000,
    # set the contours annotation intervals to 5000
    annotation=5000,
)
fig.contour(
    pen="0.2p,red",
    # pass the data as 3 1d data columns
    x=grid['long'],
    y=grid['lat'],
    z=grid['intensity'],
    # set the contours z values intervals to 5000
    levels=5000,
)
fig.show()

# read in the xyz file into a 1D numpy array

fig = pygmt.Figure()
fig.basemap(region=[-180, 180, -90, 90], projection="H15c", frame=True)
fig.coast(land="#666666", water="skyblue")

fig.contour(
    pen="0.1p",
    # pass the data as 3 1d data columns
    x=grid['long'],
    y=grid['lat'],
    z=grid['intensity'],
    # set the contours z values intervals to 1000
    levels=1000,
    # set the contours annotation intervals to 5000
    annotation=5000,
)

#make index contours red
fig.contour(
    pen="0.2p,red",
    # pass the data as 3 1d data columns
    x=grid['long'],
    y=grid['lat'],
    z=grid['intensity'],
    # set the contours z values intervals to 5000
    levels=5000,
)


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