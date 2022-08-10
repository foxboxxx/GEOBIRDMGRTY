import pandas as pd
import numpy as np
import pygmt
import pyIGRF
import seaborn as sb
from gifmaker import gifMaker

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

# <--- Trial code testing how my bird data looks when plotted via gridimage --->
'''
birdBench = np.genfromtxt('foo3.csv', delimiter=' ')
birdTestData = pd.DataFrame(columns = ["long", "lat", "individualCount"])
print(birdBench[179][359])
for x in np.arange(-180, 179):
    for y in np.arange(89, -90, -1):
        birdTestData = birdTestData.append({"long": x,"lat": y,"individualCount": birdBench[y + 90][x + 180]}, ignore_index = True)
print(birdTestData)
birdTestData.to_csv("alpha.csv", index = False)
birdTestData = pd.read_csv("alpha.csv")

gBird = pygmt.xyz2grd(data = birdTestData, region = worldRegion, spacing = "222km")
fig = pygmt.Figure()
fig.basemap(region = worldRegion, projection = "H15c", frame=True)
fig.coast(land = "burlywood", water = "lightblue", shorelines = "0.5p,black")
fig.grdimage(grid = gBird, transparency = 25, projection = 'H15c', cmap = "inferno")
fig.show()
fig.savefig("overall_heatmap_test_1.png")
#  <--- END --->
'''
def avianHeatmap(birdData, birdName, includeMagnetism, minYear, maxYear):
    mag = "NoField"
    birdData = pd.read_csv(birdData)
    listOfImages = []

    # Standardizing the data as some types are incorrect and some parts of the data have missing values . . .
    birdData = birdData[['decimalLongitude', 'decimalLatitude', 'individualCount', 'year']]
    birdData['decimalLongitude'] = birdData['decimalLongitude'].fillna(0)
    birdData['decimalLatitude'] = birdData['decimalLatitude'].fillna(0)
    birdData['decimalLongitude'] = birdData['decimalLongitude'].astype(int)
    birdData['decimalLatitude'] = birdData['decimalLatitude'].astype(int)
    birdData['individualCount'] = birdData['individualCount'].fillna(1)
    birdData['individualCount'] = birdData['individualCount'].astype(int)
    birdData = birdData.dropna(subset=['year'])
    birdData['year'] = birdData['year'].astype(int)

    # Filtering the data to only contain Pectoral Sandpipers in South America, with an individual count of less than 1000 for any given sighting
    birdData = birdData.query('5 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')
    earthFrame = np.zeros(shape=(180, 360))

    # Nested for loops to cycle through each year to create a heat map of each
    for i in np.arange(minYear, maxYear, 1):
        tempFrame = birdData.query('@i == year')
        for x in np.arange(0, tempFrame.shape[0]):
            earthFrame[tempFrame.iloc[x]['decimalLatitude'] + 90, tempFrame.iloc[x]['decimalLongitude'] + 180,] += tempFrame.iloc[x]['individualCount']
            print(x, " -> ", earthFrame[tempFrame.iloc[x]['decimalLatitude'] + 90, tempFrame.iloc[x]['decimalLongitude'] + 180,])
            templateFrame = pd.DataFrame(columns = ["long", "lat", "intensity"])
        
        #Trial Work
        print(tempFrame)
        trialFrame = tempFrame.groupby(['decimalLatitude', 'decimalLongitude']).sum()
        trialFrame.drop('year', axis=1)
        trialFrame = trialFrame.reset_index()
        print(trialFrame)
        # trialFrame.to_csv("beta.csv", index = False)
        # trialFrameF = pd.read_csv("beta.csv")
        # print(trialFrameF)

    # Converting DataFrame to csv and then back to DataFrame to standardize the delimiters and have everything work well for pygmt
        # print(tempFrame)
        # print(tempFrame['individualCount'].sum())
        tempFrame['individualCount'] = (tempFrame['individualCount'] / tempFrame['individualCount'].sum()) * 100
        tempFrame.to_csv("alpha.csv", index = False)

        # print(tempFrame)
        # print(tempFrame['individualCount'].sum())
        birdDataF = pd.read_csv("alpha.csv")
        np.savetxt("foo.csv", earthFrame, delimiter=",")

        print(birdDataF)

    # Using the DataFrame made previously and converting the information to a grd file to create a grid image (or heat map) on the globe of migratory population densities
        gBird = pygmt.xyz2grd(data = birdDataF, region = worldRegion, spacing = "222km")
        print(gBird)
        fig = pygmt.Figure()
        fig.basemap(region = worldRegion, projection = "H15c", frame=["a", '+t{} Migratory Changes ({} - {})'.format(birdName, minYear, maxYear)])
        fig.coast(land = "burlywood", water = "lightblue", shorelines = "0.5p,black")
        pygmt.makecpt(cmap="inferno", series=[0, 50])
        fig.grdimage(grid = gBird, transparency = 25, projection = 'H15c')
        fig.colorbar(frame=["x+Sightings"])

    # Magnetic Data
        if includeMagnetism == True:
            mag = "Magnetic Field"
            for k in np.arange(-180, 180):
                for j in np.arange(90, -90, -1):
                    templateFrame = templateFrame.append({"long": k,"lat": j,"intensity": pyIGRF.igrf_value(j, k, 0, i)[6]}, ignore_index=True)

    # Magnetism Aspect
            templateFrame.to_csv('template.csv', index = False)
            reading = pd.read_csv('template.csv')
            frameGrid = pygmt.xyz2grd(data = reading, region = worldRegion, spacing = "277.5km")
            fig.grdimage(grid=frameGrid, transparency = 75, projection = 'H15c', cmap = 'jet')  
            fig.contour(
                pen="0.2p",
                x = reading['long'],
                y = reading['lat'],
                z = reading['intensity'],
                # Contour interval
                levels = 1000,
                # Contour interval marking
                annotation = 5000,
            )
            fig.contour(
                pen = "0.2p,red",
                x = reading['long'],
                y = reading['lat'],
                z = reading['intensity'],
                #contour interval
                levels = 5000,
            )
        fig.savefig("{} {} {}.png".format(birdName, i, mag))

        listOfImages.append("{} {} {}.png".format(birdName, i,mag))

    #Creating the gif of all the maps combined
    gifMaker(listOfImages, "{}from{}to{}_{}.gif".format(birdName, minYear, maxYear, mag), 0.25)
    # <!--- End ---!>

# Method

# avianHeatmap('plovercsv.csv', "American Golden Plover", False, 2000, 2020)
# avianHeatmap('plovercsv.csv', "American Golden Plover", True, 2000, 2020)
# avianHeatmap('pectoralSandpiperUnfiltered.csv', "Pectoral Sandpiper", False, 2000, 2020)
# avianHeatmap('pectoralSandpiperUnfiltered.csv', "Pectoral Sandpiper", True, 2000, 2020)
avianHeatmap('swainsonHawk.csv', "Swainsons Hawk", False, 2000, 2002)
exit()


avianHeatmap('swainsonHawk.csv', "Swainsons Hawk", True, 2000, 2020)
avianHeatmap('whiteRumpedRaw.csv', "White-rumped Sandpiper", False, 2000, 2020)
avianHeatmap('whiteRumpedRaw.csv', "White-rumped Sandpiper", True, 2000, 2020)
avianHeatmap('forktailedUnfiltered.csv', "Fork-tailed Flycatcher", False, 2000, 2020)
avianHeatmap('forktailedUnfiltered.csv', "Fork-tailed Flycatcher", True, 2000, 2020)


# <--- Pectoral Sandpiper Grid Images from 2000 - 2020 (First Trial) --->
pectoralSandpiper = pd.read_csv(r"pectoralSandpiperUnfiltered.csv")

# Standardizing the data as some types are incorrect and some parts of the data have missing values . . .
pSandpiperHeat = pectoralSandpiper[['decimalLongitude', 'decimalLatitude', 'individualCount', 'year']]
pSandpiperHeat['decimalLongitude'] = pSandpiperHeat['decimalLongitude'].fillna(0)
pSandpiperHeat['decimalLatitude'] = pSandpiperHeat['decimalLatitude'].fillna(0)
pSandpiperHeat['decimalLongitude'] = pSandpiperHeat['decimalLongitude'].astype(int)
pSandpiperHeat['decimalLatitude'] = pSandpiperHeat['decimalLatitude'].astype(int)
pSandpiperHeat['individualCount'] = pSandpiperHeat['individualCount'].fillna(1)
pSandpiperHeat['individualCount'] = pSandpiperHeat['individualCount'].astype(int)
pSandpiperHeat = pSandpiperHeat.dropna(subset=['year'])
pSandpiperHeat['year'] = pSandpiperHeat['year'].astype(int)

# Filtering the data to only contain Pectoral Sandpipers in South America, with an individual count of less than 1000 for any given sighting
pSandpiperHeat = pSandpiperHeat.query('5 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')
a = np.zeros(shape=(180, 360))

# Nested for loops to cycle through each year to create a heat map of each
createdPectoralImages = []
for i in np.arange(2000, 2021):
    tempFrame = pSandpiperHeat.query('@i == year')
    for x in np.arange(0, tempFrame.shape[0]):
        a[tempFrame.iloc[x]['decimalLatitude'] + 90, tempFrame.iloc[x]['decimalLongitude'] + 180,] = a[tempFrame.iloc[x]['decimalLatitude'] + 90, tempFrame.iloc[x]['decimalLongitude'] + 180,] + tempFrame.iloc[x]['individualCount']
        print(x, " -> ", a[tempFrame.iloc[x]['decimalLatitude'] + 90, tempFrame.iloc[x]['decimalLongitude'] + 180,])

# Converting DataFrame to csv and then back to DataFrame to standardize the delimiters and have everything work well for pygmt
    tempFrame.to_csv("alpha.csv", index = False)
    pectoralData = pd.read_csv("alpha.csv")
    print(pectoralData)

# Using the DataFrame made previously and converting the information to a grd file to create a grid image (or heat map) on the globe of migratory population densities
    gBird = pygmt.xyz2grd(data = pectoralData, region = worldRegion, spacing = "222km")
    fig = pygmt.Figure()
    fig.basemap(region = worldRegion, projection = "H15c", frame=["a", '+t"Pectoral Sandpiper Migratory Changes (2000 - 2020)"'])
    fig.coast(land = "burlywood", water = "lightblue", shorelines = "0.5p,black")
    fig.grdimage(grid = gBird, transparency = 25, projection = 'H15c', cmap = "inferno")
    fig.colorbar(frame=["x+Sightings"])
    fig.savefig("pectoralBetaMap{}.png".format(i))
    createdPectoralImages.append("pectoralBetaMap{}.png".format(i))

#Creating the gif of all the maps combined
gifMaker(createdPectoralImages, "pectoralBetaGif.gif", 0.25)

# <!--- End ---!>

# <--- American Golden Plover Grid Images from 2000 - 2020 (Next Trial) --->
goldenPlover = pd.read_csv(r"plovercsv.csv")
createdAmericanGoldenImages = []

# Standardizing the data as some types are incorrect and some parts of the data have missing values . . .
goldenPlover = goldenPlover[['decimalLongitude', 'decimalLatitude', 'individualCount', 'year']]
goldenPlover['decimalLongitude'] = goldenPlover['decimalLongitude'].fillna(0)
goldenPlover['decimalLatitude'] = goldenPlover['decimalLatitude'].fillna(0)
goldenPlover['decimalLongitude'] = goldenPlover['decimalLongitude'].astype(int)
goldenPlover['decimalLatitude'] = goldenPlover['decimalLatitude'].astype(int)
goldenPlover['individualCount'] = goldenPlover['individualCount'].fillna(1)
goldenPlover['individualCount'] = goldenPlover['individualCount'].astype(int)
goldenPlover = goldenPlover.dropna(subset=['year'])
goldenPlover['year'] = goldenPlover['year'].astype(int)

# Filtering the data to only contain Pectoral Sandpipers in South America, with an individual count of less than 1000 for any given sighting
goldenPlover = goldenPlover.query('5 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')
a = np.zeros(shape=(180, 360))

# Nested for loops to cycle through each year to create a heat map of each
createdPectoralImages = []
for i in np.arange(2000, 2021, 1):
    tempFrame = goldenPlover.query('@i == year')
    for x in np.arange(0, tempFrame.shape[0]):
        a[tempFrame.iloc[x]['decimalLatitude'] + 90, tempFrame.iloc[x]['decimalLongitude'] + 180,] = a[tempFrame.iloc[x]['decimalLatitude'] + 90, tempFrame.iloc[x]['decimalLongitude'] + 180,] + tempFrame.iloc[x]['individualCount']
        print(x, " -> ", a[tempFrame.iloc[x]['decimalLatitude'] + 90, tempFrame.iloc[x]['decimalLongitude'] + 180,])
        templateFrame = pd.DataFrame(columns = ["long", "lat", "intensity"])

# Converting DataFrame to csv and then back to DataFrame to standardize the delimiters and have everything work well for pygmt
    tempFrame.to_csv("alpha.csv", index = False)
    ploverData = pd.read_csv("alpha.csv")

# Using the DataFrame made previously and converting the information to a grd file to create a grid image (or heat map) on the globe of migratory population densities
    gBird = pygmt.xyz2grd(data = ploverData, region = worldRegion, spacing = "222km")
    fig = pygmt.Figure()
    fig.basemap(region = worldRegion, projection = "H15c", frame=["a", '+t"American Golden-Plover Migratory Changes (2000 - 2020)"'])
    fig.coast(land = "burlywood", water = "lightblue", shorelines = "0.5p,black")
    fig.grdimage(grid = gBird, transparency = 25, projection = 'H15c', cmap = "inferno")
    fig.colorbar(frame=["x+Sightings"])

# Magnetic Data
    for k in np.arange(-180, 180):
        for j in np.arange(90, -90, -1):
            templateFrame = templateFrame.append({"long": k,"lat": j,"intensity": pyIGRF.igrf_value(j, k, 0, i)[6]}, ignore_index=True)

# Magnetism Aspect
    templateFrame.to_csv('template.csv', index = False)
    reading = pd.read_csv('template.csv')
    frameGrid = pygmt.xyz2grd(data = reading, region = worldRegion, spacing = "277.5km")
    fig.grdimage(grid=frameGrid, transparency=50, projection = 'H15c', cmap = 'plasma')  
    fig.contour(
        pen="0.2p",
        x = reading['long'],
        y = reading['lat'],
        z = reading['intensity'],
        # Contour interval
        levels = 1000,
        # Contour interval marking
        annotation = 5000,
    )
    fig.contour(
        pen = "0.2p,red",
        x = reading['long'],
        y = reading['lat'],
        z = reading['intensity'],
        #contour interval
        levels = 5000,
    )

    fig.savefig("ploverBetaMap{}.png".format(i))

    createdAmericanGoldenImages.append("ploverBeta1{}.png".format(i))

#Creating the gif of all the maps combined
gifMaker(createdAmericanGoldenImages, "ploverBeta.gif", 0.25)
# <!--- End ---!>


# <--- Initial trial run of the 1980 IGRF csv file and creating a grid image with it --->
grid = pd.read_csv('gridIgrf1980.csv')  
grid2 = pygmt.xyz2grd(data = grid, region = worldRegion, spacing = "222km")
fig = pygmt.Figure()
fig.basemap(region = worldRegion, projection = "H15c", frame=True)
fig.coast(land = "burlywood", water = "lightblue", shorelines = "0.5p,black")
fig.grdimage(grid = grid2, transparency = 25, projection = 'H15c', cmap = "inferno")
# initial contour attempt
# fig.grdcontour(grid=grid2, projection = 'H15c', interval = 2500)
fig.contour(
    pen = "0.1p,black",
    x = grid['long'],
    y = grid['lat'],
    z = grid['intensity'],
    #contour interval
    levels = 1000,
    #contour interval marking
    annotation = 5000,
)
fig.contour(
    pen = "0.2p,red",
    x = grid['long'],
    y = grid['lat'],
    z = grid['intensity'],
    #contour interval
    levels=5000,
)
fig.colorbar(frame=["x+lMagnetic Strength (nT)"])
fig.savefig("TESTBED.png")
# <!--- End ---!>

# mag = igrf.igrf('2010-07-12', glat=65, glon=-148, alt_km=100)
# spacing = "111km"

# <--- Creating maps of Earth's magnetic field from 1900 to 2020 over 5 year intervals --->
for i in np.arange(1900.5, 2020.5, 5.0):
    templateFrame = pd.DataFrame(columns = ["long", "lat", "intensity"])
    for x in np.arange(-180, 180):
        for y in np.arange(90, -90, -1):
            templateFrame = templateFrame.append({"long": x,"lat": y,"intensity": pyIGRF.igrf_value(y, x, 0, i)[6]}, ignore_index=True)

    print(templateFrame)
    # newDf = templateFrame(index = False)
    templateFrame.to_csv('template.csv', index = False)
    reading = pd.read_csv('template.csv')
    print(reading)
    frameGrid = pygmt.xyz2grd(data = reading, region = worldRegion, spacing = "277.5km")
    fig = pygmt.Figure()
    fig.basemap(region=worldRegion, projection="H15c", frame=["a", '+t"Changes in Earths Magnetic field (1900 - 2020)"'])
    fig.coast(land="burlywood", water="lightblue", shorelines = True)
    fig.grdimage(grid=frameGrid, transparency=30, projection = 'H15c')  
    fig.contour(
        pen = "0.2p",

        x = reading['long'],
        y = reading['lat'],
        z = reading['intensity'],
        #contour interval
        levels = 1000,
        #contour interval marking
        annotation = 5000,
    )
    fig.contour(
        pen = "0.2p,red",
        x = reading['long'],
        y = reading['lat'],
        z = reading['intensity'],
        #contour interval
        levels = 5000,
    )
    fig.savefig("run2_EMF_Field{}.png".format(i))
# <!--- End ---!>


# <--- Creating maps of Earths magnetic field from 2000 to 2020 over 1 year intervals --->
for i in np.arange(2000.5, 2021.5, 1):
    templateFrame = pd.DataFrame(columns = ["long", "lat", "intensity"])
    for x in np.arange(-180, 180):
        for y in np.arange(90, -90, -1):
            templateFrame = templateFrame.append({"long": x,"lat": y,"intensity": pyIGRF.igrf_value(y, x, 0, i)[6]}, ignore_index=True)

    print(templateFrame)
    # newDf = templateFrame(index = False)
    templateFrame.to_csv('template.csv', index = False)
    reading = pd.read_csv('template.csv')
    frameGrid = pygmt.xyz2grd(data = reading, region = worldRegion, spacing = "277.5km")
    fig = pygmt.Figure()
    fig.basemap(region=worldRegion, projection="H15c", frame=["a", '+t"Changes in Earths Magnetic field (2000 - 2020)"'])
    fig.coast(land="burlywood", water="lightblue", shorelines = True)
    fig.grdimage(grid=frameGrid, transparency=30, projection = 'H15c')  
    fig.contour(
        pen="0.2p",
        x = reading['long'],
        y = reading['lat'],
        z = reading['intensity'],
        #contour interval
        levels = 1000,
        #contour interval marking
        annotation = 5000,
    )
    fig.contour(
        pen = "0.2p,red",
        x = reading['long'],
        y = reading['lat'],
        z = reading['intensity'],
        #contour interval
        levels = 5000,
    )
    fig.savefig("Closer_EMF_Field{}.png".format(i))
# <!--- End ---!>

grid = pd.read_csv('gridIgrf1980.csv')
print(grid)
grid2 = pygmt.xyz2grd(data = grid, region = worldRegion, spacing = "222km")
# grd = pygmt.xyz2grd(data = igrfDataFull, x = igrfDataFull.long, y = igrfDataFull.lat, z = igrfDataFull.intensity)
fig = pygmt.Figure()

#just gradient for 1980
fig.grdimage(grid = grid2, projection = 'H15c')
print(grid)
fig.show()

#just contour lines for 1980
fig = pygmt.Figure()
fig.grdcontour(grid = grid2)
fig.show()

#full map put together with previous elements
fig = pygmt.Figure()
fig.basemap(region = worldRegion, projection = "H15c", frame = True)
fig.coast(land = "#666666", water = "lightblue", shorelines = True)
fig.grdimage(grid = grid2, transparency = 25, projection = 'H15c')
# initial contour attempt
# fig.grdcontour(grid=grid2, projection = 'H15c', interval = 2500)
fig.contour(
    pen = "0.1p",
    x = grid['long'],
    y = grid['lat'],
    z = grid['intensity'],
    levels = 1000,
    annotation = 5000,
)
fig.contour(
    pen = "0.2p,red",
    x = grid['long'],
    y = grid['lat'],
    z = grid['intensity'],
    levels = 5000,
)
fig.show()

# read in the xyz file into a 1D numpy array

fig = pygmt.Figure()
fig.basemap(region = [-180, 180, -90, 90], projection = "H15c", frame = True)
fig.coast(land = "#666666", water = "skyblue")

fig.contour(
    pen = "0.1p",
    # pass the data as 3 1d data columns
    x = grid['long'],
    y = grid['lat'],
    z = grid['intensity'],
    # set the contours z values intervals to 1000
    levels = 1000,
    # set the contours annotation intervals to 5000
    annotation = 5000,
)

#make index contours red
fig.contour(
    pen="0.2p,red",
    # pass the data as 3 1d data columns
    x = grid['long'],
    y = grid['lat'],
    z = grid['intensity'],
    # set the contours z values intervals to 5000
    levels = 5000,
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