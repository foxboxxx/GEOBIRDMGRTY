import pandas as pd
import numpy as np
import pygmt

def yearCount(smallSet, bigSet, yearMin, yearMax):
    for i in np.arange(yearMin, yearMax):
        smallSet = smallSet.append(pd.DataFrame({'year': i, 'countS': bigSet.query('year == @i')['individualCount'].sum()}, index = [0]), ignore_index = True)
    return smallSet

def mapMaker():
    print("WIP")

def graphMaker(radius, yearMin, yearMax, magMin, magMax, percentMin, percentMax, birdName, rawBirdFile, populationCap, rawMagData, region, latitude, longitude, latMin, latMax, lonMin, lonMax):
    filteredMagData = pd.read_csv(rawMagData)
    magData = filteredMagData.query('@yearMin <= year <= @yearMax')
    latName = ""
    lonName = ""
    rName = str(radius)
    rawBirdData = pd.read_csv(rawBirdFile)
    rawBirdData = rawBirdData.loc[rawBirdData['occurrenceStatus']=='PRESENT', ['eventDate', 'individualCount', 'decimalLatitude', 'decimalLongitude', 'day', 'month', 'year']].copy()

    filteredBig = rawBirdData.query('@latMin <= decimalLatitude <= @latMax & @lonMin <= decimalLongitude <= @lonMax & individualCount < @populationCap')
    filteredSmall = rawBirdData.query('(@latitude - @radius) <= decimalLatitude <= (@latitude + @radius) & (@longitude - @radius) <= decimalLongitude <= (@longitude + @radius) & individualCount < @populationCap')

    if(latitude >= 0):
        latName = str(abs(latitude)) + "N"
    else:
        latName = str(abs(latitude)) + "S"

    if(longitude >= 0):
        lonName = str(abs(longitude)) + "E"
    else:
        lonName = str(abs(longitude)) + "W"

    smallCount = largeCount = pd.DataFrame({'year':[], 'countS':[]})
    smallCount = yearCount(smallCount, filteredSmall, yearMin, yearMax)
    largeCount = yearCount(largeCount, filteredBig, yearMin, yearMax)
    title = str(rName + "R_Analysis_of_" + birdName + "_%_In_" + region + "_(" + latName + lonName + ")")

    correlation = pygmt.Figure()
    correlation.basemap(region=[yearMin,yearMax,magMin,magMax], projection = "X15c/15c", frame = ["St", "xaf+lYear"])
    correlation.basemap(frame=["a", '+t' + title])
    with pygmt.config(
        MAP_FRAME_PEN = "blue",
        MAP_TICK_PEN = "blue", 
        FONT_ANNOT_PRIMARY = "blue", 
        FONT_LABEL = "blue",
    ):
        correlation.basemap(frame=["W", 'yaf+l"Magnetic Strength (nT)"'])
        
    correlation.plot(x = magData.year, y = magData.intensity, pen="1p,blue")
    correlation.plot(x = magData.year, y = magData.intensity, style="c0.2c", color = "blue", label = '"Magnetic Strength (nT)"')

    with pygmt.config(
        MAP_FRAME_PEN = "red",
        MAP_TICK_PEN = "red",
        FONT_ANNOT_PRIMARY = "red",
        FONT_LABEL = "red",
    ):
        correlation.basemap(region = [yearMin,yearMax,percentMin,percentMax], frame=["E", 'yaf+l' + birdName + "_Population_Density_%"])
        
    correlation.plot(x = smallCount.year, y = (smallCount.countS/largeCount.countS)*100, pen = "1p,red")
    correlation.plot(x = smallCount.year, y = (smallCount.countS/largeCount.countS)*100, style = "s0.25c", color = "red", label = '"PsBird Population Density %"')

    correlation.legend(position = "jTL+o0.1c", box = True)
    correlation.savefig(birdName + "Correlation" + latName + lonName + rName + "R.png",show = False)
    print("{}R Analysis of {} % In {} ({}{}) Successfully Updated.".format(rName, birdName, region, latName, lonName))

#(radius, yearMin, yearMax, magMin, magMax, percentMin, percentMax, birdName, rawBirdFile, populationCap, rawMagData, region, latitude, longitude, latMin, latMax, lonMin, lonMax)
graphMaker(5, 2000, 2020, 22000, 26000, 0, 100, "AgBird", "plovercsv.csv", 1000, "3060fullmag.csv", "South_America", -30, -60, -60, 0, -90, -30)