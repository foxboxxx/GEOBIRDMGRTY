import pandas as pd
import numpy as np
import pygmt
import datetime
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from scipy import stats
import seaborn as sb
from functions import lineBreak, histogramMaker, doubleYAxisPlotMaker, yearCount, monthCount, scatterPlot, plotter

birdList = []

for bird in birdList:
    # Magnetic Data
    SAAMagData = pd.read_csv(r"3060intensity.csv")

    # Converting Raw File to dataframes
    rawFile = pd.read_csv('{}'.format(bird))
    rawFile = rawFile.loc[rawFile['occurrenceStatus'] == 'PRESENT', ['eventDate', 'individualCount', 'decimalLatitude', 'decimalLongitude', 'day', 'month', 'year']].copy()

    # Breaking up the data into two regions, one within range of the South Atlantic Anomaly and one covering the entirety of South America
    smallerRegion = rawFile.query('-25 >= decimalLatitude >= -35 & -55 >= decimalLongitude >= -65 & individualCount < 1000').copy()
    largerRegion = rawFile.query('0 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')

    # Formatting it by year (2000 - 2020 for now)
    smallCount = yearCount(smallCount, smallerRegion, 2000, 2020 )
    largeCount = yearCount(largeCount, largerRegion, 2000, 2020)

    SAAMagData = SAAMagData.copy().query('2000 <= year < 2020')
    magStrength3060S = SAAMagData['intensity'].values.tolist()

    smallCountPercentages = (smallCount['countS'].copy() / largeCount['countS'].copy()) * 100














