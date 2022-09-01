import pandas as pd
import numpy as np
import pygmt
import datetime
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from scipy import stats
import seaborn as sb
from sorting import histogramMaker, doubleYAxisPlotMaker, yearCount, monthCount

birdList = []

for bird in birdList:
    rawFile = pd.read_csv('{}'.format(bird))
    rawFile = rawFile.loc[rawFile['occurrenceStatus'] == 'PRESENT', ['eventDate', 'individualCount', 'decimalLatitude', 'decimalLongitude', 'day', 'month', 'year']].copy()

    smallerRegion = rawFile.query('-25 >= decimalLatitude >= -35 & -55 >= decimalLongitude >= -65 & individualCount < 1000').copy()
    largerRegion = rawFile.query('0 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')

    smallCount = yearCount(smallCount, smallerRegion, 2000, 2020 )
    largeCount = yearCount(largeCount, largerRegion, 2000, 2020)








