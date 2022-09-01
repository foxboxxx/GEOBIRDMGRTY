import pandas as pd
import numpy as np
import pygmt
import datetime
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from scipy import stats
import seaborn as sb
from sorting import histogramMaker, doubleYAxisPlotMaker

birdList = []

for bird in birdList:
    rawFile = pd.read_csv('{}'.format(bird))
    rawFile = rawFile.loc[rawFile['occurrenceStatus'] == 'PRESENT', ['eventDate', 'individualCount', 'decimalLatitude', 'decimalLongitude', 'day', 'month', 'year']].copy()
    filteredFile = rawFile.query('-25 >= decimalLatitude >= ')