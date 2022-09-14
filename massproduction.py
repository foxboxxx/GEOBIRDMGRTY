from multiprocessing import current_process
import pandas as pd
import numpy as np
import pygmt
import datetime
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from scipy import stats
import seaborn as sb
import xlsxwriter
from functions import lineBreak, histogramMaker, doubleYAxisPlotMaker, yearCount, monthCount, scatterPlot, plotter

# Magnetic Data
SAAMagData = pd.read_csv(r"3060intensity.csv").query('2000 <= year < 2020')

# Climate Data
climateDataGen = pd.read_csv(r"weatherDataNew.csv")
cDataFiltered = {'STATION': climateDataGen['STATION'], 'LATITUDE': climateDataGen['LATITUDE'], 'LONGITUDE': climateDataGen['LONGITUDE'], 'DATE': climateDataGen['DATE'], 'PRCP': climateDataGen['PRCP'], 'TAVG': climateDataGen['TAVG'], 'TMAX': climateDataGen['TMAX'], 'TMIN': climateDataGen['TMIN']}
cdf1 = pd.DataFrame(cDataFiltered)
cdf1 = cdf1.query('STATION == "PA000086086"').copy()
yearlyTemps = pd.DataFrame(columns = ['year', 'average'])
yearlyPrcps = pd.DataFrame(columns = ['year', 'average'])
for i in range(0,20):
    total = 0
    count = 0
    for j in range(0, cdf1['DATE'].size):
        string1 = cdf1.iloc[j]['DATE']
        if(string1[:4] == str(2000 + i)):
            total += cdf1.iloc[j]['TAVG']
            count += 1
    yearlyTemps.loc[len(yearlyTemps.index)] = [2000 + i, (total/count)] 
for i in range(0,20):
    total = 0
    count = 0
    for j in range(0, cdf1['DATE'].size):
        string1 = cdf1.iloc[j]['DATE']
        if string1[:4] == str(2000 + i) and cdf1.iloc[j]['PRCP'] >= 0:
            total += cdf1.iloc[j]['PRCP']
            count += 1
    yearlyPrcps.loc[len(yearlyPrcps.index)] = [2000 + i, (total/count)]
CLMTempData = yearlyTemps['average'].values.tolist()
CLMPrcpData = yearlyPrcps['average'].values.tolist()

# Creating DataFrame for analysis
finalDf = pd.DataFrame(columns=['species', 'family', 'mag r-value', 'mag p-value', 'prcp r-value', 'prcp p-value', 'temp r-value', 'temp p-value'])

# Bird Data
file = pd.read_csv(r'X:\additionalmigratorydata\0000831-220831081235567.csv', error_bad_lines=False)
iterations = file['species'].unique()

for species in iterations:

    # Converting Raw File to dataframes
    rawFile = file.query('species == @species')
    rawFile = rawFile.loc[rawFile['occurrenceStatus'] == 'PRESENT', ['species', 'family', 'eventDate', 'individualCount', 'decimalLatitude', 'decimalLongitude', 'day', 'month', 'year']].copy()
    familyName = str(rawFile['family'].unique()[0])

    # Breaking up the data into two regions, one within range of the South Atlantic Anomaly and one covering the entirety of South America
    smallerRegion = rawFile.query('-25 >= decimalLatitude >= -35 & -55 >= decimalLongitude >= -65 & individualCount < 1000').copy()
    largerRegion = rawFile.query('0 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')


    # Formatting it by year (2000 - 2020 for now)
    smallCount = largeCount = pd.DataFrame({'year':[], 'countS':[]})
    smallCount = yearCount(smallCount, smallerRegion, 2000, 2020 )
    largeCount = yearCount(largeCount, largerRegion, 2000, 2020)

    ''' Graphing Code
    SAAMagData = SAAMagData.copy().query('2000 <= year < 2020')
    magStrength3060S = SAAMagData['intensity'].values.tolist()

    smallCountPercentages = (smallCount['countS'].copy() / largeCount['countS'].copy()) * 100

    '''
    # Dataframe for pearson correlation tests
    percentages = (smallCount['countS'].copy() / largeCount['countS'].copy()) * 100
    print(SAAMagData)
    print(percentages)

    magRVal = float(stats.pearsonr(SAAMagData['intensity'],percentages)[0])
    magPVal = float(stats.pearsonr(SAAMagData['intensity'], percentages)[1])
    tempRVal = float(stats.pearsonr(CLMTempData,percentages)[0])
    tempPVal = float(stats.pearsonr(CLMTempData, percentages)[1])
    prcpRVal = float(stats.pearsonr(CLMPrcpData,percentages)[0])
    prcpPVal = float(stats.pearsonr(CLMPrcpData, percentages)[1])

    print("Magnetic r-value: " + str(magRVal))
    print("Magnetic p-value: " + str(magPVal))
    print("Temperature r-value: " + str(tempRVal))
    print("Temperature p-value: " + str(tempPVal))
    print("Precipitation r-value: " + str(prcpRVal))
    print("Precipitation p-value: " + str(prcpPVal))

    currentBirdAnalysis = pd.DataFrame([[species, familyName, magRVal, magPVal, prcpRVal, prcpPVal, tempRVal, tempPVal]], columns=['species', 'family', 'mag r-value', 'mag p-value', 'prcp r-value', 'prcp p-value','temp r-value', 'temp p-value'])
    finalDf = pd.concat([finalDf, currentBirdAnalysis])

print(finalDf)
writer = pd.ExcelWriter('migratoryData.xlsx', engine='xlsxwriter')
finalDf.to_excel(writer, sheet_name='Sheet1')
workbook  = writer.book
worksheet = writer.sheets['Sheet1']
(max_row, max_col) = finalDf.shape
# Conditional Formatting (needs more editing)
# worksheet.conditional_format(3, 1, max_row, max_col, {'type': '3_color_scale'})
writer.save()










# <--! Proof of concept for heatmap analysis of correlation findings:!-->
# basicFrame = {
#     "Migratory Data": percentages,
#     "Magnetic Data": SAAMagData['intensity'].values.tolist(),
#     "Temperature Data": CLMTempData,
#     "Precipitation Data": CLMPrcpData
#     }
# data = pd.DataFrame(basicFrame)
# print(data)
# print(data.corr())

# # sb.heatmap(data.corr(method = 'pearson'))

# plt.figure(figsize=(8, 12))
# heatmap = sb.heatmap(data.corr()[['Migratory Data']].sort_values(by='Migratory Data', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
# heatmap.set_title('Correlation Graph', fontdict={'fontsize':18}, pad=16)
# plt.show()  
