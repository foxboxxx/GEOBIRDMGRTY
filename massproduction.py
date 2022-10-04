from multiprocessing import current_process
# from re import T
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
import gc

print("Working...")
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
exportedDf = pd.DataFrame(columns=['species', 'family', 'mag r-value', 'mag p-value', 'prcp r-value', 'prcp p-value', 'temp r-value', 'temp p-value'])


# # Bird Data First Iteration
# file = pd.read_csv(r'X:\additionalmigratorydata\0000831-220831081235567.csv', sep = '\t')
# # file = pd.read_csv(r'plovercsv.csv', error_bad_lines=False)
# iterations = file['species'].unique()

# for species in iterations:
    
#     print(species)

#     # Converting Raw File to dataframes
#     rawFile = file.query('species == @species')
#     rawFile = rawFile.loc[rawFile['occurrenceStatus'] == 'PRESENT', ['species', 'family', 'eventDate', 'individualCount', 'decimalLatitude', 'decimalLongitude', 'day', 'month', 'year']].copy()
#     familyName = str(rawFile['family'].unique()[0])

#     # Breaking up the data into two regions, one within range of the South Atlantic Anomaly and one covering the entirety of South America
#     smallerRegion = rawFile.query('-25 >= decimalLatitude >= -35 & -55 >= decimalLongitude >= -65 & individualCount < 1000').copy()
#     largerRegion = rawFile.query('0 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')

#     # Formatting it by year (2000 - 2020 for now)
#     smallCount = largeCount = pd.DataFrame({'year':[], 'countS':[]})
#     smallCount = yearCount(smallCount, smallerRegion, 2000, 2020 )
#     largeCount = yearCount(largeCount, largerRegion, 2000, 2020)

#     ''' Graphing Code
#     SAAMagData = SAAMagData.copy().query('2000 <= year < 2020')
#     magStrength3060S = SAAMagData['intensity'].values.tolist()

#     smallCountPercentages = (smallCount['countS'].copy() / largeCount['countS'].copy()) * 100
#     '''

#     # Dataframe for pearson correlation tests
#     percentages = (smallCount['countS'].copy() / largeCount['countS'].copy()) * 100
#     print(SAAMagData)
#     print(percentages)

#     magRVal = float(stats.pearsonr(SAAMagData['intensity'],percentages)[0])
#     magPVal = float(stats.pearsonr(SAAMagData['intensity'], percentages)[1])
#     tempRVal = float(stats.pearsonr(CLMTempData,percentages)[0])
#     tempPVal = float(stats.pearsonr(CLMTempData, percentages)[1])
#     prcpRVal = float(stats.pearsonr(CLMPrcpData,percentages)[0])
#     prcpPVal = float(stats.pearsonr(CLMPrcpData, percentages)[1])

#     print("Magnetic r-value: " + str(magRVal))
#     print("Magnetic p-value: " + str(magPVal))
#     print("Temperature r-value: " + str(tempRVal))
#     print("Temperature p-value: " + str(tempPVal))
#     print("Precipitation r-value: " + str(prcpRVal))
#     print("Precipitation p-value: " + str(prcpPVal))

#     currentBirdAnalysis = pd.DataFrame([[species, familyName, magRVal, magPVal, prcpRVal, prcpPVal, tempRVal, tempPVal]], columns=['species', 'family', 'mag r-value', 'mag p-value', 'prcp r-value', 'prcp p-value','temp r-value', 'temp p-value'])
#     finalDf = pd.concat([finalDf, currentBirdAnalysis])
    
#     print(species)

# print(finalDf)
# writer = pd.ExcelWriter('migratoryData.xlsx', engine='xlsxwriter')
# finalDf.to_excel(writer, sheet_name='Sheet1')
# workbook  = writer.book
# worksheet = writer.sheets['Sheet1']
# (max_row, max_col) = finalDf.shape
# # Conditional Formatting (needs more editing)
# # worksheet.conditional_format(3, 1, max_row, max_col, {'type': '3_color_scale'})
# writer.save()


# Bird Data Second Iteration:

# chunksize = 10 ** 2
# with pd.read_csv(r'X:\additionalmigratorydata\0000831-220831081235567.csv', chunksize = chunksize, sep='\t+') as reader:
#     for chunk in reader:
#         chunk.to_csv("b.csv")
#         exit()


# chunksize = 1 * (10 ** 7)
# data = pd.read_csv(r'X:\additionalmigratorydata\0000831-220831081235567.csv', sep = '\t', nrows = chunksize)
# data.to_csv("b.csv")

# exit()

# chunksize = 1 * (10 ** 7)
# data = pd.read_csv(r'X:\additionalmigratorydata\0000831-220831081235567.csv', sep = '\t', skiprows = 5, nrows = 5)
# data.to_csv("b.csv")
# exit()

# Creating DataFrame for analysis
finalDf = pd.DataFrame(columns=['species', 'family', 
                                '2000SA', '2000A', 
                                '2001SA', '2001A',
                                '2002SA', '2002A', 
                                '2003SA', '2003A',
                                '2004SA', '2004A', 
                                '2005SA', '2005A',
                                '2006SA', '2006A',
                                '2007SA', '2007A',
                                '2008SA', '2008A',
                                '2009SA', '2009A',
                                '2010SA', '2010A',
                                '2011SA', '2011A',
                                '2012SA', '2012A',
                                '2013SA', '2013A',
                                '2014SA', '2014A',
                                '2015SA', '2015A',
                                '2016SA', '2016A',
                                '2017SA', '2017A',
                                '2018SA', '2018A',
                                '2019SA', '2019A',
                                'WinterSA', 'WinterNA', 'WinterElse',
                                'SpringSA', 'SpringNA', 'SpringElse',
                                'SummerSA', 'SummerNA', 'SummerElse', 
                                'FallSA', 'FallNA', 'FallElse',
                                ])


chunksize = 5 * (10 ** 6)
iterr = 0
with pd.read_csv(r'X:\additionalmigratorydata\0000831-220831081235567.csv', sep = '\t', chunksize = chunksize, on_bad_lines = 'skip') as reader:
    for chunk in reader:
        # data = pd.read_csv(r'X:\additionalmigratorydata\0000831-220831081235567.csv', sep = '\t', skiprows = [i for i in range(1, (chunks * chunksize))], nrows = chunksize, )
        iterr = iterr + 1
        print("Chunk # ", iterr)
        iterations = chunk['species'].unique()
        for species in iterations:
            
            # Filtering data down to the bare bones
            filt = chunk.query('species == @species')
            filt = filt.loc[filt['occurrenceStatus'] == 'PRESENT', ['species', 'family', 'eventDate', 'individualCount', 'decimalLatitude', 'decimalLongitude', 'day', 'month', 'year']].copy()
            familyName = str(filt['family'].unique()[0])
            
            # Breaking up the data into two regions, one within range of the South Atlantic Anomaly and one covering the entirety of South America
            smallerRegion = filt.query('-25 >= decimalLatitude >= -35 & -55 >= decimalLongitude >= -65 & individualCount < 1000').copy()
            largerRegion = filt.query('0 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')
            naRegion = filt.query('0 <= decimalLatitude <= 70 & -135 <= decimalLongitude <= -60 & individualCount < 1000').copy()
            # Migratory Determination
            winterSA = (largerRegion.query('month <= 2 | month == 12').copy()).shape[0]
            winterNA = (naRegion.query('month <= 2 | month == 12').copy()).shape[0]
            winterE = ((filt.query('month <= 2 | month == 12').copy()).shape[0]) - (winterSA + winterNA)
            
            springSA = (largerRegion.query('3 <= month <= 5').copy()).shape[0]
            springNA = (naRegion.query('3 <= month <= 5').copy()).shape[0]
            springE = ((filt.query('3 <= month <= 5').copy()).shape[0]) - (springSA + springNA)
            
            summerSA = (largerRegion.query('6 <= month <= 8').copy()).shape[0]
            summerNA = (naRegion.query('6 <= month <= 8').copy()).shape[0]
            summerE = ((filt.query('6 <= month <= 8').copy()).shape[0]) - (summerSA + summerNA)
            
            fallSA = (largerRegion.query('9 <= month <= 11').copy()).shape[0]
            fallNA = (naRegion.query('9 <= month <= 11').copy()).shape[0]
            fallE = ((filt.query('9 <= month <= 11').copy()).shape[0]) - (fallSA + fallNA)                   
            
            # Formatting it by year (2000 - 2020 for now)
            smallCount = largeCount = pd.DataFrame({'year':[], 'countS':[]})
            smallCount = yearCount(smallCount, smallerRegion, 2000, 2020 )
            largeCount = yearCount(largeCount, largerRegion, 2000, 2020)
            
            # Storing data in final dataframe
            dataB = [[species, familyName, smallCount['countS'][0], largeCount['countS'][0],
                                                            smallCount['countS'][1], largeCount['countS'][1], 
                                                            smallCount['countS'][2], largeCount['countS'][2],
                                                            smallCount['countS'][3], largeCount['countS'][3], 
                                                            smallCount['countS'][4], largeCount['countS'][4], 
                                                            smallCount['countS'][5], largeCount['countS'][5], 
                                                            smallCount['countS'][6], largeCount['countS'][6], 
                                                            smallCount['countS'][7], largeCount['countS'][7], 
                                                            smallCount['countS'][8], largeCount['countS'][8], 
                                                            smallCount['countS'][9], largeCount['countS'][9], 
                                                            smallCount['countS'][10], largeCount['countS'][10], 
                                                            smallCount['countS'][11], largeCount['countS'][11], 
                                                            smallCount['countS'][12], largeCount['countS'][12], 
                                                            smallCount['countS'][13], largeCount['countS'][13], 
                                                            smallCount['countS'][14], largeCount['countS'][14], 
                                                            smallCount['countS'][15], largeCount['countS'][15], 
                                                            smallCount['countS'][16], largeCount['countS'][16], 
                                                            smallCount['countS'][17], largeCount['countS'][17], 
                                                            smallCount['countS'][18], largeCount['countS'][18], 
                                                            smallCount['countS'][19], largeCount['countS'][19],
                                                            winterSA, winterNA, winterE,
                                                            springSA, springNA, springE, 
                                                            summerSA, summerNA, summerE,
                                                            fallSA, fallNA, fallE,]]
            currentBird = pd.DataFrame(dataB, columns = ['species', 'family', 
                                                        '2000SA', '2000A', 
                                                        '2001SA', '2001A',
                                                        '2002SA', '2002A', 
                                                        '2003SA', '2003A',
                                                        '2004SA', '2004A', 
                                                        '2005SA', '2005A',
                                                        '2006SA', '2006A',
                                                        '2007SA', '2007A',
                                                        '2008SA', '2008A',
                                                        '2009SA', '2009A',
                                                        '2010SA', '2010A',
                                                        '2011SA', '2011A',
                                                        '2012SA', '2012A',
                                                        '2013SA', '2013A',
                                                        '2014SA', '2014A',
                                                        '2015SA', '2015A',
                                                        '2016SA', '2016A',
                                                        '2017SA', '2017A',
                                                        '2018SA', '2018A',
                                                        '2019SA', '2019A',
                                                        'WinterSA', 'WinterNA', 'WinterElse',
                                                        'SpringSA', 'SpringNA', 'SpringElse',
                                                        'SummerSA', 'SummerNA', 'SummerElse', 
                                                        'FallSA', 'FallNA', 'FallElse',
                                                        ])
            print("#1", currentBird)
            if str(currentBird['species'][0]) not in finalDf['species'].unique():
                finalDf = pd.concat([finalDf, currentBird], ignore_index = True)
                
            elif str(currentBird['species'][0]) in finalDf['species'].unique():
                print("#1.5", currentBird.iloc[0][1])
                speciesName = str(currentBird['species'][0])
                print("#2", len(currentBird))
                print("#3", currentBird.shape[1])
                for x in np.arange(2, currentBird.shape[1]):
                    y = finalDf[finalDf['species'] == speciesName].index[0]
                    print("#5", y, speciesName, species)
                    finalDf.iat[y,x] = (finalDf.iloc[y][x] + currentBird.iloc[0][x])
            finalDf.to_csv("mbirdanalysis5.csv")  
            # finalDf.to_csv("mbirdanalysis1.csv")
            # exit()
        print(finalDf)    
        #finalDf.to_csv("mbirdanalysis3.csv")



file = pd.read_csv(r'mbirdanalysis5.csv')
sbird = file['species'].unique()

for birds in sbird:
    
    percentages = pd.DataFrame(columns = ['percentages'])
    x = file[file['species'] == birds].index[0]
    familyName = file.loc[x]['family']
    print(file.iloc[x][42])
    for i in np.arange(3,43,2):
        if file.iat[x,(i + 1)] != 0:
            percentages.loc[i] = ((file.iat[x,i] / file.iat[x,(i + 1)]) * 100)
        elif file.iat[x,(i + 1)] == 0:
            percentages.loc[i] = 0
        
# Dataframe for pearson correlation tests

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

    currentBirdAnalysis = pd.DataFrame([[birds, familyName, magRVal, magPVal, prcpRVal, prcpPVal, tempRVal, tempPVal,winterSA, winterNA, winterE,
                                                            springSA, springNA, springE, 
                                                            summerSA, summerNA, summerE,
                                                            fallSA, fallNA, fallE,]], columns=['species', 'family', 'mag r-value', 'mag p-value', 'prcp r-value', 'prcp p-value','temp r-value', 'temp p-value','WinterSA', 'WinterNA', 'WinterElse',
                                                        'SpringSA', 'SpringNA', 'SpringElse',
                                                        'SummerSA', 'SummerNA', 'SummerElse', 
                                                        'FallSA', 'FallNA', 'FallElse',])
    exportedDf = pd.concat([exportedDf, currentBirdAnalysis])

print(exportedDf)
writer = pd.ExcelWriter('migratoryDataNightly.xlsx', engine='xlsxwriter')
exportedDf.to_excel(writer, sheet_name='Sheet1')
workbook  = writer.book
worksheet = writer.sheets['Sheet1']
(max_row, max_col) = exportedDf.shape
# Conditional Formatting (needs more editing)
# worksheet.conditional_format(3, 1, max_row, max_col, {'type': '3_color_scale'})
writer.save()
    


    
#     filt = data.query('species == @species')
#     filt = filt.loc[filt['occurrenceStatus'] == 'PRESENT', ['species', 'family', 'eventDate', 'individualCount', 'decimalLatitude', 'decimalLongitude', 'day', 'month', 'year']].copy()

#     # Breaking up the data into two regions, one within range of the South Atlantic Anomaly and one covering the entirety of South America
#     smallerRegion = filt.query('-25 >= decimalLatitude >= -35 & -55 >= decimalLongitude >= -65 & individualCount < 1000').copy()
#     largerRegion = filt.query('0 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')

#     # Formatting it by year (2000 - 2020 for now)
#     smallCount = largeCount = pd.DataFrame({'year':[], 'countS':[]})
#     smallCount = yearCount(smallCount, smallerRegion, 2000, 2020 )
#     largeCount = yearCount(largeCount, largerRegion, 2000, 2020)

#     percentages = (smallCount['countS'].copy() / largeCount['countS'].copy()) * 100

#     magRVal = float(stats.pearsonr(SAAMagData['intensity'],percentages)[0])
#     magPVal = float(stats.pearsonr(SAAMagData['intensity'], percentages)[1])
#     tempRVal = float(stats.pearsonr(CLMTempData,percentages)[0])
#     tempPVal = float(stats.pearsonr(CLMTempData, percentages)[1])
#     prcpRVal = float(stats.pearsonr(CLMPrcpData,percentages)[0])
#     prcpPVal = float(stats.pearsonr(CLMPrcpData, percentages)[1])

#     currentBirdAnalysis = pd.DataFrame([[species, familyName, magRVal, magPVal, prcpRVal, prcpPVal, tempRVal, tempPVal]], columns=['species', 'family', 'mag r-value', 'mag p-value', 'prcp r-value', 'prcp p-value','temp r-value', 'temp p-value'])
    

# print(data)










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
