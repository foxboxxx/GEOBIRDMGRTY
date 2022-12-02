import pandas as pd
import numpy as np
import pygmt
import datetime
import matplotlib.pyplot as plt
import matplotlib as matplotlib
# import statsmodels.graphics.api as smg
# import statsmodels.api as sm
from scipy import stats
import seaborn as sb
from functions import lineBreak, histogramMaker, doubleYAxisPlotMaker, yearCount, monthCount, scatterPlot, plotter

print("Begininng Updating Sequence...")
#testData = pd.read_csv(r"https://docs.google.com/spreadsheets/d/e/2PACX-1vQ1QvEdtQLKMGXxyN0QhQL46AdUdJ9iJVB-50NBHe1s_tnx2KCymG0ZYX43eB-SEjTl9Dj_5WHU8qj3/pub?output=csv")
magneticData37 = pd.read_csv(r"magneticData37.csv")
magData3060 = pd.read_csv(r"3060intensity.csv")
fullMagData3060 = pd.read_csv(r"3060fullMag.csv")
#print(magneticData37)
birdData = pd.read_csv(r"filteredPloverData.csv")
magData30N90W = pd.read_csv(r'30N90Wdata.csv')
kpIndex = pd.read_csv(r'kpindex.csv')

kpy = kpIndex.query('2000 <= YYY < 2020')
print(kpy)

kpIntoMonths = pd.DataFrame()
# trouble shooting
# print(((kpy['days'] - 24836.875)/30.4375))

kpIntoMonths['month'] = ((kpy['days'] - 24836.875)/30.4375)
kpIntoMonths['index'] = kpy['Kp']
kpIntoMonths['year'] = kpy['YYY']
print(kpIntoMonths)


#useless
kpAvg = pd.DataFrame()
kpAvg['month'] = np.arange(0, 240)
kpAvg['avInd'] = ""
listWWW = []
for i in kpAvg['month']:
    target = kpIntoMonths.query('@i <= month <= @i + 1')
    listWWW.append(target['index'].sum() / (len(target['index'])))
    print(listWWW)
kpAvg['avInd'] = listWWW
print(kpAvg)
#end of useless


print(kpIntoMonths)
print("KP Index Successfully Filtered to 2000 - 2020")


climateDataGen = pd.read_csv(r"weatherDataNew.csv")
print(climateDataGen)
print(climateDataGen['STATION'])
cDataFiltered = {'STATION': climateDataGen['STATION'], 'LATITUDE': climateDataGen['LATITUDE'], 'LONGITUDE': climateDataGen['LONGITUDE'], 'DATE': climateDataGen['DATE'], 'PRCP': climateDataGen['PRCP'], 'TAVG': climateDataGen['TAVG'], 'TMAX': climateDataGen['TMAX'], 'TMIN': climateDataGen['TMIN']}
cdf1 = pd.DataFrame(cDataFiltered)
cdf1 = cdf1.query('STATION == "PA000086086"').copy()
yearlyTemps = pd.DataFrame(columns = ['year', 'average'])
yearlyPrcps = pd.DataFrame(columns = ['year', 'average'])

print(cdf1)
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
    total1 = 0
    count1 = 0
    for j in range(0, cdf1['DATE'].size):
        string1 = cdf1.iloc[j]['DATE']
        if string1[:4] == str(2000 + i) and cdf1.iloc[j]['PRCP'] >= 0:
            total1 += cdf1.iloc[j]['PRCP']
            count1 += 1
    yearlyPrcps.loc[len(yearlyPrcps.index)] = [2000 + i, (total1/count1)] 

print(yearlyPrcps)




# climateDataGen = climateDataGen.loc['STATION', 'LATITUDE', 'LONGITUDE', 'DATE', 'PRCP', 'TAVG', 'TMAX', 'TMIN']
# climateDataGen = climateDataGen.query('STATION == PA000086086 & TMIN != null & TMAX != null')

messingData = pd.read_csv(r"plovercsv.csv")
messingData = messingData.loc[messingData['occurrenceStatus']=='PRESENT',['eventDate','individualCount','decimalLatitude','decimalLongitude','day','month','year']].copy()

pSandpiperRaw = pd.read_csv(r"pectoralSandpiperUnfiltered.csv")
pSandpiperRaw = pSandpiperRaw.loc[pSandpiperRaw['occurrenceStatus']=='PRESENT',['eventDate','individualCount','decimalLatitude','decimalLongitude','day','month','year']].copy()

# Changing the DataFrame to only include lat, lon, count, and year while debugging with print statements
pSandpiperHeat = pSandpiperRaw[['decimalLongitude', 'decimalLatitude', 'individualCount', 'year']]
# print("7/17", pSandpiperHeat)
pSandpiperHeat['decimalLongitude'] = pSandpiperHeat['decimalLongitude'].fillna(0)
# print("lat added", pSandpiperHeat)
pSandpiperHeat['decimalLatitude'] = pSandpiperHeat['decimalLatitude'].fillna(0)
# print("long added", pSandpiperHeat)
pSandpiperHeat['decimalLongitude'] = pSandpiperHeat['decimalLongitude'].astype(int)
# print("long turned to int", pSandpiperHeat)
pSandpiperHeat['decimalLatitude'] = pSandpiperHeat['decimalLatitude'].astype(int)
# print("lat turned to int", pSandpiperHeat)
pSandpiperHeat['individualCount'] = pSandpiperHeat['individualCount'].fillna(1)
pSandpiperHeat['individualCount'] = pSandpiperHeat['individualCount'].astype(int)
pSandpiperHeat = pSandpiperHeat.dropna(subset=['year'])
pSandpiperHeat['year'] = pSandpiperHeat['year'].astype(int)

# Modifying the dataframe so that it only includes sightings from South America to see how the migratory patterns there specifically are changing
pSandpiperHeat = pSandpiperHeat.query('0 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')
print("7/18/22", pSandpiperHeat)

a = np.zeros(shape=(180, 360))

# Debug statements
# print("a by itself", a)
# print("a from 100 to 230", a[100,230])
# print("number of rows (supposedly) from the dataframe", pSandpiperHeat.shape[0])
# print("first val", pSandpiperHeat.iloc[1242]['decimalLongitude'], pSandpiperHeat.iloc[1242]['decimalLatitude'], pSandpiperHeat.iloc[1242]['individualCount'])

for x in np.arange(0, pSandpiperHeat.shape[0]):
    # print("here", x, pSandpiperHeat.iloc[x]['decimalLongitude'], pSandpiperHeat.iloc[x]['decimalLatitude'], pSandpiperHeat.iloc[x]['individualCount'])
    a[pSandpiperHeat.iloc[x]['decimalLatitude'] + 90, pSandpiperHeat.iloc[x]['decimalLongitude'] + 180,] = a[pSandpiperHeat.iloc[x]['decimalLatitude'] + 90, pSandpiperHeat.iloc[x]['decimalLongitude'] + 180,] + pSandpiperHeat.iloc[x]['individualCount']
    print(x, "----------> ", a[pSandpiperHeat.iloc[x]['decimalLatitude'] + 90, pSandpiperHeat.iloc[x]['decimalLongitude'] + 180,])

a = a.astype(int)
print(a)
np.savetxt("foo3.csv", a, delimiter=" ")
print("11 -> ", a[95,15])
# f, ax = plt.subplots(figsize=(15, 15))
ax = sb.heatmap(a, linewidths=.5, xticklabels = 1, yticklabels = 1)
# plt.show()
plt.savefig('heatmapFirst.png')

sHawkRaw = pd.read_csv(r"swainsonHawk.csv")
sHawkRaw = sHawkRaw.loc[sHawkRaw['occurrenceStatus']=='PRESENT', ['eventDate', 'individualCount', 'decimalLatitude', 'decimalLongitude', 'day', 'month', 'year']].copy()

wSandpiperRaw = pd.read_csv(r"whiteRumpedRaw.csv")
wSandpiperRaw = wSandpiperRaw.loc[wSandpiperRaw['occurrenceStatus']=='PRESENT', ['eventDate', 'individualCount', 'decimalLatitude', 'decimalLongitude', 'day', 'month', 'year']].copy()

fTailedRaw = pd.read_csv(r"forktailedUnfiltered.csv")
fTailedRaw = fTailedRaw.loc[fTailedRaw['occurrenceStatus']=='PRESENT', ['eventDate', 'individualCount', 'decimalLatitude', 'decimalLongitude', 'day', 'month', 'year']].copy()

sScreamerRaw = pd.read_csv(r"unfilteredSouthernScreamer.csv")
sScreamerRaw = sScreamerRaw.loc[sScreamerRaw['occurrenceStatus']=='PRESENT', ['eventDate', 'individualCount', 'decimalLatitude', 'decimalLongitude', 'day', 'month', 'year']].copy()

#controlOne = messingData[ (messingData['decimalLatitude'] >= 0) & ((messingData['decimalLongitude'] <= -90) & messingData['decimalLongitude'] >= -30)].index
#controlTwo = messingData[ (messingData['decimalLatitude'] >= -59) & (messingData['decimalLongitude'] <= -61) & (messingData['decimalLongitude'] >= -29) & (messingData['decimalLongitude'] <= -31)].index
ms1, ms2, ms3, ms4, ms5 = messingData.shape[0], pSandpiperRaw.shape[0], sHawkRaw.shape[0], wSandpiperRaw.shape[0], fTailedRaw.shape[0]
input("Press Enter to continue...")
messingData, pSandpiperRaw, sHawkRaw, wSandpiperRaw, fTailedRaw = messingData.dropna(subset=['individualCount']), pSandpiperRaw.dropna(subset=['individualCount']), sHawkRaw.dropna(subset=['individualCount']), wSandpiperRaw.dropna(subset=['individualCount']), fTailedRaw.dropna(subset=['individualCount'])
print((1 - (messingData.shape[0]/ms1))*100, (1 - (pSandpiperRaw.shape[0]/ms2))*100, (1 - (sHawkRaw.shape[0]/ms3))*100, (1 - (wSandpiperRaw.shape[0]/ms4))*100, (1 - (fTailedRaw.shape[0]/ms5))*100)
input("Press Enter to continue...")

southAmericanScreamer = sScreamerRaw.query('8 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')
specificScreamer = sScreamerRaw.query('-25 >= decimalLatitude >= -35 & -55 >= decimalLongitude >= -65 & individualCount < 1000')

southAmericanPlover = messingData.query('8 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')
northAmericanPlover = messingData.query('8 <= decimalLatitude <= 50 & -120 <= decimalLongitude <= -60 & individualCount < 1000')

specificMessing = messingData.query('-25 >= decimalLatitude >= -35 & -55 >= decimalLongitude >= -65 & individualCount < 1000')
specificMessing2 = messingData.query('-26 >= decimalLatitude >= -34 & -56 >= decimalLongitude >= -64 & individualCount < 1000')
specificMessing3 = messingData.query('25 <= decimalLatitude <= 35 & -85 >= decimalLongitude >= -95 & individualCount < 1000')


southAmericanPiper = pSandpiperRaw.query('8 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')
northAmericanPiper = pSandpiperRaw.query('8 <= decimalLatitude <= 50 & -120 <= decimalLongitude <= -60 & individualCount < 1000')

specificPiper = pSandpiperRaw.query('-25 >= decimalLatitude >= -35 &  -55 >= decimalLongitude >= -65 & individualCount < 1000')
specificPiper2 = pSandpiperRaw.query('-26 >= decimalLatitude >= -33 &  -56 >= decimalLongitude >= -64 & individualCount < 1000')
specificPiper3 = pSandpiperRaw.query('25 <= decimalLatitude <= 35 & -85 >= decimalLongitude >= -95 & individualCount < 1000')


southAmericanHawk = sHawkRaw.query('8 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')
northAmericanHawk = sHawkRaw.query('8 <= decimalLatitude <= 50 & -120 <= decimalLongitude <= -60 & individualCount < 1000')

specificHawk = sHawkRaw.query('-25 >= decimalLatitude >= -35 &  -55 >= decimalLongitude >= -65 & individualCount < 1000')
specificHawk2 = sHawkRaw.query('-26 >= decimalLatitude >= -33 &  -56 >= decimalLongitude >= -64 & individualCount < 1000')
specificHawk3 = sHawkRaw.query('25 <= decimalLatitude <= 35 & -85 >= decimalLongitude >= -95 & individualCount < 1000')


southAmericanRump = wSandpiperRaw.query('8 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')
northAmericanRump = wSandpiperRaw.query('8 <= decimalLatitude <= 50 & -120 <= decimalLongitude <= -60 & individualCount < 1000')


specificRump = wSandpiperRaw.query('-25 >= decimalLatitude >= -35 & -55 >= decimalLongitude >= -65 & individualCount < 1000')
specificRump2 = wSandpiperRaw.query('-26 >= decimalLatitude >= -33 &  -57 >= decimalLongitude >= -64 & individualCount < 1000')
specificRump3 = wSandpiperRaw.query('25 <= decimalLatitude <= 35 & -85 >= decimalLongitude >= -95 & individualCount < 1000')


southAmericanForktail = fTailedRaw.query('8 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')
print(southAmericanForktail['individualCount'].sum())
print(fTailedRaw['individualCount'].sum())
exit()

specificForktail = fTailedRaw.query('-25 >= decimalLatitude >= -35 & -55 >= decimalLongitude >= -65 & individualCount < 1000')


smallCountMonthW = largeCountMonthW = smallCountMonthP  = largeCountMonthP = smallCountMonth = smallCount = largeCount = smallCountP = largeCountP = smallCountH = largeCountH = smallCountW = largeCountW = smallCountF = largeCountF = smallCountNAPlover = largeCountNAPlover = smallCountNAPiper = largeCountNAPiper = smallCountNARump = largeCountNARump = smallCountNAHawk = largeCountNAHawk = smallCountScreamer = largeCountScreamer = pd.DataFrame({'year':[], 'countS':[]})
smallCount = yearCount(smallCount, specificMessing, 2000, 2020)
largeCount = yearCount(largeCount, southAmericanPlover, 2000, 2020)
smallCountMonth = monthCount(smallCount, specificMessing, 2000, 2020)
largeCountMonth = monthCount(largeCount, southAmericanPlover, 2000, 2020)
print(smallCountMonth)

smallCountP = yearCount(smallCountP, specificPiper, 2000, 2020)
largeCountP = yearCount(largeCountP, southAmericanPiper, 2000, 2020)
smallCountMonthP = monthCount(smallCountP, specificPiper, 2000, 2020)
largeCountMonthP = monthCount(largeCountP, southAmericanPiper, 2000, 2020)

smallCountW = yearCount(smallCountW, specificRump, 2000, 2020)
largeCountW = yearCount(largeCountW, southAmericanRump, 2000, 2020)
smallCountMonthW = monthCount(smallCountW, specificRump, 2000, 2020)
largeCountMonthW = monthCount(largeCountW, southAmericanRump, 2000, 2020)

#<-------------IMPORTANT CODE FOR HISTOGRAM-------------------->
ddd = pd.DataFrame((smallCountMonth.loc[0:len(smallCountMonth)]['count']) / (largeCountMonth.loc[0:len(smallCountMonth)]['count'] + 0.0001) * 100)

ag_set_1 = pd.DataFrame((smallCountMonth.loc[0:len(smallCountMonth)]['count']) / (largeCountMonth.loc[0:len(largeCountMonth)]['count'] + 0.0001) * 100)
ps_set_1 = pd.DataFrame((smallCountMonthP.loc[0:len(smallCountMonthP)]['count']) / (largeCountMonthP.loc[0:len(largeCountMonthP)]['count'] + 0.0001) * 100)
wr_set_1 = pd.DataFrame((smallCountMonthW.loc[0:len(smallCountMonthW)]['count'])/ (largeCountMonthW.loc[0:len(largeCountMonthW)]['count'] + 0.0001) * 100)

print("IMPORTANTIMPORTANTIMPORTANTIMPORTANTIMPORTANTIMPORTANTIMPORTANTIMPORTANTIMPORTANTIMPORTANTIMPORTANTIMPORTANTIMPORTANT")
print(ag_set_1, ps_set_1, wr_set_1)
print(ddd)

histogramMaker(ddd, "Test", 'newhistotest.png')
histogramMaker(ag_set_1, "American Golden-Plover", "2DHistogram_AgbirdSet1.png")
histogramMaker(ps_set_1, "Pectoral Sandpiper", "2DHistogram_PsbirdSet1.png")
histogramMaker(wr_set_1, "White-rumped Sandpiper", "2DHistogram_WsbirdSet_1.png")


fig, ax = plt.subplots()

ln2 = ax.plot(kpIntoMonths['month'], kpIntoMonths['index'], color = 'blue')
ax.tick_params(axis='y', labelcolor = 'blue')
ax.set(ylabel = 'KpIndex')

axs = ax.twinx()

ln1 = axs.plot(np.arange(0, len(wr_set_1['count'])), wr_set_1['count'], color = 'red')
axs.set(xlabel='Month #', ylabel='Bird Spotting Percentage', title='30S60W WrBird / KpIndex Monthly')
axs.tick_params(axis='y', labelcolor = 'red')
axs.grid(color = 'grey', linestyle = '--')

lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc="upper right")
fig.tight_layout()

# kpIntoMonths
plt.savefig("TRIALKPINDEX4.png")

print("Ag Bird: " + str(stats.pearsonr(kpAvg['avInd'], ag_set_1['count'])))
print("Ps Bird: " + str(stats.pearsonr(kpAvg['avInd'], ps_set_1['count'])))
print("Ws Bird: " + str(stats.pearsonr(kpAvg['avInd'], wr_set_1['count'])))



#---------------------------



smallCountH = yearCount(smallCountH, specificHawk, 2000, 2020)
largeCountH = yearCount(largeCountH, southAmericanHawk, 2000, 2020)

smallCountF = yearCount(smallCountF, specificForktail, 2000, 2020)
largeCountF = yearCount(largeCountF, southAmericanForktail, 2000, 2020)

smallCountNAPlover = yearCount(smallCountNAPlover, specificMessing3, 2000, 2020)
largeCountNAPlover = yearCount(largeCountNAPlover, northAmericanPlover, 2000, 2020)

smallCountNAPiper = yearCount(smallCountNAPiper, specificPiper3, 2000, 2020)
largeCountNAPiper = yearCount(largeCountNAPiper, northAmericanPiper, 2000, 2020)

smallCountNARump = yearCount(smallCountNARump, specificRump3, 2000, 2020)
largeCountNARump = yearCount(largeCountNARump, northAmericanRump, 2000, 2020)

smallCountNAHawk = yearCount(smallCountNAHawk, specificHawk3, 2000,2020)
largeCountNAHawk = yearCount(largeCountNAHawk, northAmericanHawk, 2000, 2020)

smallCountScreamer = yearCount(smallCountScreamer, specificScreamer, 2000, 2020)
largeCountScreamer = yearCount(largeCountScreamer, southAmericanScreamer, 2000, 2020)


newm = magData3060.copy().query('2000 <= year < 2020')
magStrength3060S = newm['intensity'].values.tolist()

climate3060S = yearlyTemps['average'].values.tolist()
prcp3060S = yearlyPrcps['average'].values.tolist()

otherm = magData30N90W.copy().query('2000 <= year < 2020')
magStrength30N90W = otherm['intensity'].values.tolist()


print(matplotlib.get_cachedir())


#correlation tests
#these are the dataframes needed for the all birds correlation


# x1 = np.arange(1, 13)
# y1 = (smallCountMonth.loc[0:11]['count'] / (largeCountMonth.loc[0:11]['count'] + 0.01)) * 100
# fig, ax = plt.subplots()
# plotter(0,20,ax, "")
# #ax.plot(x1, y1)
# ax.set(xlabel='Month #', ylabel='Number of Birds Spotted',
#        title='30S60W AgBird Monthly Plot')
# ax.grid()
# ax.legend(loc="upper left")
# fig.savefig("30S60WAgBirdMonthly.png")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('Five Year Intervals American Golden Plover')
plotter(smallCountMonth, largeCountMonth, 0,6,ax1, "2000-2005")
plotter(smallCountMonth, largeCountMonth, 6,11,ax2, "2006-2010")
plotter(smallCountMonth, largeCountMonth, 11,16,ax3, "2011-2015")
plotter(smallCountMonth, largeCountMonth, 16,20,ax4, "2016-2020")
for ax in fig.get_axes():
    ax.label_outer()
fig.savefig("AgfiveDecadesPopulationDensityChanges.png")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('Five Year Intervals Pectoral Sandpiper')
plotter(smallCountMonthP, largeCountMonthP, 0, 6, ax1, "2000-2005")
plotter(smallCountMonthP, largeCountMonthP, 6, 11, ax2, "2006-2010")
plotter(smallCountMonthP, largeCountMonthP, 11, 16, ax3, "2011-2015")
plotter(smallCountMonthP, largeCountMonthP, 16, 20, ax4, "2016-2020")
for ax in fig.get_axes():
    ax.label_outer()
fig.savefig("PsfiveDecadesPopulationDensityChanges.png")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('Five Year Intervals White-rumped Sandpiper')
plotter(smallCountMonthW, largeCountMonthW, 0, 6, ax1, "2000-2005")
plotter(smallCountMonthW, largeCountMonthW, 6, 11, ax2, "2006-2010")
plotter(smallCountMonthW, largeCountMonthW, 11, 16, ax3, "2011-2015")
plotter(smallCountMonthW, largeCountMonthW, 16, 20, ax4, "2016-2020")
for ax in fig.get_axes():
    ax.label_outer()
fig.savefig("WrfiveDecadesPopulationDensityChanges.png")

# ax1.plot(x1, y1)
# ax2.plot(x1, (smallCountMonth.loc[12:23]['count'] / (largeCountMonth.loc[12:23]['count'] + 0.01)) * 100, 'tab:orange')
# ax3.plot(x1, (smallCountMonth.loc[24:35]['count'] / (largeCountMonth.loc[24:35]['count'] + 0.01)) * 100, 'tab:green')
# ax4.plot(x1, (smallCountMonth.loc[36:47]['count'] / (largeCountMonth.loc[36:47]['count'] + 0.01)) * 100, 'tab:red')


fig, ax5 = plt.subplots()
ax5.set_title("Daily Average Temperatures in Paraguay from 2000-2020")
ax5.set_xlabel("Day Number")
ax5.set_ylabel("Temperature (Farenheit)")
ax5.plot(np.arange(cdf1['TAVG'].size), cdf1['TAVG'])
fig.savefig("tteettt")



print("|------------------------------------------------------------------------|")#section start
smallCountPercentages = (smallCount['countS'].copy() / largeCount['countS'].copy()) * 100
smallCountWPercentages = (smallCountW['countS'].copy() / largeCountW['countS'].copy()) * 100
smallCountPPercentages = (smallCountP['countS'].copy() / largeCountP['countS'].copy()) * 100
smallCountHPercentages = (smallCountH['countS'].copy() / largeCountH['countS'].copy()) * 100
allBirdPercentages = smallCountPercentages.multiply(0.25) + smallCountWPercentages.multiply(0.25) + smallCountPPercentages.multiply(0.25) + smallCountHPercentages.multiply(0.25)

print("> Magnetic 30S60W All Birds: " + "r value = " + str(stats.pearsonr(magStrength3060S,allBirdPercentages)[0]) + "; p value = " + str(stats.pearsonr(magStrength3060S, allBirdPercentages)[1]))
print("> Temperature 30S60W All Birds: " + "r value = " + str(stats.pearsonr(climate3060S,allBirdPercentages)[0]) + "; p value = " + str(stats.pearsonr(climate3060S, allBirdPercentages)[1]))
print("> Precipitation 30S60W All Birds: " + "r value = " + str(stats.pearsonr(prcp3060S,allBirdPercentages)[0]) + "; p value = " + str(stats.pearsonr(prcp3060S, allBirdPercentages)[1]))

# fvalue, pvalue = stats.f_oneway(climate3060S, prcp3060S, allBirdPercentages)
# fvalue2, pvalue2 = stats.f_oneway(magStrength3060S, allBirdPercentages)
l1 = [1,2,4,8,16,32,64]
l2 = [1000,2000,4000,8000,16000,32000,64000]
f, p = stats.f_oneway(l1,l2)
print(str(f) + " and " + str(p))

# print(str(fvalue) + " and " + str(pvalue))
# print(str(fvalue2) + " and " + str(pvalue2))

scatterPlot(magStrength3060S, allBirdPercentages, "30S60W All Birds Magnetic Strength", "30S60WAbBird.png", "Magnetic Strength (nT)", "darkslategrey")
doubleYAxisPlotMaker(2000,2020, magStrength3060S, allBirdPercentages, "5R Analysis of All Birds 30S60W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "DOUBLEAXIS_AbBird30S60W_M.png", "tab:red", "darkslategrey")

scatterPlot(climate3060S, allBirdPercentages, "30S60W All Birds Temperature", "30S60WAbBird_T.png", "Temperature (F)", "darkslategrey")
doubleYAxisPlotMaker(2000,2020, climate3060S, allBirdPercentages, "5R Analysis of All Birds 30S60W from 2000-2020(T)", "Temperature (F)", "Population Density %", "DOUBLEAXIS_AbBird30S60W_T.png", "tab:red", "darkslategrey")

scatterPlot(prcp3060S, allBirdPercentages, "30S60W All Birds Precipitation", "30S60WAbBird_P.png", "Precipitation (in)", "darkslategrey")
doubleYAxisPlotMaker(2000,2020, prcp3060S, allBirdPercentages, "5R Analysis of All Birds 30S60W from 2000-2020(P)", "Precipitation (in)", "Population Density %", "DOUBLEAXIS_AbBird30S60W_P.png", "tab:red", "darkslategrey")

#(yearMin, yearMax, data1, data2, title, dataLabel1, dataLabel2, filename, color1, color2):
#5R Analysis of American Golden-Plover 30S60W from 2000-2020



# a, b = np.polyfit(magStrength3060S, allBirdPercentages, deg = 1)
# aBest = a * magStrength3060S + allBirdPercentages
# aBerr = magStrength3060S.std() * np.sqrt(1/len(magStrength3060S) + (magStrength3060S - magStrength3060S.mean())**2 / np.sum((magStrength3060S - magStrength3060S.mean())**2))

# fig, ax = plt.subplots()
# ax.plot(magStrength3060S, aBest, '-')
# ax.fill_between(magStrength3060S, aBest - aBerr, aBest + aBerr, alpha = 0.2)
# ax.plot(magStrength3060S, allBirdPercentages, 'o', color='tab:blue')

ploverSmallList3060S = smallCount.copy()['countS'].values.tolist()
ploverLargeList3060S = largeCount.copy()['countS'].values.tolist()
print(ploverSmallList3060S)
percentagesPlover3060S = []
for x in range(len(ploverSmallList3060S)):
    percentagesPlover3060S.append((ploverSmallList3060S[x]/ploverLargeList3060S[x]) * 100)
print("> Magnetic 30S60W American Golden-Plover: " + "r value = " + str(stats.pearsonr(magStrength3060S,percentagesPlover3060S)[0]) + "; p value = " + str(stats.pearsonr(magStrength3060S, percentagesPlover3060S)[1]))
print("> Temperature 30S60W American Golden-Plover: " + "r value = " + str(stats.pearsonr(climate3060S,percentagesPlover3060S)[0]) + "; p value = " + str(stats.pearsonr(climate3060S, percentagesPlover3060S)[1]))
print("> Precipitation 30S60W American Golden-Plover: " + "r value = " + str(stats.pearsonr(prcp3060S,percentagesPlover3060S)[0]) + "; p value = " + str(stats.pearsonr(prcp3060S, percentagesPlover3060S)[1]))

scatterPlot(climate3060S, percentagesPlover3060S, "30S60W American Golden-Plover Temperature", "30S60WAgBird_T.png", "Temperature (F)", "tab:blue")
doubleYAxisPlotMaker(2000,2020, climate3060S, percentagesPlover3060S, "5R Analysis of American Golden-Plover 30S60W from 2000-2020(T)", "Temperature (F)", "Population Density %", "DOUBLEAXIS_AgBird30S60W_T.png", "tab:red", "tab:blue")
doubleYAxisPlotMaker(2000,2020, climate3060S, percentagesPlover3060S, "5R Analysis of American Golden-Plover 30S60W from 2000-2020(T)", "Temperature (F)", "Population Density %", "STS_DOUBLEAXIS_AgBird30S60W_T.png", "tab:red", "tab:blue")

scatterPlot(magStrength3060S, percentagesPlover3060S, "30S60W American Golden-Plover Magnetic Strength", "30S60WAgBird.png", "Magnetic Strength (nT)", "tab:blue")
doubleYAxisPlotMaker(2000,2020, magStrength3060S, percentagesPlover3060S, "5R Analysis of American Golden-Plover 30S60W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "DOUBLEAXIS_AgBird30S60W_M.png", "tab:red", "tab:blue")
doubleYAxisPlotMaker(2000,2020, magStrength3060S, percentagesPlover3060S, "5R Analysis of American Golden-Plover 30S60W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "WESEF_DOUBLEAXIS_AgBird30S60W_M.png", "tab:red", "tab:blue")
doubleYAxisPlotMaker(2000,2020, magStrength3060S, percentagesPlover3060S, "5R Analysis of American Golden-Plover 30S60W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "STS_DOUBLEAXIS_AgBird30S60W_M.png", "tab:red", "tab:blue")

scatterPlot(prcp3060S, percentagesPlover3060S, "30S60W American Golden-Plover Precipitation", "30S60WAgBird_P.png", "Precipitation (in)", "tab:blue")
doubleYAxisPlotMaker(2000,2020, prcp3060S, percentagesPlover3060S, "5R Analysis of American Golden-Plover 30S60W from 2000-2020(P)", "Precipitation (in)", "Population Density %", "DOUBLEAXIS_AgBird30S60W_P.png", "tab:red", "tab:blue")
doubleYAxisPlotMaker(2000,2020, prcp3060S, percentagesPlover3060S, "5R Analysis of American Golden-Plover 30S60W from 2000-2020(P)", "Precipitation (in)", "Population Density %", "STS_DOUBLEAXIS_AgBird30S60W_P.png", "tab:red", "tab:blue")

#White-rumped Sandpiper South America
whiteSmallList3060S = smallCountW.copy()['countS'].values.tolist()
whiteLargeList3060S = largeCountW.copy()['countS'].values.tolist()
percentagesWhite3060S = []
for x in range(len(whiteSmallList3060S)):
    percentagesWhite3060S.append((whiteSmallList3060S[x]/whiteLargeList3060S[x]) * 100)
print("> Magnetic 30S60W White-rumped Sandpiper: " + "r value = " + str(stats.pearsonr(magStrength3060S,percentagesWhite3060S)[0]) + "; p value = " + str(stats.pearsonr(magStrength3060S, percentagesWhite3060S)[1]))
print("> Temperature 30S60W White-rumped Sandpiper: " + "r value = " + str(stats.pearsonr(climate3060S,percentagesWhite3060S)[0]) + "; p value = " + str(stats.pearsonr(climate3060S, percentagesWhite3060S)[1]))
print("> Precipitation 30S60W White-rumped Sandpiper: " + "r value = " + str(stats.pearsonr(prcp3060S,percentagesWhite3060S)[0]) + "; p value = " + str(stats.pearsonr(prcp3060S, percentagesWhite3060S)[1]))
scatterPlot(magStrength3060S, percentagesWhite3060S, "30S60W White-rumped Sandpiper Magnetic Strength", "30S60WWrBird_M.png", "Magnetic Strength (nT)", "tab:green")
doubleYAxisPlotMaker(2000,2020, magStrength3060S, percentagesWhite3060S, "5R Analysis of White-rumped Sandpiper 30S60W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "DOUBLEAXIS_WrBird30S60W_M.png", "tab:red", "tab:blue")
doubleYAxisPlotMaker(2000,2020, magStrength3060S, percentagesWhite3060S, "5R Analysis of White-rumped Sandpiper 30S60W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "WESEF_DOUBLEAXIS_WrBird30S60W_M.png", "tab:red", "tab:green")
doubleYAxisPlotMaker(2000,2020, magStrength3060S, percentagesWhite3060S, "5R Analysis of White-rumped Sandpiper 30S60W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "STS_DOUBLEAXIS_WrBird30S60W_M.png", "tab:red", "tab:green")

scatterPlot(climate3060S, percentagesWhite3060S, "30S60W White-rumped Sandpiper Temperature", "30S60WWrBird_T.png", "Temperature (F)", "tab:green")
doubleYAxisPlotMaker(2000,2020, climate3060S, percentagesWhite3060S, "5R Analysis of White-rumped Sandpiper 30S60W from 2000-2020(T)", "Temperature (F)", "Population Density %", "DOUBLEAXIS_WrBird30S60W_T.png", "tab:red", "tab:blue")
doubleYAxisPlotMaker(2000,2020, prcp3060S, percentagesWhite3060S, "5R Analysis of White-rumped Sandpiper 30S60W from 2000-2020(P)", "Precipitation (in)", "Population Density %", "STS_DOUBLEAXIS_WrBird30S60W_T.png", "tab:red", "tab:green")

scatterPlot(prcp3060S, percentagesWhite3060S, "30S60W White-rumped Sandpiper Precipitation", "30S60WWrBird_P.png", "Precipitation (in)", "tab:green")
doubleYAxisPlotMaker(2000,2020, prcp3060S, percentagesWhite3060S, "5R Analysis of White-rumped Sandpiper 30S60W from 2000-2020(P)", "Precipitation (in)", "Population Density %", "DOUBLEAXIS_WrBird30S60W_P.png", "tab:red", "tab:blue")
doubleYAxisPlotMaker(2000,2020, prcp3060S, percentagesWhite3060S, "5R Analysis of White-rumped Sandpiper 30S60W from 2000-2020(P)", "Precipitation (in)", "Population Density %", "STS_DOUBLEAXIS_WrBird30S60W_P.png", "tab:red", "tab:green")


#Pectoral Sandpiper South America
pectoralSmallList3060S = smallCountP.copy()['countS'].values.tolist()
pectoralLargeList3060S = largeCountP.copy()['countS'].values.tolist()
percentagesPectoral3060S = []
for x in range(len(pectoralSmallList3060S)):
    percentagesPectoral3060S.append((pectoralSmallList3060S[x]/pectoralLargeList3060S[x]) * 100)
print("> Magnetic 30S60W Pectoral Sandpiper: " + "r value = " + str(stats.pearsonr(magStrength3060S,percentagesPectoral3060S)[0]) + "; p value = " + str(stats.pearsonr(magStrength3060S, percentagesPectoral3060S)[1]))
print("> Temperature 30S60W Pectoral Sandpiper: " + "r value = " + str(stats.pearsonr(climate3060S,percentagesPectoral3060S)[0]) + "; p value = " + str(stats.pearsonr(climate3060S, percentagesPectoral3060S)[1]))
print("> Precipitation 30S60W Pectoral Sandpiper: " + "r value = " + str(stats.pearsonr(prcp3060S,percentagesPectoral3060S)[0]) + "; p value = " + str(stats.pearsonr(prcp3060S, percentagesPectoral3060S)[1]))

scatterPlot(magStrength3060S, percentagesPectoral3060S, "30S60W Pectoral Sandpiper Magnetic Strength", "30S60WPsBird_M.png", "Magnetic Strength (nT)", "tab:purple")
doubleYAxisPlotMaker(2000,2020, magStrength3060S, percentagesPectoral3060S, "5R Analysis of Pectoral Sandpiper 30S60W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "DOUBLEAXIS_PsBird30S60W_M.png", "tab:red", "tab:blue")
doubleYAxisPlotMaker(2000,2020, magStrength3060S, percentagesPectoral3060S, "5R Analysis of Pectoral Sandpiper 30S60W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "WESEF_DOUBLEAXIS_PsBird30S60W_M.png", "tab:red", "tab:purple")
doubleYAxisPlotMaker(2000,2020, magStrength3060S, percentagesPectoral3060S, "5R Analysis of Pectoral Sandpiper 30S60W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "STS_DOUBLEAXIS_PsBird30S60W_M.png", "tab:red", "tab:purple")

scatterPlot(climate3060S, percentagesPectoral3060S, "30S60W Pectoral Sandpiper Temperature", "30S60WPsBird_T.png", "Temperature (F)", "tab:purple")
doubleYAxisPlotMaker(2000,2020, climate3060S, percentagesPectoral3060S, "5R Analysis of Pectoral Sandpiper 30S60W from 2000-2020(T)", "Temperature (F)", "Population Density %", "DOUBLEAXIS_PsBird30S60W_T.png", "tab:red", "tab:blue")
doubleYAxisPlotMaker(2000,2020, climate3060S, percentagesPectoral3060S, "5R Analysis of Pectoral Sandpiper 30S60W from 2000-2020(T)", "Temperature (F)", "Population Density %", "STS_DOUBLEAXIS_PsBird30S60W_T.png", "tab:red", "tab:purple")

scatterPlot(prcp3060S, percentagesPectoral3060S, "30S60W Pectoral Sandpiper Precipitation", "30S60WPsBird_P.png", "Precipitation (in)", "tab:purple")
doubleYAxisPlotMaker(2000,2020, prcp3060S, percentagesPectoral3060S, "5R Analysis of Pectoral Sandpiper 30S60W from 2000-2020(P)", "Precipitation (in)", "Population Density %", "DOUBLEAXIS_PsBird30S60W_P.png", "tab:red", "tab:blue")
doubleYAxisPlotMaker(2000,2020, prcp3060S, percentagesPectoral3060S, "5R Analysis of Pectoral Sandpiper 30S60W from 2000-2020(P)", "Precipitation (in)", "Population Density %", "STS_DOUBLEAXIS_PsBird30S60W_P.png", "tab:red", "tab:purple")



#Swainson's Hawk South America
hawkSmallList3060S = smallCountH.copy()['countS'].values.tolist()
hawkLargeList3060S = largeCountH.copy()['countS'].values.tolist()
percentagesHawk3060S = []
for x in range(len(hawkSmallList3060S)):
    percentagesHawk3060S.append((hawkSmallList3060S[x]/hawkLargeList3060S[x]) * 100)
print("> Magnetic 30S60W Swainson's Hawk: " + "r value = " + str(stats.pearsonr(magStrength3060S,percentagesHawk3060S)[0]) + "; p value = " + str(stats.pearsonr(magStrength3060S, percentagesHawk3060S)[1]))
print("> Temperature 30S60W Swainson's Hawk: " + "r value = " + str(stats.pearsonr(climate3060S,percentagesHawk3060S)[0]) + "; p value = " + str(stats.pearsonr(climate3060S, percentagesHawk3060S)[1]))
print("> Precipitation 30S60W Swainson's Hawk: " + "r value = " + str(stats.pearsonr(prcp3060S,percentagesHawk3060S)[0]) + "; p value = " + str(stats.pearsonr(prcp3060S, percentagesHawk3060S)[1]))

scatterPlot(magStrength3060S, percentagesHawk3060S, "30S60W Swainson's Hawk Magnetic Strength", "30S60WShBird_M.png", "Magnetic Strength (nT)", "rosybrown")
doubleYAxisPlotMaker(2000,2020, magStrength3060S, percentagesHawk3060S, "5R Analysis of Swainson's Hawk 30S60W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "DOUBLEAXIS_ShBird30S60W_M.png", "tab:red", "tab:blue")
doubleYAxisPlotMaker(2000,2020, magStrength3060S, percentagesHawk3060S, "5R Analysis of Swainson's Hawk 30S60W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "WESEF_DOUBLEAXIS_ShBird30S60W_M.png", "tab:red", "rosybrown")
doubleYAxisPlotMaker(2000,2020, magStrength3060S, percentagesHawk3060S, "5R Analysis of Swainson's Hawk 30S60W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "STS_DOUBLEAXIS_ShBird30S60W_M.png", "tab:red", "rosybrown")

scatterPlot(climate3060S, percentagesHawk3060S, "30S60W Swainson's Hawk Temperature", "30S60WShBird_T.png", "Temperature (F)", "rosybrown")
doubleYAxisPlotMaker(2000,2020, climate3060S, percentagesHawk3060S, "5R Analysis of Swainson's Hawk 30S60W from 2000-2020(T)", "Temperature (F)", "Population Density %", "DOUBLEAXIS_ShBird30S60W_T.png", "tab:red", "tab:blue")
doubleYAxisPlotMaker(2000,2020, climate3060S, percentagesHawk3060S, "5R Analysis of Swainson's Hawk 30S60W from 2000-2020(T)", "Temperature (F)", "Population Density %", "STS_DOUBLEAXIS_ShBird30S60W_T.png", "tab:red", "rosybrown")

scatterPlot(prcp3060S, percentagesHawk3060S, "30S60W Swainson's Hawk Precipitation", "30S60WShBird_P.png", "Precipitation (in)", "rosybrown")
doubleYAxisPlotMaker(2000,2020, prcp3060S, percentagesHawk3060S, "5R Analysis of Swainson's Hawk 30S60W from 2000-2020(P)", "Precipitation (in)", "Population Density %", "DOUBLEAXIS_ShBird30S60W_P.png", "tab:red", "tab:blue")
doubleYAxisPlotMaker(2000,2020, prcp3060S, percentagesHawk3060S, "5R Analysis of Swainson's Hawk 30S60W from 2000-2020(P)", "Precipitation (in)", "Population Density %", "STS_DOUBLEAXIS_ShBird30S60W_P.png", "tab:red", "rosybrown")



#Fork-tailed Flycatcher South America
forkSmallList3060S = smallCountF.copy()['countS'].values.tolist()
forkLargeList3060S = largeCountF.copy()['countS'].values.tolist()
percentagesFork3060S = []
for x in range(len(forkSmallList3060S)):
    percentagesFork3060S.append((forkSmallList3060S[x]/forkLargeList3060S[x]) * 100)
print("> Magnetic 30S60W Fork-tailed Flycatcher: " + "r value = " + str(stats.pearsonr(magStrength3060S,percentagesFork3060S)[0]) + "; p value = " + str(stats.pearsonr(magStrength3060S, percentagesFork3060S)[1]))
print("> Temperature 30S60W Fork-tailed Flycatcher: " + "r value = " + str(stats.pearsonr(climate3060S,percentagesFork3060S)[0]) + "; p value = " + str(stats.pearsonr(climate3060S, percentagesFork3060S)[1]))
print("> Precipitation 30S60W Fork-tailed Flycatcher: " + "r value = " + str(stats.pearsonr(prcp3060S,percentagesFork3060S)[0]) + "; p value = " + str(stats.pearsonr(prcp3060S, percentagesFork3060S)[1]))

scatterPlot(magStrength3060S, percentagesFork3060S, "30S60W Fork-tailed Flycatcher Magnetic Strength", "30S60WFtBird_M.png", "Magnetic Strength (nT)", "tab:orange")
doubleYAxisPlotMaker(2000,2020, magStrength3060S, percentagesFork3060S, "5R Analysis of Fork-tailed Flycatcher 30S60W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "DOUBLEAXIS_FtBird30S60W_M.png", "tab:red", "tab:blue")
doubleYAxisPlotMaker(2000,2020, magStrength3060S, percentagesFork3060S, "5R Analysis of Fork-tailed Flycatcher 30S60W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "WESEF_DOUBLEAXIS_FtBird30S60W_M.png", "tab:red", "tab:orange")
doubleYAxisPlotMaker(2000,2020, magStrength3060S, percentagesFork3060S, "5R Analysis of Fork-tailed Flycatcher 30S60W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "STS_DOUBLEAXIS_FtBird30S60W_M.png", "tab:red", "darkorange")


scatterPlot(climate3060S, percentagesFork3060S, "30S60W Fork-tailed Flycatcher Temperature", "30S60WFtBird_T.png", "Temperature (F)", "tab:orange")
doubleYAxisPlotMaker(2000,2020, climate3060S, percentagesFork3060S, "5R Analysis of Fork-tailed Flycatcher 30S60W from 2000-2020(T)", "Temperature (F)", "Population Density %", "DOUBLEAXIS_FtBird30S60W_T.png", "tab:red", "tab:blue")
doubleYAxisPlotMaker(2000,2020, climate3060S, percentagesFork3060S, "5R Analysis of Fork-tailed Flycatcher 30S60W from 2000-2020(T)", "Temperature (F)", "Population Density %", "STS_DOUBLEAXIS_FtBird30S60W_T.png", "tab:red", "tab:orange")


scatterPlot(prcp3060S, percentagesFork3060S, "30S60W Fork-tailed Flycatcher Precipitation", "30S60WFtBird_P.png", "Precipitation (in)", "tab:orange")
doubleYAxisPlotMaker(2000,2020, prcp3060S, percentagesFork3060S, "5R Analysis of Fork-tailed Flycatcher 30S60W from 2000-2020(P)", "Precipitation (in)", "Population Density %", "DOUBLEAXIS_FtBird30S60W_P.png", "tab:red", "tab:blue")
doubleYAxisPlotMaker(2000,2020, prcp3060S, percentagesFork3060S, "5R Analysis of Fork-tailed Flycatcher 30S60W from 2000-2020(P)", "Precipitation (in)", "Population Density %", "STS_DOUBLEAXIS_FtBird30S60W_P.png", "tab:red", "tab:orange")




#<--------------------------------------WIP RN-------------------------------------------->
# toto = magData3060.copy().query('year >= 2005 & year < 2020')
# nMMNn = toto['intensity'].values.tolist()

#Southern Screamer South America
screamSmallList3060S = smallCountScreamer.copy()['countS'].values.tolist()
screamLargeList3060S = largeCountScreamer.copy()['countS'].values.tolist()
percentagesScream3060S = []
for x in range(len(screamSmallList3060S)):
    percentagesScream3060S.append((screamSmallList3060S[x]/screamLargeList3060S[x]) * 100)
print("> 30S60W Southern Screamer: " + "r value = " + str(stats.pearsonr(magStrength3060S,percentagesScream3060S)[0]) + "; p value = " + str(stats.pearsonr(magStrength3060S, percentagesScream3060S)[1]))
scatterPlot(magStrength3060S, percentagesScream3060S, "30S60W Southern Screamer Magnetic Strength", "30S60WSsBird.png", "Magnetic Strength (nT)", "tab:blue")


#American Golden-Plover Correlation North America
NAPloverSmallList3060S = smallCountNAPlover.copy()['countS'].values.tolist()
NAPloverLargeList3060S = largeCountNAPlover.copy()['countS'].values.tolist()
percentagesNAPlover30N90W = []
for x in range(len(NAPloverSmallList3060S)):
    percentagesNAPlover30N90W.append((NAPloverSmallList3060S[x]/NAPloverLargeList3060S[x]) * 100)
print("> 30N90W American Golden-Plover: " + "r value = " + str(stats.pearsonr(magStrength30N90W,percentagesNAPlover30N90W)[0]) + "; p value = " + str(stats.pearsonr(magStrength30N90W, percentagesNAPlover30N90W)[1]))
scatterPlot(magStrength30N90W, percentagesNAPlover30N90W, "30N90W American Golden-Plover Magnetic Strength", "30N90WAgBird.png", "Magnetic Strength (nT)", "tab:blue")
doubleYAxisPlotMaker(2000,2020, magStrength30N90W, percentagesNAPlover30N90W, "5R Analysis of American Golden-Plover 30N90W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "DOUBLEAXIS_AgBird30N90W_M.png", "tab:red", "tab:blue")

#White-rumped Sandpiper Correlation North America
NAWhiteSmallList3060S = smallCountNARump.copy()['countS'].values.tolist()
NAWhiteLargeList3060S = largeCountNARump.copy()['countS'].values.tolist()
percentagesNAWhite30N90W = []
for x in range(len(NAWhiteSmallList3060S)):
    percentagesNAWhite30N90W.append((NAWhiteSmallList3060S[x]/NAWhiteLargeList3060S[x]) * 100)
print("> 30N90W White-rumped Sandpiper: " + "r value = " + str(stats.pearsonr(magStrength30N90W,percentagesNAWhite30N90W)[0]) + "; p value = " + str(stats.pearsonr(magStrength30N90W, percentagesNAWhite30N90W)[1]))
scatterPlot(magStrength30N90W, percentagesNAWhite30N90W, "30N90W White-rumped Sandpiper Magnetic Strength", "30N90WWrBird.png", "Magnetic Strength (nT)", "tab:green")
doubleYAxisPlotMaker(2000,2020, magStrength30N90W, percentagesNAWhite30N90W, "5R Analysis of White-rumped Sandpiper 30N90W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "DOUBLEAXIS_WrBird30N90W_M.png", "tab:red", "tab:blue")

#Pectoral Sandpiper Correlation North America
NAPiperSmallList3060S = smallCountNAPiper.copy()['countS'].values.tolist()
NAPiperLargeList3060S = largeCountNAPiper.copy()['countS'].values.tolist()
percentagesNAPiper30N90W = []
for x in range(len(NAPiperSmallList3060S)):
    percentagesNAPiper30N90W.append((NAPiperSmallList3060S[x]/NAPiperLargeList3060S[x]) * 100)
print("> 30N90W Pectoral Sandpiper: " + "r value = " + str(stats.pearsonr(magStrength30N90W,percentagesNAPiper30N90W)[0]) + "; p value = " + str(stats.pearsonr(magStrength30N90W, percentagesNAPiper30N90W)[1]))
scatterPlot(magStrength30N90W, percentagesNAPiper30N90W, "30N90W Pectoral Sandpiper Magnetic Strength", "30N90WPsBird.png", "Magnetic Strength (nT)", "tab:purple")
doubleYAxisPlotMaker(2000,2020, magStrength30N90W, percentagesNAPiper30N90W, "5R Analysis of Pectoral Sandpiper 30N90W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "DOUBLEAXIS_PsBird30N90W_M.png", "tab:red", "tab:blue")

#Swainson's Hawk Correlation North America
NAHawkSmallList3060S = smallCountNAHawk.copy()['countS'].values.tolist()
NAHawkLargeList3060S = largeCountNAHawk.copy()['countS'].values.tolist()
percentagesNAHawk30N90W = []
for x in range(len(NAHawkSmallList3060S)):
    percentagesNAHawk30N90W.append((NAHawkSmallList3060S[x]/NAHawkLargeList3060S[x]) * 100)
print("> 30N90W Swainson's Hawk: " + "r value = " + str(stats.pearsonr(magStrength30N90W,percentagesNAHawk30N90W)[0]) + "; p value = " + str(stats.pearsonr(magStrength30N90W, percentagesNAHawk30N90W)[1]))
scatterPlot(magStrength30N90W, percentagesNAHawk30N90W, "30N90W Swainson's Hawk Magnetic Strength", "30N90WShBird.png", "Magnetic Strength (nT)", "rosybrown")
doubleYAxisPlotMaker(2000,2020, magStrength30N90W, percentagesNAHawk30N90W, "5R Analysis of Swainson's Hawk 30N90W from 2000-2020(M)", "Magnetic Strength (nT)", "Population Density %", "DOUBLEAXIS_ShBird30N90W_M.png", "tab:red", "tab:blue")


print("|------------------------------------------------------------------------|")#section end
#Custom double Y-axis plot for WESEF:
t = np.arange(2000,2020, 1)
fig, ax1 = plt.subplots()

ax1.set_xlabel('Year')
ax1.set_ylabel("Magnetic Strength (nT)", color = "tab:red")
ax1.set_title("5R Analysis of Migratory Birds 30S60W from 2000-2020(M)")
ln1 = ax1.plot(t, magStrength3060S, color = "tab:red", label = "Magnetic Strength (nT)")
ax1.tick_params(axis='y', labelcolor = "tab:red")
ax1.grid(color = 'grey', linestyle = "--")
# ax1.legend(loc="upper right")

ax2 = ax1.twinx()  

ax2.set_ylabel("Population Density %", color = "tab:blue") 
ln2 = ax2.plot(t, percentagesPlover3060S, color = "tab:cyan", label = "AgBird Population Density %")
ln3 = ax2.plot(t, percentagesWhite3060S, color = "tab:green", label = "WrBird Population Density %")
ln4 = ax2.plot(t, percentagesPectoral3060S, color = "tab:purple", label = "PsBird Population Density%")
ln5 = ax2.plot(t, percentagesHawk3060S, color = "tab:orange", label = "ShBird Population Density %")
ax2.set_ylim([0,100])
ax2.tick_params(axis='y', labelcolor = "tab:blue")
# ax2.legend(loc="upper right", bbox_to_anchor=(1, 0.90))

#GOD TIER LEGEND HELPER GOD BLESS
lns = ln1 + ln2 + ln3 + ln4 + ln5
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="upper right")

fig.tight_layout()  # otherwise the right y-label is slightly clipped...
fig.savefig("WESEF_Graph_Beta.png")
print("WESEF_Graph_Beta" + " successfully uploaded.")


#---------------------------#

# hie_data = sm.datasets.randhie.load_pandas()
# corr_matrix = np.corrcoef(hie_data.data.T)
# smg.plot_corr(corr_matrix, xnames=hie_data.names)
# plt.show()

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#testing with for loops
# smallCount = largeCount = smallCountP = largeCountP = smallCountH = largeCountH = smallCountW = largeCountW = smallCountF = largeCountF = pd.DataFrame({'year':[], 'countS':[]})
# listOfCounts = [[smallCount, specificMessing], [largeCount, southAmericanPlover], [smallCountP, specificPiper], [largeCountP, southAmericanPiper], [smallCountH, specificHawk], [largeCountH, southAmericanHawk], [smallCountW, specificRump], [largeCountW, southAmericanRump], [smallCountF, specificForktail], [largeCountF, southAmericanForktail]]
# for x in listOfCounts:
#     print("1")
#     print(x[0])
#     for i in np.arange(2000,2020):
#         x[0] = x[0].append(pd.DataFrame({'year': i, 'countS': x[1].query('year == @i')['individualCount'].sum()}, index = [0]), ignore_index = True)
#     print(x[0])
#     print("2")
# print(smallCount)
# print("Success.............")
# print(smallCount.year);print(3)

#-----------------------------------------experimental group--------------------------------------------------
EsmallCount = pd.DataFrame({'year':[],'countS':[]})
for i in np.arange(1970, 2020):
    EsmallCount = EsmallCount.append(pd.DataFrame({'year': i, 'countS': specificMessing.query('year == @i')['individualCount'].sum()}, index=[0]), ignore_index=True)

ElargeCount = pd.DataFrame({'year':[], 'countS':[]})
for i in np.arange(1970,2020):
    ElargeCount = ElargeCount.append(pd.DataFrame({'year':i, 'countS': southAmericanPlover.query('year == @i')['individualCount'].sum()}, index = [0]), ignore_index=True)

#piper loop
EsmallCountP = pd.DataFrame({'year':[],'countS':[]})
for i in np.arange(1970, 2020):
    EsmallCountP = EsmallCountP.append(pd.DataFrame({'year': i, 'countS': specificPiper.query('year == @i')['individualCount'].sum()}, index=[0]), ignore_index=True)

ElargeCountP = pd.DataFrame({'year':[], 'countS':[]})
for i in np.arange(1970,2020):
    ElargeCountP = ElargeCountP.append(pd.DataFrame({'year':i, 'countS': southAmericanPiper.query('year == @i')['individualCount'].sum()}, index = [0]), ignore_index=True)

#hawk loop
EsmallCountH = pd.DataFrame({'year':[],'countS':[]})
for i in np.arange(1970, 2020):
    EsmallCountH = EsmallCountH.append(pd.DataFrame({'year': i, 'countS': specificHawk.query('year == @i')['individualCount'].sum()}, index=[0]), ignore_index=True)

ElargeCountH = pd.DataFrame({'year':[], 'countS':[]})
for i in np.arange(1970,2020):
    ElargeCountH = ElargeCountH.append(pd.DataFrame({'year':i, 'countS': southAmericanHawk.query('year == @i')['individualCount'].sum()}, index = [0]), ignore_index=True)

#whiterumped loop
EsmallCountW = pd.DataFrame({'year':[], 'countS':[]})
for i in np.arange(1970,2020):
    EsmallCountW = EsmallCountW.append(pd.DataFrame({'year': i, 'countS': specificRump.query('year == @i')['individualCount'].sum()},  index = [0]), ignore_index = True)

ElargeCountW = pd.DataFrame({'year':[],'countS':[]})
for i in np.arange(1970,2020):
    ElargeCountW = ElargeCountW.append(pd.DataFrame({'year': i, 'countS': southAmericanRump.query('year == @i')['individualCount'].sum()}, index = [0]), ignore_index = True)
#-------------------------------------------------------------------------------------------------------------------------
QQsmallCount = QQlargeCount = QQsmallCountP = QQlargeCountP = QQsmallCountH = QQlargeCountH = QQsmallCountW = QQlargeCountW = QQsmallCountF = QQlargeCountF = pd.DataFrame({'year':[], 'countS':[]})
QQsmallCount = yearCount(QQsmallCount, specificMessing2, 2000, 2020)
QQlargeCount = yearCount(QQlargeCount, southAmericanPlover, 2000, 2020)

QQsmallCountP = yearCount(QQsmallCountP, specificPiper2, 2000, 2020)
QQlargeCountP = yearCount(QQlargeCountP, southAmericanPiper, 2000, 2020)

QQsmallCountH = yearCount(QQsmallCountH, specificHawk2, 2000, 2020)
QQlargeCountH = yearCount(QQlargeCountH, southAmericanHawk, 2000, 2020)

QQsmallCountW = yearCount(QQsmallCountW, specificRump2, 2000, 2020)
QQlargeCountW = yearCount(QQlargeCountW, southAmericanRump, 2000, 2020)

#new double y axis plot tester
t = np.arange(2000,2020, 1)
data1 = magStrength3060S
data2 = percentagesPlover3060S

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('Magnetic Strength (nT)', color=color)
ax1.set_title('5R Analysis of American Golden-Plover 30S60W from 2000-2020')
ln1 = ax1.plot(t, data1, color=color, label = "Annual Magnetic Strength (nT)")
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(color = 'grey', linestyle = "--")
# ax1.legend(loc="upper right")

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Population Density (%)', color=color)  # we already handled the x-label with ax1
ln2 = ax2.plot(t, data2, color=color, label = "Annual AgBird Population Density %")
ax2.tick_params(axis='y', labelcolor=color)
# ax2.legend(loc="upper right", bbox_to_anchor=(1, 0.90))

#GOD TIER LEGEND HELPER GOD BLESS
monthList = np.array(["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sep", "Oct", "Nov", "Dec"])
seriesN = pd.Series(monthList)

lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="upper right")

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.savefig("newDoubleY-axisPlot.png")

lis = lis2 = lis3 = list((np.arange(0,240) / 12) + 2000)
lis = lis2 = lis3 = [int(x) for x in lis]

ag_set_1['year'] = lis
ag_set_1['month'] = pd.concat([seriesN]*20, ignore_index = True)
nm = pd.Series(np.array(np.arange(0,12)))
ag_set_1['numerical_months'] = pd.concat([nm] * 20, ignore_index = True)
years = list(np.arange(2000,2020))
print(ag_set_1)

ps_set_1['year'] = lis2 
ps_set_1['month'] = pd.concat([seriesN]*20, ignore_index = True)

wr_set_1['year'] = lis3
wr_set_1['month'] = pd.concat([seriesN]*20, ignore_index = True)


sb.set_theme(style="white")
pal = sb.cubehelix_palette(10, rot=-.25, light=.7)
g = sb.FacetGrid(ag_set_1, row = "year", aspect = 9, height = 1.2, palette = pal)
g.map_dataframe(sb.lineplot, x="month", y = 'count', alpha=1, linewidth=2)
g.savefig("ridgeAg.png")

sb.set_theme(style="white")
pal = sb.cubehelix_palette(10, rot=-.25, light=.7)
g = sb.FacetGrid(wr_set_1, row = "year", aspect = 9, height = 1.2, palette = pal)
g.map_dataframe(sb.lineplot, x="month", y = 'count', alpha=1, linewidth=2)
g.savefig("ridgeWr.png")

sb.set_theme(style="white")
pal = sb.cubehelix_palette(10, rot=-.25, light=.7)
g = sb.FacetGrid(ps_set_1, row = "year", aspect = 9, height = 1.2, palette = pal)
g.map_dataframe(sb.lineplot, x="month", y = 'count', alpha=1, linewidth=2)
g.savefig("ridgePs.png")

g.savefig("rIDGE1.png")

betaKpMonth = kpIntoMonths
betaKpMonth.reset_index(drop=True, inplace=True)
firstMonths = kpIntoMonths.query('0 <= month <= 12')
print(firstMonths['month'])
betaKpMonth['months'] = pd.concat([firstMonths['month']] * 20, ignore_index = True)

print(betaKpMonth)
for x in np.arange(0, len(betaKpMonth['index'])):
    if betaKpMonth['index'][x] < 7:
        betaKpMonth['index'][x] = 0
print(betaKpMonth)

sb.set_theme(style="white")
pal = sb.cubehelix_palette(10, rot=-.25, light=.7)
g = sb.FacetGrid(betaKpMonth, row = "year", aspect = 9, height = 1.2, palette = pal)
g.map_dataframe(sb.lineplot, x="months", y = 'index', alpha=1, linewidth=2)
g.savefig("RIDGE14.png")

newset = betaKpMonth
newset['count'] = ag_set_1['count']
newset['month2'] = ag_set_1['numerical_months']

print(newset)

# def twin_lineplot(x,y,color,**kwargs):
#     ax = plt.twinx()
#     sb.lineplot(x=x,y=y,color=color,**kwargs, ax=ax)
# sb.set_theme(style = "white")
# pal = sb.cubehelix_palette(10, rot = -0.25, light = 0.7)
# #g = sb.FacetGrid(betaKpMonth, sharex = True row = "year", aspect = 9, height = 1.2, palette = pal)
# g = sb.FacetGrid(newset, row="year")
# g.map(sb.lineplot, 'month', 'index', color='b')
# g.map(sb.lineplot, 'month2', 'count', color='g')

# g.savefig("testRIDGE.png")


#updated plots
#psbird monthly plots
sb.set_theme(style="dark")
g = sb.relplot(
    data = ps_set_1,
    x = "month", y = "count", col = "year", hue = "year",
    kind = "line", palette = "crest", linewidth = 4, zorder = 5,
    col_wrap = 4, height = 2, aspect = 1.5, legend = False,
)

for year, ax1 in g.axes_dict.items():

    ax1.text(.8, .85, year, transform = ax1.transAxes, fontweight="bold")

    sb.lineplot(
        data = ps_set_1, x = "month", y = "count", units="year",
        estimator = None, color = ".7", linewidth = 1, ax = ax1,
    )

ax1.set_xticks(ax1.get_xticks()[::2])

g.set_titles("")
g.set_axis_labels("", "Percent Present")
g.tight_layout()
g.savefig("ps_bird_monthly_plots.png")


#AgBird monthly
sb.set_theme(style="dark")
g = sb.relplot(
    data = ag_set_1,
    x = "month", y = "count", col = "year", hue = "year",
    kind = "line", palette = "crest", linewidth = 4, zorder = 5,
    col_wrap = 4, height = 2, aspect = 1.5, legend = False,
)

for year, ax1 in g.axes_dict.items():

    ax1.text(.8, .85, year, transform = ax1.transAxes, fontweight="bold")

    sb.lineplot(
        data = ag_set_1, x = "month", y = "count", units="year",
        estimator = None, color = ".7", linewidth = 1, ax = ax1,
    )

ax1.set_xticks(ax1.get_xticks()[::2])

g.set_titles("")
g.set_axis_labels("", "Percent Present")
g.tight_layout()
g.savefig("ag_bird_monthly_plots.png")


#wrbird monthly
sb.set_theme(style="dark")
g = sb.relplot(
    data = wr_set_1,
    x = "month", y = "count", col = "year", hue = "year",
    kind = "line", palette = "crest", linewidth = 4, zorder = 5,
    col_wrap = 4, height = 2, aspect = 1.5, legend = False,
)

for year, ax1 in g.axes_dict.items():

    ax1.text(.8, .85, year, transform = ax1.transAxes, fontweight="bold")

    sb.lineplot(
        data = wr_set_1, x = "month", y = "count", units="year",
        estimator = None, color = ".7", linewidth = 1, ax = ax1,
    )

ax1.set_xticks(ax1.get_xticks()[::2])

g.set_titles("")
g.set_axis_labels("", "Percent Present")
g.tight_layout()
g.savefig("wr_bird_monthly_plots.png")

    # ax2 = ax1.twinx()

    # sb.lineplot(
    #     data = newset, x = "month", y = "index", units="year",
    #     estimator = None, color = ".7", linewidth = 1, ax = ax2,
    # )

print(newset)

#kp index monthly
sb.set_theme(style="dark")
g = sb.relplot(
    data = newset,
    x = "months", y = "index", col = "year", hue = "year",
    kind = "line", palette = "rocket", linewidth = 4, zorder = 5,
    col_wrap = 4, height = 2, aspect = 1.5, legend = False,
)

for year, ax1 in g.axes_dict.items():

    ax1.text(.8, .85, year, transform = ax1.transAxes, fontweight="bold")

    sb.lineplot(
        data = newset, x = "months", y = "index", units="year",
        estimator = None, color = ".7", linewidth = 1, ax = ax1,
    )

ax1.set_xticks(ax1.get_xticks()[::1])

g.set_titles("")
g.set_axis_labels("", "KP Index")
g.tight_layout()
g.savefig("kp_index_monthly_plots.png")






# sb.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# # Create the data
# #rs = np.random.RandomState(1979)
# x = ag_set_1['count']
# g = np.tile(list(np.arange(2000,2020)), 12)
# df = pd.DataFrame(dict(x=x, g=g))
# m = df.g.map(ord)
# df["x"] += m

# # Initialize the FacetGrid object
# pal = sb.cubehelix_palette(10, rot=-.25, light=.7)
# g = sb.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)

# # Draw the densities in a few steps
# g.map(sb.kdeplot, "x",
#       bw_adjust=.5, clip_on=False,
#       fill=True, alpha=1, linewidth=1.5)
# g.map(sb.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)

# # passing color=None to refline() uses the hue mapping
# g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


# # Define and use a simple function to label the plot in axes coordinates
# def label(x, color, label):
#     ax = plt.gca()
#     ax.text(0, .2, label, fontweight="bold", color=color,
#             ha="left", va="center", transform=ax.transAxes)

# g.map(label, "x")

# # Set the subplots to overlap
# g.figure.subplots_adjust(hspace=-.25)

# # Remove axes details that don't play well with overlap
# g.set_titles("")
# g.set(yticks=[], ylabel="")
# g.despine(bottom=True, left=True)

# g.savefig("EE.png")



#printing
# print(smallCount)
# print(largeCount)
# print(smallCountP)
# print(largeCountP)
# print(smallCountH)
# print(largeCountH)

#First testing with filtering (filters out for all of South America)

#controlLocation = birdData[ (birdData['decimalLatitude'] >= 0) & ((birdData['decimalLongitude'] <= -90) & birdData['decimalLongitude'] >= -30)].index
#birdData.drop(controlLocation, inplace = True)

#This filters out only for Argentenia
'''
newControlLocation = birdData[ (birdData['countryCode'] != 'AR')].index
birdData.drop(newControlLocation, inplace = True)
'''

#preciseBirdData = birdData.loc[birdData['occurrenceStatus']=='PRESENT',['eventDate','individualCount','decimalLatitude','decimalLongitude','day','month','year']].copy()
#birdData = pd.DataFrame()


preciseBirdData = birdData
subsetOne = preciseBirdData
#print(subsetOne)
#oldRegion
region = [subsetOne.decimalLongitude.min() - 1, subsetOne.decimalLongitude.max() + 1, subsetOne.decimalLatitude.min() - 1, subsetOne.decimalLatitude.max() + 1]
#print(region)
#print(subsetOne.head())

finalRegion = [-90,177,-55,1]
n232n = [-90, -30, -55, 1]
worldRegion = [-170,180,-60,80]
AmericanRegion = [-170,0,-60,80]

#1970-1979
seventies = preciseBirdData.query('1970 <= year < 1980')
seventiesFig = pygmt.Figure()
seventiesFig.basemap(region=finalRegion,projection="M8i",frame=["a", '+t"American Golden-Plover (AgBird) - South America (1970-1980)"'])
seventiesFig.coast(land="burlywood", water="lightblue")
pygmt.makecpt(cmap="viridis", series=[1970,1979])
seventiesFig.plot(
    x=seventies.decimalLongitude,
    y=seventies.decimalLatitude,
    color = seventies.year,
    cmap= True,
    size= 0.05 + ((seventies.individualCount + 1)/10000),
    style= "cc",
    pen = "black"
)
seventiesFig.colorbar(frame='af+l"Year"')
seventiesFig.savefig("AgBird_seventies.png",show = False)
print("American Golden-Plover (AgBird) - South America (1970-1980) Successfully Updated.")

#1980-1989
eighties = preciseBirdData.query('1980 <= year < 1990')
eightiesFig = pygmt.Figure()
eightiesFig.basemap(region=finalRegion,projection="M8i",frame=["a", '+t"American Golden-Plover (AgBird) - South America (1970-1980)"'])
eightiesFig.coast(land="burlywood", water="lightblue")
pygmt.makecpt(cmap="viridis", series=[1980,1989])
eightiesFig.plot(
    x=eighties.decimalLongitude,
    y=eighties.decimalLatitude,
    color = eighties.year,
    cmap= True,
    size= 0.05 + ((eighties.individualCount + 1)/10000),
    style= "cc",
    pen = "black"
)
eightiesFig.colorbar(frame='af+l"Year"')
eightiesFig.savefig("AgBird_eighties.png",show = False)
print("American Golden-Plover (AgBird) - South America (1970-1980) Successfully Updated.")

#1990-1999
nineties = preciseBirdData.query('1990 <= year < 2000')
ninetiesFig = pygmt.Figure()
ninetiesFig.basemap(region=finalRegion,projection="M8i",frame=["a", '+t"American Golden-Plover (AgBird) - South America (1990-2000)"'])
ninetiesFig.coast(land="burlywood", water="lightblue")
pygmt.makecpt(cmap="viridis", series=[1990,1999])
ninetiesFig.plot(
    x=nineties.decimalLongitude,
    y=nineties.decimalLatitude,
    color = nineties.year,
    cmap= True,
    size= 0.05 + ((nineties.individualCount + 1)/10000),
    style= "cc",
    pen = "black"
)
ninetiesFig.colorbar(frame='af+l"Year"')
ninetiesFig.savefig("AgBird_nineties.png",show = False)
print("American Golden-Plover (AgBird) - South America (1990-2000) Successfully Updated.")

#2000-2009
twozero = preciseBirdData.query('2000 <= year < 2010')
twozeroFig = pygmt.Figure()
twozeroFig.basemap(region=finalRegion,projection="M8i",frame=["a", '+t"American Golden-Plover (AgBird) - South America (2000-2010)"'])
twozeroFig.coast(land="burlywood", water="lightblue")
pygmt.makecpt(cmap="viridis", series=[2000,2009])
twozeroFig.plot(
    x=twozero.decimalLongitude,
    y=twozero.decimalLatitude,
    color = twozero.year,
    cmap= True,
    size= 0.05 + ((twozero.individualCount + 1)/10000),
    style= "cc",
    pen = "black"
)
twozeroFig.colorbar(frame='af+l"Year"')
twozeroFig.savefig("AgBird_twozero.png",show = False)
print("American Golden-Plover (AgBird) - South America (2000-2010) Successfully Updated.")

#2010-Present
twoten = preciseBirdData.query('2010 <= year')
twotenFig = pygmt.Figure()
twotenFig.basemap(region=finalRegion,projection="M8i",frame=["a", '+t"American Golden-Plover (AgBird) - South America (2010-Present)"'])
twotenFig.coast(land="burlywood", water="lightblue")
pygmt.makecpt(cmap="viridis", series=[2010,2021])
twotenFig.plot(
    x=twoten.decimalLongitude,
    y=twoten.decimalLatitude,
    color = twoten.year,
    cmap= True,
    size= 0.05 + ((twoten.individualCount + 1)/10000),
    style= "cc",
    pen = "black"
)
twotenFig.colorbar(frame='af+l"Year"')
twotenFig.savefig("AgBird_twoten.png",show = False)
print("American Golden-Plover (AgBird) - South America (2010-Present) Successfully Updated.")

#seventies.to_csv('seventies.csv')
#eighties.to_csv('eighties.csv')
#nineties.to_csv('nineties.csv')
#twozero.to_csv('twozero.csv')
#twoten.to_csv('twoten.csv')

#Everything
fig = pygmt.Figure()
fig.basemap(region=finalRegion, projection="M8i", frame=["a", '+t"American Golden-Plover (AgBird) - South America"'])
fig.coast(land="burlywood", water="lightblue")
pygmt.makecpt(cmap="viridis", series=[subsetOne.year.min(), subsetOne.year.max()])
fig.plot(
    x=subsetOne.decimalLongitude, 
    y=subsetOne.decimalLatitude, 
    color = subsetOne.year,
    cmap=True,
    size=0.05 + ((subsetOne.individualCount + 1)/10000),
    style="cc",
    pen="black"
)
fig.colorbar(frame='af+l"Year "')
fig.savefig("AgBird_seventies_to_twenties.png",show = False)
print("American Golden-Plover (AgBird) - South America Successfully Updated.")

xList = []
yList = []
length = len(subsetOne)

#PectoralSandpipers Everything
newDf = pd.read_csv(r"pectoralSandpiperFiltered.csv", encoding='latin1')
#newDf = newDf.query("2000 >= year >= 1970").copy()
pSand = pygmt.Figure()
pSand.basemap(region=finalRegion, projection="M8i", frame=["a", '+t"Pectoral Sandpiper (PsBird) - South America"'])
pSand.coast(land="burlywood", water="lightblue")
pygmt.makecpt(cmap="plasma", series=[newDf.year.min(), newDf.year.max()])
pSand.plot(
    x=newDf.decimalLongitude, 
    y=newDf.decimalLatitude, 
    color = newDf.year,
    cmap=True,
    size=0.05 + ((newDf.individualCount + 1)/10000),
    style="cc",
    pen="black"
)
pSand.colorbar(frame='af+l"Year "')
pSand.savefig("PsBird_seventies_to_twenties.png",show = False)
print("Pectoral Sandpiper (PsBird) - South America Successfully Updated.")

#swainsonHawk
hawkDf = pd.read_csv(r"swainsonHawkFiltered.csv")
#hawkDf = hawkDf.query("2000 >= year >= 1970").copy()
sHawk = pygmt.Figure()
sHawk.basemap(region=finalRegion, projection="M8i", frame=["a", '+t"Swainsons Hawk (ShBird) - South America"'])
sHawk.coast(land="burlywood", water="lightblue")
pygmt.makecpt(cmap="inferno", series=[hawkDf.year.min(), hawkDf.year.max()])
sHawk.plot(
    x=hawkDf.decimalLongitude, 
    y=hawkDf.decimalLatitude, 
    color = hawkDf.year,
    cmap=True,
    size=0.05 + ((hawkDf.individualCount + 1)/10000),
    style="cc",
    pen="black"
)
sHawk.colorbar(frame='af+l"Year "')
sHawk.savefig("ShBird_seventies_to_twenties.png",show = False)
print("Swainsons Hawk (ShBird) - South America Successfully Updated.")


#White Rumped Pectoral
wRump = pd.read_csv(r"whiteRumpedFiltered.csv")
#wRump = wRump.query("2000 >= year >= 1970").copy()
wrp = pygmt.Figure()
wrp.basemap(region=finalRegion, projection="M8i", frame=["a", '+t"Pectoral White-Rumped Sandpiper (WrBird) - South America"'])
wrp.coast(land="burlywood", water="lightblue")
pygmt.makecpt(cmap="plasma", series=[wRump.year.min(), wRump.year.max()])
wrp.plot(
    x=wRump.decimalLongitude, 
    y=wRump.decimalLatitude, 
    color = wRump.year,
    cmap=True,
    size=0.05 + ((wRump.individualCount + 1)/10000),
    style="cc",
    pen="black"
)
wrp.colorbar(frame='af+l"Year "')
wrp.savefig("WrBird_seventies_to_twenties.png",show = False)
print("Pectoral White-Rumped Sandpiper (WrBird) - South America Successfully Updated.")

#forktailed flycatcher (control, native south american bird)
fTailed = pd.read_csv(r"forktailedFiltered.csv")
ftd = pygmt.Figure()
ftd.basemap(region = worldRegion, projection = "M8i", frame=["a", '+t"Fork-tailed Flycatcher [Control] (FtBird) - South America"'])
ftd.coast(land="burlywood", water="lightblue")
pygmt.makecpt(cmap="plasma", series=[fTailed.year.min(), fTailed.year.max()])
ftd.plot(
    x=fTailed.decimalLongitude, 
    y=fTailed.decimalLatitude, 
    color = fTailed.year,
    cmap=True,
    size=0.05 + ((fTailed.individualCount + 1)/10000),
    style="cc",
    pen="black"
)
ftd.colorbar(frame='af+l"Year "')
ftd.savefig("FtBird_seventies_to_twenties.png",show = False)
print("Fork-tailed Flycatcher [Control] (FtBird) - South America Successfully Updated.")


#shawkworld
sH = pd.read_csv(r"swainsonHawk.csv")
sH = sH.loc[sH['occurrenceStatus']=='PRESENT',['eventDate','individualCount','decimalLatitude','decimalLongitude','day','month','year']].copy()
sH = sH.query("year >= 1970")
sH = sH.query("individualCount <= 10000")
HawkFull = pygmt.Figure()
HawkFull.basemap(region=worldRegion, projection="M8i", frame=["a", '+t"Swainsons Hawk (ShBird) - World"'])
HawkFull.coast(land="burlywood", water="lightblue", resolution="f")
pygmt.makecpt(cmap="plasma", series=[sH.year.min(), sH.year.max()])
HawkFull.plot(
    x=sH.decimalLongitude, 
    y=sH.decimalLatitude, 
    color = sH.year,
    cmap=True,
    size=0.05 + ((sH.individualCount + 1)/10000),
    style="cc",
    pen="black"
)
HawkFull.colorbar(frame='af+l"Year "')
HawkFull.savefig("ShBird_Full_Map.png",show = False)
print("Swainsons Hawk (ShBird) - World Successfully Updated.")

#fullWorldOfPlover
tB = pd.read_csv(r"plovercsv.csv")
tB = tB.loc[tB['occurrenceStatus']=='PRESENT',['eventDate','individualCount','decimalLatitude','decimalLongitude','day','month','year']].copy()
tB = tB.query("year >= 1970")
tB = tB.query("individualCount <= 10000")
ploverFull = pygmt.Figure()
ploverFull.basemap(region=worldRegion, projection="M8i", frame=["a", '+t"American Golden-Plover (AgBird) - World"'])
ploverFull.coast(land="burlywood", water="lightblue", resolution="f")
pygmt.makecpt(cmap="inferno", series=[tB.year.min(), tB.year.max()])
ploverFull.plot(
    x=tB.decimalLongitude, 
    y=tB.decimalLatitude, 
    color = tB.year,
    cmap=True,
    size=0.05 + ((tB.individualCount + 1)/10000),
    style="cc",
    pen="black"
)
ploverFull.colorbar(frame='af+l"Year "')
ploverFull.savefig("AgBird_Full_Map.png",show = False)
print("American Golden-Plover (AgBird) - World Successfully Updated.")


#fullWorldOfSandpiper
pW = pd.read_csv(r"pectoralSandpiperUnfiltered.csv")
pW = pW.loc[pW['occurrenceStatus']=='PRESENT',['eventDate','individualCount','decimalLatitude','decimalLongitude','day','month','year']].copy()
pW = pW.query("year >= 1970")
pW = pW.query("individualCount <= 10000")
sandFull = pygmt.Figure()
sandFull.basemap(region=worldRegion, projection="M8i", frame=["a", '+t"Pectoral Sandpiper (PsBird)- World"'])
sandFull.coast(land="burlywood", water="lightblue", resolution="f")
pygmt.makecpt(cmap="plasma", series=[pW.year.min(), pW.year.max()])
sandFull.plot(
    x=pW.decimalLongitude, 
    y=pW.decimalLatitude, 
    color = pW.year,
    cmap=True,
    size=0.05 + ((pW.individualCount + 1)/10000),
    style="cc",
    pen="black"
)
sandFull.colorbar(frame='af+l"Year "')
sandFull.savefig("PsBird_Full_Map.png",show = False)
print("Pectoral Sandpiper (PsBird) - World Successfully Updated.")

#Full World Of White-Rumped Sandpiper
wW = pd.read_csv(r"whiteRumpedRaw.csv")
wW = wW.loc[wW['occurrenceStatus']=='PRESENT',['eventDate','individualCount','decimalLatitude','decimalLongitude','day','month','year']].copy()
wW = wW.query("year >= 1970")
wW = wW.query("individualCount <= 10000")
whiteFull = pygmt.Figure()
whiteFull.basemap(region=worldRegion, projection="M8i", frame=["a", '+t"White-Rumped Sandpiper (WrBird) - World"'])
whiteFull.coast(land="burlywood", water="lightblue", resolution="f")
pygmt.makecpt(cmap="plasma", series=[pW.year.min(), pW.year.max()])
whiteFull.plot(
    x=wW.decimalLongitude, 
    y=wW.decimalLatitude, 
    color = wW.year,
    cmap=True,
    size=0.05 + ((wW.individualCount + 1)/10000),
    style="cc",
    pen="black"
)
whiteFull.colorbar(frame='af+l"Year "')
whiteFull.savefig("WrBird_Full_Map.png",show = False)
print("White-Rumped Sandpiper (WrBird) - World Successfully Updated.")

#-----------------AMERICAS FOR PPT:
#Americas Swainson's Hawk
sH = pd.read_csv(r"swainsonHawk.csv")
sH = sH.loc[sH['occurrenceStatus']=='PRESENT',['eventDate','individualCount','decimalLatitude','decimalLongitude','day','month','year']].copy()
sH = sH.query("year >= 1970")
sH = sH.query("individualCount <= 10000")
HawkFull = pygmt.Figure()
HawkFull.basemap(region=AmericanRegion, projection="M8i", frame=["a", '+t"Swainsons Hawk (ShBird) - Americas"'])
HawkFull.coast(land="burlywood", water="lightblue", resolution="f")
pygmt.makecpt(cmap="plasma", series=[sH.year.min(), sH.year.max()])
HawkFull.plot(
    x=sH.decimalLongitude, 
    y=sH.decimalLatitude, 
    color = sH.year,
    cmap=True,
    size=0.05 + ((sH.individualCount + 1)/10000),
    style="cc",
    pen="black"
)
HawkFull.colorbar(frame='af+l"Year "')
HawkFull.savefig("ShBird_America_Map.png",show = False)
print("Swainsons Hawk (ShBird) - Americas Successfully Updated.")

#Americas American Golden-PLover
tB = pd.read_csv(r"plovercsv.csv")
tB = tB.loc[tB['occurrenceStatus']=='PRESENT',['eventDate','individualCount','decimalLatitude','decimalLongitude','day','month','year']].copy()
tB = tB.query("year >= 1970")
tB = tB.query("individualCount <= 10000")
ploverFull = pygmt.Figure()
ploverFull.basemap(region=AmericanRegion, projection="M8i", frame=["a", '+t"American Golden-Plover (AgBird) - Americas"'])
ploverFull.coast(land="burlywood", water="lightblue", resolution="f")
pygmt.makecpt(cmap="inferno", series=[tB.year.min(), tB.year.max()])
ploverFull.plot(
    x=tB.decimalLongitude, 
    y=tB.decimalLatitude, 
    color = tB.year,
    cmap=True,
    size=0.05 + ((tB.individualCount + 1)/10000),
    style="cc",
    pen="black"
)
ploverFull.colorbar(frame='af+l"Year "')
ploverFull.savefig("AgBird_America_Map.png",show = False)
print("American Golden-Plover (AgBird) - Americas Successfully Updated.")


#Americas Pectoral Sandpiper
pW = pd.read_csv(r"pectoralSandpiperUnfiltered.csv")
pW = pW.loc[pW['occurrenceStatus']=='PRESENT',['eventDate','individualCount','decimalLatitude','decimalLongitude','day','month','year']].copy()
pW = pW.query("year >= 1970")
pW = pW.query("individualCount <= 10000")
sandFull = pygmt.Figure()
sandFull.basemap(region=AmericanRegion, projection="M8i", frame=["a", '+t"Pectoral Sandpiper (PsBird) - Americas"'])
sandFull.coast(land="burlywood", water="lightblue", resolution="f")
pygmt.makecpt(cmap="plasma", series=[pW.year.min(), pW.year.max()])
sandFull.plot(
    x=pW.decimalLongitude, 
    y=pW.decimalLatitude, 
    color = pW.year,
    cmap=True,
    size=0.05 + ((pW.individualCount + 1)/10000),
    style="cc",
    pen="black"
)
sandFull.colorbar(frame='af+l"Year "')
sandFull.savefig("PsBird_America_Map.png",show = False)
print("Pectoral Sandpiper (PsBird) - Americas Successfully Updated.")

#Americas White-rumped Sandpiper
wW = pd.read_csv(r"whiteRumpedRaw.csv")
wW = wW.loc[wW['occurrenceStatus']=='PRESENT',['eventDate','individualCount','decimalLatitude','decimalLongitude','day','month','year']].copy()
wW = wW.query("year >= 1970")
wW = wW.query("individualCount <= 10000")
whiteFull = pygmt.Figure()
whiteFull.basemap(region=AmericanRegion, projection="M8i", frame=["a", '+t"White-Rumped Sandpiper (WrBird) - Americas"'])
whiteFull.coast(land="burlywood", water="lightblue", resolution="f")
pygmt.makecpt(cmap="plasma", series=[pW.year.min(), pW.year.max()])
whiteFull.plot(
    x=wW.decimalLongitude, 
    y=wW.decimalLatitude, 
    color = wW.year,
    cmap=True,
    size=0.05 + ((wW.individualCount + 1)/10000),
    style="cc",
    pen="black"
)
whiteFull.colorbar(frame='af+l"Year "')
whiteFull.savefig("WrBird_America_Map.png",show = False)
print("White-Rumped Sandpiper (WrBird) - Americas Successfully Updated.")
#---------------------------------------------



#allBirdsSA
abSA = pygmt.Figure()
abSA.basemap(region=finalRegion, projection="M8i", frame=["a", '+t"AgBird, Psbird and ShBird - South America"'])
abSA.coast(land="burlywood", water="lightblue", resolution="f")
pygmt.makecpt(cmap="inferno", series=[subsetOne.year.min(), subsetOne.year.max()])
abSA.plot(
    x=subsetOne.decimalLongitude, 
    y=subsetOne.decimalLatitude, 
    color = subsetOne.year,
    cmap=True,
    size=0.05 + ((subsetOne.individualCount + 1)/10000),
    style="cc",
    pen="0.01p,black"
)
abSA.plot(
    x=hawkDf.decimalLongitude, 
    y=hawkDf.decimalLatitude, 
    color = hawkDf.year,
    cmap=True,
    size=0.05 + ((hawkDf.individualCount + 1)/10000),
    style="sc",
    pen="0.01p,black"
)
abSA.plot(
    x=newDf.decimalLongitude, 
    y=newDf.decimalLatitude, 
    color = newDf.year,
    cmap=True,
    size=0.05 + ((newDf.individualCount + 1)/10000),
    style="tc",
    pen="0.01p,black"
)

abSA.colorbar(frame='af+l"Year "')
abSA.savefig("AllBirds_South_America.png",show = False)
print("AgBird, Psbird and ShBird - South America Successfully Updated.")


#Debugging
#print(datetime)

for i in range(length):
    xList.append(datetime.date(subsetOne.iloc[i].year, subsetOne.iloc[i].month, subsetOne.iloc[i].day))
    yList.append(subsetOne.iloc[i].individualCount)

#debugging -> print(xList)

populationPlot = pygmt.Figure()
populationPlot.plot(
    projection="X10c/5c",
    #First val is xMin, Second val is xMax, Third val is yMin, and Fourth Value is yMax (REGION)
    region=[datetime.date(subsetOne.year.min(), 1, 1), datetime.date(subsetOne.year.max(), 12, 31), 0, 100],
    frame=["WSne", 'xaf+l"Years (2000-2020)"', 'yaf+l"Number of Occurences Per Spotting"'],
    x=xList,
    y=yList,
    style="c0.2c",
    pen="1p",
    color="lightred"

)
populationPlot.savefig("AgBird_Population_Plot.png",show = False)
print("AgBird Population Plot Successfully Updated.")


#print(testdf)
#print(testData)

#print(birdData)
#print(birdData['individualCount'].sum())


graph = pygmt.Figure()

graph.basemap(region=[2010,2020,22400,22700], projection="X15c/15c", frame = ["St", "xaf+lYear"])

with pygmt.config(
    MAP_FRAME_PEN = "black",
    MAP_TICK_PEN = "black",
    FONT_ANNOT_PRIMARY = "black",
    FONT_LABEL = "black",
):
    graph.basemap(frame=["W", 'yaf+l"Magnetic Strength(nT)"'])

graph.plot(x = magneticData37.year, y = magneticData37.totalChange, pen="1p,black")
graph.plot(x = magneticData37.year, y = magneticData37.totalChange, style = "s0.2c", color = "black", label = 'yaf+l"Magnetic Strength (nT)"')
graph.savefig("magneticChanges.png",show = False)
print("Magnetic Changes Test Graph Successfully Updated.")

#plover sa correlation 2000-2020
correlation = pygmt.Figure()
correlation.basemap(region=[2000,2020,22000,26000], projection = "X15c/15c", frame = ["St", "xaf+lYear"])
correlation.basemap(frame=["a", '+t"5R Analysis of AgBird % In South America (30S 60W)"'])
with pygmt.config(
    MAP_FRAME_PEN = "blue",
    MAP_TICK_PEN = "blue", 
    FONT_ANNOT_PRIMARY = "blue", 
    FONT_LABEL = "blue",
):
    correlation.basemap(frame=["W", 'yaf+l"Magnetic Strength (nT)"'])
    
correlation.plot(x = magData3060.year, y = magData3060.intensity, pen="1p,blue")
correlation.plot(x = magData3060.year, y = magData3060.intensity, style="c0.2c", color = "blue", label = '"Magnetic Strength (nT)"')

with pygmt.config(
    MAP_FRAME_PEN = "red",
    MAP_TICK_PEN = "red",
    FONT_ANNOT_PRIMARY = "red",
    FONT_LABEL = "red",
):
    correlation.basemap(region = [2000,2020,0,100], frame=["E", 'yaf+l"AgBird Population Density %"'])
    
#original algorithm -> (((specificMessing[specificMessing.year == magData3060.year].sum()['individualCount']) / (southAmericanPlover[southAmericanPlover.year == magData3060.year].sum()['individualCount'])) * 100)    
# print(southAmericanPlover.query('year == 2019')['individualCount'].sum())
# print(specificMessing.query('year == 2019')['individualCount'].sum())
correlation.plot(x = smallCount.year, y = (smallCount.countS/largeCount.countS)*100, pen = "1p,red")
correlation.plot(x = smallCount.year, y = (smallCount.countS/largeCount.countS)*100, style = "s0.25c", color = "red", label = '"AgBird Population Density %"')

correlation.legend(position = "jTL+o0.1c", box = True)
correlation.savefig("AgCorrelation.png",show = False)
print("5R Analysis of AgBird % In South America (30S 60W) Successfully Updated.")

#new correlation test thing


#plover na correlation 2000-2020
correlationNA = pygmt.Figure()
correlationNA.basemap(region=[2000,2020,46000,50000], projection = "X15c/15c", frame = ["St", "xaf+lYear"])
correlationNA.basemap(frame=["a", '+t"5R Analysis of AgBird % In North America (30N 90W)"'])
with pygmt.config(
    MAP_FRAME_PEN = "blue",
    MAP_TICK_PEN = "blue", 
    FONT_ANNOT_PRIMARY = "blue", 
    FONT_LABEL = "blue",
):
    correlationNA.basemap(frame=["W", 'yaf+l"Magnetic Strength (nT)"'])
    
correlationNA.plot(x = magData30N90W.year, y = magData30N90W.intensity, pen="1p,blue")
correlationNA.plot(x = magData30N90W.year, y = magData30N90W.intensity, style="c0.2c", color = "blue", label = '"Magnetic Strength (nT)"')

with pygmt.config(
    MAP_FRAME_PEN = "red",
    MAP_TICK_PEN = "red",
    FONT_ANNOT_PRIMARY = "red",
    FONT_LABEL = "red",
):
    correlationNA.basemap(region = [2000,2020,0,50], frame=["E", 'yaf+l"AgBird Population Density %"'])
    
correlationNA.plot(x = smallCountNAPlover.year, y = (smallCountNAPlover.countS/largeCountNAPlover.countS)*100, pen = "1p,red")
correlationNA.plot(x = smallCountNAPlover.year, y = (smallCountNAPlover.countS/largeCountNAPlover.countS)*100, style = "s0.25c", color = "red", label = '"AgBird Population Density %"')

correlationNA.legend(position = "jTL+o0.1c", box = True)
correlationNA.savefig("AgCorrelation30N90W.png",show = False)
print("5R Analysis of AgBird % In North America (30N 90W) Successfully Updated.")


#piper na correlation (30N 90W) 2000-2020
pscorrelationNA = pygmt.Figure()
pscorrelationNA.basemap(region=[2000,2020,46000,50000], projection = "X15c/15c", frame = ["St", "xaf+lYear"])
pscorrelationNA.basemap(frame=["a", '+t"5R Analysis of PsBird % In North America (30N 90W)"'])
with pygmt.config(
    MAP_FRAME_PEN = "blue",
    MAP_TICK_PEN = "blue", 
    FONT_ANNOT_PRIMARY = "blue", 
    FONT_LABEL = "blue",
):
    pscorrelationNA.basemap(frame=["W", 'yaf+l"Magnetic Strength (nT)"'])
    
pscorrelationNA.plot(x = magData30N90W.year, y = magData30N90W.intensity, pen="1p,blue")
pscorrelationNA.plot(x = magData30N90W.year, y = magData30N90W.intensity, style="c0.2c", color = "blue", label = '"Magnetic Strength (nT)"')

with pygmt.config(
    MAP_FRAME_PEN = "red",
    MAP_TICK_PEN = "red",
    FONT_ANNOT_PRIMARY = "red",
    FONT_LABEL = "red",
):
    pscorrelationNA.basemap(region = [2000,2020,0,50], frame=["E", 'yaf+l"PsBird Population Density %"'])
    
pscorrelationNA.plot(x = smallCountNAPiper.year, y = (smallCountNAPiper.countS/largeCountNAPiper.countS)*100, pen = "1p,red")
pscorrelationNA.plot(x = smallCountNAPiper.year, y = (smallCountNAPiper.countS/largeCountNAPiper.countS)*100, style = "s0.25c", color = "red", label = '"PsBird Population Density %"')

pscorrelationNA.legend(position = "jTL+o0.1c", box = True)
pscorrelationNA.savefig("PsCorrelation30N90W.png",show = False)
print("5R Analysis of PsBird % In North America (30N 90W) Successfully Updated.")

#rump na correlation (30N 90W) 2000-2020
wrcorrelationNA = pygmt.Figure()
wrcorrelationNA.basemap(region=[2000,2020,46000,50000], projection = "X15c/15c", frame = ["St", "xaf+lYear"])
wrcorrelationNA.basemap(frame=["a", '+t"5R Analysis of WrBird % In North America (30N 90W)"'])
with pygmt.config(
    MAP_FRAME_PEN = "blue",
    MAP_TICK_PEN = "blue", 
    FONT_ANNOT_PRIMARY = "blue", 
    FONT_LABEL = "blue",
):
    wrcorrelationNA.basemap(frame=["W", 'yaf+l"Magnetic Strength (nT)"'])
    
wrcorrelationNA.plot(x = magData30N90W.year, y = magData30N90W.intensity, pen="1p,blue")
wrcorrelationNA.plot(x = magData30N90W.year, y = magData30N90W.intensity, style="c0.2c", color = "blue", label = '"Magnetic Strength (nT)"')

with pygmt.config(
    MAP_FRAME_PEN = "red",
    MAP_TICK_PEN = "red",
    FONT_ANNOT_PRIMARY = "red",
    FONT_LABEL = "red",
):
    wrcorrelationNA.basemap(region = [2000,2020,0,100], frame=["E", 'yaf+l"WrBird Population Density %"'])
    
wrcorrelationNA.plot(x = smallCountNARump.year, y = (smallCountNARump.countS/largeCountNARump.countS)*100, pen = "1p,red")
wrcorrelationNA.plot(x = smallCountNARump.year, y = (smallCountNARump.countS/largeCountNARump.countS)*100, style = "s0.25c", color = "red", label = '"WrBird Population Density %"')

wrcorrelationNA.legend(position = "jTL+o0.1c", box = True)
wrcorrelationNA.savefig("WrCorrelation30N90W.png",show = False)
print("5R Analysis of WrBird % In North America (30N 90W) Successfully Updated.")

#rump na correlation (30N 90W) 2000-2020
shcorrelationNA = pygmt.Figure()
shcorrelationNA.basemap(region=[2000,2020,46000,50000], projection = "X15c/15c", frame = ["St", "xaf+lYear"])
shcorrelationNA.basemap(frame=["a", '+t"5R Analysis of ShBird % In North America (30N 90W)"'])
with pygmt.config(
    MAP_FRAME_PEN = "blue",
    MAP_TICK_PEN = "blue", 
    FONT_ANNOT_PRIMARY = "blue", 
    FONT_LABEL = "blue",
):
    shcorrelationNA.basemap(frame=["W", 'yaf+l"Magnetic Strength (nT)"'])
    
shcorrelationNA.plot(x = magData30N90W.year, y = magData30N90W.intensity, pen="1p,blue")
shcorrelationNA.plot(x = magData30N90W.year, y = magData30N90W.intensity, style="c0.2c", color = "blue", label = '"Magnetic Strength (nT)"')

with pygmt.config(
    MAP_FRAME_PEN = "red",
    MAP_TICK_PEN = "red",
    FONT_ANNOT_PRIMARY = "red",
    FONT_LABEL = "red",
):
    shcorrelationNA.basemap(region = [2000,2020,0,100], frame=["E", 'yaf+l"ShBird Population Density %"'])
    
shcorrelationNA.plot(x = smallCountNAHawk.year, y = (smallCountNAHawk.countS/largeCountNAHawk.countS)*100, pen = "1p,red")
shcorrelationNA.plot(x = smallCountNAHawk.year, y = (smallCountNAHawk.countS/largeCountNAHawk.countS)*100, style = "s0.25c", color = "red", label = '"ShBird Population Density %"')

shcorrelationNA.legend(position = "jTL+o0.1c", box = True)
shcorrelationNA.savefig("ShCorrelation30N90W.png",show = False)
print("5R Analysis of ShBird % In North America (30N 90W) Successfully Updated.")

#piper correlation 2000-2020
correlationP = pygmt.Figure()
correlationP.basemap(region=[2000,2020,22000,26000], projection = "X15c/15c", frame = ["St", "xaf+lYear"])
correlationP.basemap(frame=["a", '+t"5R Analysis of PsBird % In South America (30S 60W)"'])
with pygmt.config(
    MAP_FRAME_PEN = "blue",
    MAP_TICK_PEN = "blue", 
    FONT_ANNOT_PRIMARY = "blue", 
    FONT_LABEL = "blue",
):
    correlationP.basemap(frame=["W", 'yaf+l"Magnetic Strength (nT)"'])
    
correlationP.plot(x = magData3060.year, y = magData3060.intensity, pen="1p,blue")
correlationP.plot(x = magData3060.year, y = magData3060.intensity, style="c0.2c", color = "blue", label = '"Magnetic Strength (nT)"')

with pygmt.config(
    MAP_FRAME_PEN = "red",
    MAP_TICK_PEN = "red",
    FONT_ANNOT_PRIMARY = "red",
    FONT_LABEL = "red",
):
    correlationP.basemap(region = [2000,2020,0,100], frame=["E", 'yaf+l"PsBird Population Density %"'])
    
correlationP.plot(x = smallCountP.year, y = (smallCountP.countS/largeCountP.countS)*100, pen = "1p,red")
correlationP.plot(x = smallCountP.year, y = (smallCountP.countS/largeCountP.countS)*100, style = "s0.25c", color = "red", label = '"PsBird Population Density %"')

correlationP.legend(position = "jTL+o0.1c", box = True)
correlationP.savefig("PsCorrelation.png",show = False)
print("5R Analysis of PsBird % In South America (30S 60W) Successfully Updated.")


#hawk correlation 2000-2020
correlationH = pygmt.Figure()
correlationH.basemap(region=[2000,2020,22000,26000], projection = "X15c/15c", frame = ["St", "xaf+lYear"])
correlationH.basemap(frame=["a", '+t"5R Analysis of ShBird % In South America (30S 60W)"'])
with pygmt.config(
    MAP_FRAME_PEN = "blue",
    MAP_TICK_PEN = "blue", 
    FONT_ANNOT_PRIMARY = "blue", 
    FONT_LABEL = "blue",
):
    correlationH.basemap(frame=["W", 'yaf+l"Magnetic Strength (nT)"'])
    
correlationH.plot(x = magData3060.year, y = magData3060.intensity, pen="1p,blue")
correlationH.plot(x = magData3060.year, y = magData3060.intensity, style="c0.2c", color = "blue", label = '"Magnetic Strength (nT)"')

with pygmt.config(
    MAP_FRAME_PEN = "red",
    MAP_TICK_PEN = "red",
    FONT_ANNOT_PRIMARY = "red",
    FONT_LABEL = "red",
):
    correlationH.basemap(region = [2000,2020,0,100], frame=["E", 'yaf+l"ShBird Population Density %"'])
    
correlationH.plot(x = smallCountH.year, y = (smallCountH.countS/largeCountH.countS)*100, pen = "1p,red")
correlationH.plot(x = smallCountH.year, y = (smallCountH.countS/largeCountH.countS)*100, style = "s0.25c", color = "red", label = '"ShBird Population Density %"')

correlationH.legend(position = "jTL+o0.1c", box = True)
correlationH.savefig("ShCorrelation.png",show = False)
print("5R Analysis of ShBird % In South America (30S 60W) Successfully Updated.")


#white rumped correlation 2000-2020
correlationW = pygmt.Figure()
correlationW.basemap(region=[2000,2020,22000,26000], projection = "X15c/15c", frame = ["St", "xaf+lYear"])
correlationW.basemap(frame=["a", '+t"5R Analysis of WrBird % In South America (30S 60W)"'])
with pygmt.config(
    MAP_FRAME_PEN = "blue",
    MAP_TICK_PEN = "blue", 
    FONT_ANNOT_PRIMARY = "blue", 
    FONT_LABEL = "blue",
):
    correlationW.basemap(frame=["W", 'yaf+l"Magnetic Strength (nT)"'])
    
correlationW.plot(x = magData3060.year, y = magData3060.intensity, pen="1p,blue")
correlationW.plot(x = magData3060.year, y = magData3060.intensity, style="c0.2c", color = "blue", label = '"Magnetic Strength (nT)"')

with pygmt.config(
    MAP_FRAME_PEN = "red",
    MAP_TICK_PEN = "red",
    FONT_ANNOT_PRIMARY = "red",
    FONT_LABEL = "red",
):
    correlationW.basemap(region = [2000,2020,0,100], frame=["E", 'yaf+l"WrBird Population Density %"'])
    
correlationW.plot(x = smallCountW.year, y = (smallCountW.countS/largeCountW.countS)*100, pen = "1p,red")
correlationW.plot(x = smallCountW.year, y = (smallCountW.countS/largeCountW.countS)*100, style = "s0.25c", color = "red", label = '"WrBird Population Density %"')

correlationW.legend(position = "jTL+o0.1c", box = True)
correlationW.savefig("WrCorrelation.png",show = False)
print("5R Analysis of WrBird % In South America (30S 60W) Successfully Updated.")

#all birds correlation 2000-2020
correlationA = pygmt.Figure()
correlationA.basemap(region=[2000,2020,22000,26000], projection = "X15c/15c", frame = ["St", "xaf+lYear"])
correlationA.basemap(frame=["a", '+t"5R Analysis of All Birds % In South America (30S 60W)"'])
with pygmt.config(
    MAP_FRAME_PEN = "blue",
    MAP_TICK_PEN = "blue", 
    FONT_ANNOT_PRIMARY = "blue", 
    FONT_LABEL = "blue",
):
    correlationA.basemap(frame=["W", 'yaf+l"Magnetic Strength (nT)"'])
    
correlationA.plot(x = magData3060.year, y = magData3060.intensity, pen="1p,blue")
correlationA.plot(x = magData3060.year, y = magData3060.intensity, style="c0.2c", color = "blue", label = '"Magnetic Strength (nT)"')

with pygmt.config(
    MAP_FRAME_PEN = "red",
    MAP_TICK_PEN = "red",
    FONT_ANNOT_PRIMARY = "red",
    FONT_LABEL = "red",
):
    correlationA.basemap(region = [2000,2020,0,100], frame=["E", 'yaf+l"All Birds Population Density %"'])
    
# correlationA.plot(x = smallCount.year, y = ((smallCount.countS + smallCountP.countS + smallCountH.countS + smallCountW.countS)/(largeCount.countS + largeCountP.countS + largeCountH.countS + largeCountW.countS))*100, pen = "1p,red")
# correlationA.plot(x = smallCount.year, y = ((smallCount.countS + smallCountP.countS + smallCountH.countS + smallCountW.countS)/(largeCount.countS + largeCountP.countS + largeCountH.countS + largeCountW.countS))*100, style = "s0.25c", color = "red", label = '"All Birds Population Density %"')
correlationA.plot(x = smallCount.year, y = ((smallCount.countS/largeCount.countS) + (smallCountP.countS/largeCountP.countS) + (smallCountW.countS/largeCountW.countS) + (smallCountH.countS/largeCountH.countS)) * 25, pen = "1p,red")
correlationA.plot(x = smallCount.year, y = ((smallCount.countS/largeCount.countS) + (smallCountP.countS/largeCountP.countS) + (smallCountW.countS/largeCountW.countS) + (smallCountH.countS/largeCountH.countS)) * 25, style = "s0.25c", color = "red", label = '"All Birds Population Density %"')

correlationA.legend(position = "jTL+o0.1c", box = True)
correlationA.savefig("AbCorrelation.png",show = False)
print("5R Analysis of All Birds % In South America (30S 60W) Successfully Updated.")

#all birds correlation 1970-2020
EcorrelationA = pygmt.Figure()
EcorrelationA.basemap(region=[1970,2020,22000,27000], projection = "X15c/15c", frame = ["St", "xaf+lYear"])
EcorrelationA.basemap(frame=["a", '+t"5R Analysis of All Birds % In South America (30S 60W)"'])
with pygmt.config(
    MAP_FRAME_PEN = "blue",
    MAP_TICK_PEN = "blue", 
    FONT_ANNOT_PRIMARY = "blue", 
    FONT_LABEL = "blue",
):
    EcorrelationA.basemap(frame=["W", 'yaf+l"Magnetic Strength (nT)"'])
    
EcorrelationA.plot(x = fullMagData3060.year, y = fullMagData3060.intensity, pen="1p,blue")
EcorrelationA.plot(x = fullMagData3060.year, y = fullMagData3060.intensity, style="c0.2c", color = "blue", label = '"Magnetic Strength (nT)"')

with pygmt.config(
    MAP_FRAME_PEN = "red",
    MAP_TICK_PEN = "red",
    FONT_ANNOT_PRIMARY = "red",
    FONT_LABEL = "red",
):
    EcorrelationA.basemap(region = [1970,2020,0,100], frame=["E", 'yaf+l"All Birds Population Density %"'])
    
EcorrelationA.plot(x = EsmallCount.year, y = ((EsmallCount.countS + EsmallCountP.countS + EsmallCountH.countS + EsmallCountW.countS)/(ElargeCount.countS + ElargeCountP.countS + ElargeCountH.countS + ElargeCountW.countS))*100, pen = "1p,red")
EcorrelationA.plot(x = EsmallCount.year, y = ((EsmallCount.countS + EsmallCountP.countS + EsmallCountH.countS + EsmallCountW.countS)/(ElargeCount.countS + ElargeCountP.countS + ElargeCountH.countS + ElargeCountW.countS))*100, style = "s0.25c", color = "red", label = '"All Birds Population Density %"')

EcorrelationA.legend(position = "jTL+o0.1c", box = True)
EcorrelationA.savefig("AbCorrelationFull5R.png",show = False)
print("5R Analysis of All Birds % In South America (30S 60W) Successfully Updated.")

#forktailed correlation 2000-2020
correlationF = pygmt.Figure()
correlationF.basemap(region=[2000,2020,22000,26000], projection = "X15c/15c", frame = ["St", "xaf+lYear"])
correlationF.basemap(frame=["a", '+t"5R Analysis of FtBird % In South America (30S 60W)"'])
with pygmt.config(
    MAP_FRAME_PEN = "blue",
    MAP_TICK_PEN = "blue", 
    FONT_ANNOT_PRIMARY = "blue", 
    FONT_LABEL = "blue",
):
    correlationF.basemap(frame=["W", 'yaf+l"Magnetic Strength (nT)"'])
    
correlationF.plot(x = magData3060.year, y = magData3060.intensity, pen="1p,blue")
correlationF.plot(x = magData3060.year, y = magData3060.intensity, style="c0.2c", color = "blue", label = '"Magnetic Strength (nT)"')

with pygmt.config(
    MAP_FRAME_PEN = "red",
    MAP_TICK_PEN = "red",
    FONT_ANNOT_PRIMARY = "red",
    FONT_LABEL = "red",
):
    correlationF.basemap(region = [2000,2020,0,100], frame=["E", 'yaf+l"FtBird Population Density %"'])
    
correlationF.plot(x = smallCountF.year, y = (smallCountF.countS/largeCountF.countS)*100, pen = "1p,red")
correlationF.plot(x = smallCountF.year, y = (smallCountF.countS/largeCountF.countS)*100, style = "s0.25c", color = "red", label = '"FtBird Population Density %"')

correlationF.legend(position = "jTL+o0.1c", box = True)
correlationF.savefig("FtCorrelation.png",show = False)
print("5R Analysis of FtBird % In South America (30S 60W) Successfully Updated.")



#all birds correlation 2000-2020 (3R)
# AgAdjustment1 = specificMessing.query('-27 >= decimalLatitude >= -33 &  -57 >= decimalLongitude >= -63 & individualCount < 1000')

Ecorrelation = pygmt.Figure()
Ecorrelation.basemap(region=[2000,2020,22000,27000], projection = "X15c/15c", frame = ["St", "xaf+lYear"])
Ecorrelation.basemap(frame=["a", '+t"4R Analysis of All Birds % In South America (30S 60W)"'])
with pygmt.config(
    MAP_FRAME_PEN = "blue",
    MAP_TICK_PEN = "blue", 
    FONT_ANNOT_PRIMARY = "blue", 
    FONT_LABEL = "blue",
):
    Ecorrelation.basemap(frame=["W", 'yaf+l"Magnetic Strength (nT)"'])
    
Ecorrelation.plot(x = magData3060.year, y = magData3060.intensity, pen="1p,blue")
Ecorrelation.plot(x = magData3060.year, y = magData3060.intensity, style="c0.2c", color = "blue", label = '"Magnetic Strength (nT)"')

with pygmt.config(
    MAP_FRAME_PEN = "red",
    MAP_TICK_PEN = "red",
    FONT_ANNOT_PRIMARY = "red",
    FONT_LABEL = "red",
):
    Ecorrelation.basemap(region = [2000,2020,0,100], frame=["E", 'yaf+l"All Birds Population Density %"'])
    
Ecorrelation.plot(x = QQsmallCount.year, y = ((QQsmallCount.countS + QQsmallCountP.countS + QQsmallCountH.countS + QQsmallCountW.countS)/(QQlargeCount.countS + QQlargeCountP.countS + QQlargeCountH.countS + QQlargeCountW.countS))*100, pen = "1p,red")
Ecorrelation.plot(x = QQsmallCount.year, y = ((QQsmallCount.countS + QQsmallCountP.countS + QQsmallCountH.countS + QQsmallCountW.countS)/(QQlargeCount.countS + QQlargeCountP.countS + QQlargeCountH.countS + QQlargeCountW.countS))*100, style = "s0.25c", color = "red", label = '"All Birds Population Density %"')

Ecorrelation.legend(position = "jTL+o0.1c", box = True)
Ecorrelation.savefig("AbCorrelation00203R.png",show = False)
print("4R Analysis of All Birds % In South America (30S 60W) Successfully Updated.")


print("done ")


