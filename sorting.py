import pandas as pd
import numpy as np
import pygmt
import datetime
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import statsmodels.graphics.api as smg
import statsmodels.api as sm
from scipy import stats
import seaborn as sb

print("Begininng Updating Sequence...")
#testData = pd.read_csv(r"https://docs.google.com/spreadsheets/d/e/2PACX-1vQ1QvEdtQLKMGXxyN0QhQL46AdUdJ9iJVB-50NBHe1s_tnx2KCymG0ZYX43eB-SEjTl9Dj_5WHU8qj3/pub?output=csv")
magneticData37 = pd.read_csv(r"magneticData37.csv")
magData3060 = pd.read_csv(r"3060intensity.csv")
fullMagData3060 = pd.read_csv(r"3060fullMag.csv")
#print(magneticData37)
birdData = pd.read_csv(r"filteredPloverData.csv")
magData30N90W = pd.read_csv(r'30N90Wdata.csv')

messingData = pd.read_csv(r"plovercsv.csv")
messingData = messingData.loc[messingData['occurrenceStatus']=='PRESENT',['eventDate','individualCount','decimalLatitude','decimalLongitude','day','month','year']].copy()

pSandpiperRaw = pd.read_csv(r"pectoralSandpiperUnfiltered.csv")
pSandpiperRaw = pSandpiperRaw.loc[pSandpiperRaw['occurrenceStatus']=='PRESENT',['eventDate','individualCount','decimalLatitude','decimalLongitude','day','month','year']].copy()

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
southAmericanScreamer = sScreamerRaw.query('0 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')
specificScreamer = sScreamerRaw.query('-25 >= decimalLatitude >= -35 & -55 >= decimalLongitude >= -65 & individualCount < 1000')

southAmericanPlover = messingData.query('0 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')
northAmericanPlover = messingData.query('0 <= decimalLatitude <= 50 & -120 <= decimalLongitude <= -60 & individualCount < 1000')

specificMessing = messingData.query('-25 >= decimalLatitude >= -35 & -55 >= decimalLongitude >= -65 & individualCount < 1000')
specificMessing2 = messingData.query('-26 >= decimalLatitude >= -34 & -56 >= decimalLongitude >= -64 & individualCount < 1000')
specificMessing3 = messingData.query('25 <= decimalLatitude <= 35 & -85 >= decimalLongitude >= -95 & individualCount < 1000')


southAmericanPiper = pSandpiperRaw.query('0 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')
northAmericanPiper = pSandpiperRaw.query('0 <= decimalLatitude <= 50 & -120 <= decimalLongitude <= -60 & individualCount < 1000')

specificPiper = pSandpiperRaw.query('-25 >= decimalLatitude >= -35 &  -55 >= decimalLongitude >= -65 & individualCount < 1000')
specificPiper2 = pSandpiperRaw.query('-26 >= decimalLatitude >= -33 &  -56 >= decimalLongitude >= -64 & individualCount < 1000')
specificPiper3 = pSandpiperRaw.query('25 <= decimalLatitude <= 35 & -85 >= decimalLongitude >= -95 & individualCount < 1000')


southAmericanHawk = sHawkRaw.query('0 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')
northAmericanHawk = pSandpiperRaw.query('0 <= decimalLatitude <= 50 & -120 <= decimalLongitude <= -60 & individualCount < 1000')

specificHawk = sHawkRaw.query('-25 >= decimalLatitude >= -35 &  -55 >= decimalLongitude >= -65 & individualCount < 1000')
specificHawk2 = sHawkRaw.query('-26 >= decimalLatitude >= -33 &  -56 >= decimalLongitude >= -64 & individualCount < 1000')
specificHawk3 = sHawkRaw.query('25 <= decimalLatitude <= 35 & -85 >= decimalLongitude >= -95 & individualCount < 1000')


southAmericanRump = wSandpiperRaw.query('0 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')
northAmericanRump = wSandpiperRaw.query('0 <= decimalLatitude <= 50 & -120 <= decimalLongitude <= -60 & individualCount < 1000')


specificRump = wSandpiperRaw.query('-25 >= decimalLatitude >= -35 & -55 >= decimalLongitude >= -65 & individualCount < 1000')
specificRump2 = wSandpiperRaw.query('-26 >= decimalLatitude >= -33 &  -57 >= decimalLongitude >= -64 & individualCount < 1000')
specificRump3 = wSandpiperRaw.query('25 <= decimalLatitude <= 35 & -85 >= decimalLongitude >= -95 & individualCount < 1000')


southAmericanForktail = fTailedRaw.query('0 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')
specificForktail = fTailedRaw.query('-25 >= decimalLatitude >= -35 & -55 >= decimalLongitude >= -65 & individualCount < 1000')


def yearCount(smallSet, bigSet, yearMin, yearMax):
    for i in np.arange(yearMin, yearMax):
        smallSet = smallSet.append(pd.DataFrame({'year': i, 'countS': bigSet.query('year == @i')['individualCount'].sum()}, index = [0]), ignore_index = True)
    return smallSet

smallCount = largeCount = smallCountP = largeCountP = smallCountH = largeCountH = smallCountW = largeCountW = smallCountF = largeCountF = smallCountNAPlover = largeCountNAPlover = smallCountNAPiper = largeCountNAPiper = smallCountNARump = largeCountNARump = smallCountNAHawk = largeCountNAHawk = smallCountScreamer = largeCountScreamer = pd.DataFrame({'year':[], 'countS':[]})
smallCount = yearCount(smallCount, specificMessing, 2000, 2020)
largeCount = yearCount(largeCount, southAmericanPlover, 2000, 2020)

smallCountP = yearCount(smallCountP, specificPiper, 2000, 2020)
largeCountP = yearCount(largeCountP, southAmericanPiper, 2000, 2020)

smallCountH = yearCount(smallCountH, specificHawk, 2000, 2020)
largeCountH =yearCount(largeCountH, southAmericanHawk, 2000, 2020)

smallCountW = yearCount(smallCountW, specificRump, 2000, 2020)
largeCountW = yearCount(largeCountW, southAmericanRump, 2000, 2020)

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

smallCountScreamer = yearCount(smallCountScreamer, specificScreamer, 2005, 2020)
largeCountScreamer = yearCount(largeCountScreamer, southAmericanScreamer, 2005, 2020)


newm = magData3060.copy().query('2000 <= year < 2020')
magStrength3060S = newm['intensity'].values.tolist()

otherm = magData30N90W.copy().query('2000 <= year < 2020')
magStrength30N90W = otherm['intensity'].values.tolist()



print(matplotlib.get_cachedir())

#correlation tests
#these are the dataframes needed for the all birds correlation
def scatterPlot(mag, per, title, file):
    font = {'fontname':'Helvetica', 'style':'italic'}
    plt.clf()
    plt.scatter(mag, per)
    plt.title(title, **font)
    plt.xlabel("Magnetic Strength (nT)", **font)
    plt.ylabel("Population Density %", **font)
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    plt.savefig(file)
    

print("|------------------------------------------------------------------------|")#section start
smallCountPercentages = (smallCount['countS'].copy() / largeCount['countS'].copy()) * 100
smallCountWPercentages = (smallCountW['countS'].copy() / largeCountW['countS'].copy()) * 100
smallCountPPercentages = (smallCountP['countS'].copy() / largeCountP['countS'].copy()) * 100
smallCountHPercentages = (smallCountH['countS'].copy() / largeCountH['countS'].copy()) * 100
allBirdPercentages = smallCountPercentages.multiply(0.25) + smallCountWPercentages.multiply(0.25) + smallCountPPercentages.multiply(0.25) + smallCountHPercentages.multiply(0.25)
print("> 30S60W All Birds: " + "r value = " + str(stats.pearsonr(magStrength3060S,allBirdPercentages)[0]) + "; p value = " + str(stats.pearsonr(magStrength3060S, allBirdPercentages)[1]))
scatterPlot(magStrength3060S, allBirdPercentages, "30S60W All Birds", "30S60WAbBird.png")
# a, b = np.polyfit(magStrength3060S, allBirdPercentages, deg = 1)
# aBest = a * magStrength3060S + allBirdPercentages
# aBerr = magStrength3060S.std() * np.sqrt(1/len(magStrength3060S) + (magStrength3060S - magStrength3060S.mean())**2 / np.sum((magStrength3060S - magStrength3060S.mean())**2))

# fig, ax = plt.subplots()
# ax.plot(magStrength3060S, aBest, '-')
# ax.fill_between(magStrength3060S, aBest - aBerr, aBest + aBerr, alpha = 0.2)
# ax.plot(magStrength3060S, allBirdPercentages, 'o', color='tab:blue')

ploverSmallList3060S = smallCount.copy()['countS'].values.tolist()
ploverLargeList3060S = largeCount.copy()['countS'].values.tolist()
percentagesPlover3060S = []
for x in range(len(ploverSmallList3060S)):
    percentagesPlover3060S.append((ploverSmallList3060S[x]/ploverLargeList3060S[x]) * 100)
print("> 30S60W American Golden-Plover: " + "r value = " + str(stats.pearsonr(magStrength3060S,percentagesPlover3060S)[0]) + "; p value = " + str(stats.pearsonr(magStrength3060S, percentagesPlover3060S)[1]))
scatterPlot(magStrength3060S, percentagesPlover3060S, "30S60W American Golden-Plover", "30S60WAgBird.png")

#White-rumped Sandpiper South America
whiteSmallList3060S = smallCountW.copy()['countS'].values.tolist()
whiteLargeList3060S = largeCountW.copy()['countS'].values.tolist()
percentagesWhite3060S = []
for x in range(len(whiteSmallList3060S)):
    percentagesWhite3060S.append((whiteSmallList3060S[x]/whiteLargeList3060S[x]) * 100)
print("> 30S60W White-rumped Sandpiper: " + "r value = " + str(stats.pearsonr(magStrength3060S,percentagesWhite3060S)[0]) + "; p value = " + str(stats.pearsonr(magStrength3060S, percentagesWhite3060S)[1]))
scatterPlot(magStrength3060S, percentagesWhite3060S, "30S60W White-rumped Sandpiper", "30S60WWrBird.png")

#Pectoral Sandpiper South America
pectoralSmallList3060S = smallCountP.copy()['countS'].values.tolist()
pectoralLargeList3060S = largeCountP.copy()['countS'].values.tolist()
percentagesPectoral3060S = []
for x in range(len(pectoralSmallList3060S)):
    percentagesPectoral3060S.append((pectoralSmallList3060S[x]/pectoralLargeList3060S[x]) * 100)
print("> 30S60W Pectoral Sandpiper: " + "r value = " + str(stats.pearsonr(magStrength3060S,percentagesPectoral3060S)[0]) + "; p value = " + str(stats.pearsonr(magStrength3060S, percentagesPectoral3060S)[1]))
scatterPlot(magStrength3060S, percentagesPectoral3060S, "30S60W Pectoral Sandpiper", "30S60WPsBird.png")

#Swainson's Hawk South America
hawkSmallList3060S = smallCountH.copy()['countS'].values.tolist()
hawkLargeList3060S = largeCountH.copy()['countS'].values.tolist()
percentagesHawk3060S = []
for x in range(len(hawkSmallList3060S)):
    percentagesHawk3060S.append((hawkSmallList3060S[x]/hawkLargeList3060S[x]) * 100)
print("> 30S60W Swainson's Hawk: " + "r value = " + str(stats.pearsonr(magStrength3060S,percentagesHawk3060S)[0]) + "; p value = " + str(stats.pearsonr(magStrength3060S, percentagesHawk3060S)[1]))
scatterPlot(magStrength3060S, percentagesHawk3060S, "30S60W Swainson's Hawk", "30S60WShBird.png")

#Fork-tailed Flycatcher South America
forkSmallList3060S = smallCountF.copy()['countS'].values.tolist()
forkLargeList3060S = largeCountF.copy()['countS'].values.tolist()
percentagesFork3060S = []
for x in range(len(forkSmallList3060S)):
    percentagesFork3060S.append((forkSmallList3060S[x]/forkLargeList3060S[x]) * 100)
print("> 30S60W Fork-tailed Flycatcher: " + "r value = " + str(stats.pearsonr(magStrength3060S,percentagesFork3060S)[0]) + "; p value = " + str(stats.pearsonr(magStrength3060S, percentagesFork3060S)[1]))
scatterPlot(magStrength3060S, percentagesFork3060S, "30S60W Fork-tailed Flycatcher", "30S60WFtBird.png")

#<--------------------------------------WIP RN-------------------------------------------->
# toto = magData3060.copy().query('year >= 2005 & year < 2020')
# nMMNn = toto['intensity'].values.tolist()

# #Southern Screamer South America
# screamSmallList3060S = smallCountScreamer.copy()['countS'].values.tolist()
# screamLargeList3060S = largeCountScreamer.copy()['countS'].values.tolist()
# percentagesScream3060S = []
# for x in range(len(screamSmallList3060S)):
#     percentagesScream3060S.append((screamSmallList3060S[x]/screamLargeList3060S[x]) * 100)
# print("> 30S60W Southern Screamer: " + "r value = " + str(stats.pearsonr(nMMNn,percentagesScream3060S)[0]) + "; p value = " + str(stats.pearsonr(nMMNn, percentagesScream3060S)[1]))
# scatterPlot(nMMNn, percentagesScream3060S, "30S60W Southern Screamer", "30S60WSsBird.png")

#American Golden-Plover Correlation North America
NAPloverSmallList3060S = smallCountNAPlover.copy()['countS'].values.tolist()
NAPloverLargeList3060S = largeCountNAPlover.copy()['countS'].values.tolist()
percentagesNAPlover30N90W = []
for x in range(len(NAPloverSmallList3060S)):
    percentagesNAPlover30N90W.append((NAPloverSmallList3060S[x]/NAPloverLargeList3060S[x]) * 100)
print("> 30N90W American Golden-Plover: " + "r value = " + str(stats.pearsonr(magStrength30N90W,percentagesNAPlover30N90W)[0]) + "; p value = " + str(stats.pearsonr(magStrength30N90W, percentagesNAPlover30N90W)[1]))
scatterPlot(magStrength30N90W, percentagesNAPlover30N90W, "30N90W American Golden-Plover", "30N90WAgBird.png")

#White-rumped Sandpiper Correlation North America
NAWhiteSmallList3060S = smallCountNARump.copy()['countS'].values.tolist()
NAWhiteLargeList3060S = largeCountNARump.copy()['countS'].values.tolist()
percentagesNAWhite30N90W = []
for x in range(len(NAWhiteSmallList3060S)):
    percentagesNAWhite30N90W.append((NAWhiteSmallList3060S[x]/NAWhiteLargeList3060S[x]) * 100)
print("> 30N90W White-rumped Sandpiper: " + "r value = " + str(stats.pearsonr(magStrength30N90W,percentagesNAWhite30N90W)[0]) + "; p value = " + str(stats.pearsonr(magStrength30N90W, percentagesNAWhite30N90W)[1]))
scatterPlot(magStrength30N90W, percentagesNAWhite30N90W, "30N90W White-rumped Sandpiper", "30N90WWrBird.png")

#Pectoral Sandpiper Correlation North America
NAPiperSmallList3060S = smallCountNAPiper.copy()['countS'].values.tolist()
NAPiperLargeList3060S = largeCountNAPiper.copy()['countS'].values.tolist()
percentagesNAPiper30N90W = []
for x in range(len(NAPiperSmallList3060S)):
    percentagesNAPiper30N90W.append((NAPiperSmallList3060S[x]/NAPiperLargeList3060S[x]) * 100)
print("> 30N90W Pectoral Sandpiper: " + "r value = " + str(stats.pearsonr(magStrength30N90W,percentagesNAPiper30N90W)[0]) + "; p value = " + str(stats.pearsonr(magStrength30N90W, percentagesNAPiper30N90W)[1]))
scatterPlot(magStrength30N90W, percentagesNAPiper30N90W, "30N90W Pectoral Sandpiper", "30N90WPsBird.png")

#Swainson's Hawk Correlation North America
NAHawkSmallList3060S = smallCountNAHawk.copy()['countS'].values.tolist()
NAHawkLargeList3060S = largeCountNAHawk.copy()['countS'].values.tolist()
percentagesNAHawk30N90W = []
for x in range(len(NAHawkSmallList3060S)):
    percentagesNAHawk30N90W.append((NAHawkSmallList3060S[x]/NAHawkLargeList3060S[x]) * 100)
print("> 30N90W Swainson's Hawk: " + "r value = " + str(stats.pearsonr(magStrength30N90W,percentagesNAHawk30N90W)[0]) + "; p value = " + str(stats.pearsonr(magStrength30N90W, percentagesNAHawk30N90W)[1]))
scatterPlot(magStrength30N90W, percentagesNAHawk30N90W, "30N90W Swainson's Hawk", "30N90WShBird.png")

print("|------------------------------------------------------------------------|")#section end

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
QQlargeCountH =yearCount(QQlargeCountH, southAmericanHawk, 2000, 2020)

QQsmallCountW = yearCount(QQsmallCountW, specificRump2, 2000, 2020)
QQlargeCountW = yearCount(QQlargeCountW, southAmericanRump, 2000, 2020)



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
worldRegion = [-170,180,-60,80]

#1970-1979
seventies = preciseBirdData.query('1970 <= year < 1980')
seventiesFig = pygmt.Figure()
seventiesFig.basemap(region=finalRegion,projection="M8i",frame=["a", '+t"American Golden-Plover (AgBird)- South America (1970-1980)"'])
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
print("American Golden-Plover (AgBird)- South America (1970-1980) Successfully Updated.")

#1980-1989
eighties = preciseBirdData.query('1980 <= year < 1990')
eightiesFig = pygmt.Figure()
eightiesFig.basemap(region=finalRegion,projection="M8i",frame=["a", '+t"American Golden-Plover (AgBird)- South America (1970-1980)"'])
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
print("American Golden-Plover (AgBird)- South America (1970-1980) Successfully Updated.")

#1990-1999
nineties = preciseBirdData.query('1990 <= year < 2000')
ninetiesFig = pygmt.Figure()
ninetiesFig.basemap(region=finalRegion,projection="M8i",frame=["a", '+t"American Golden-Plover (AgBird)- South America (1990-2000)"'])
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
print("American Golden-Plover (AgBird)- South America (1990-2000) Successfully Updated.")

#2000-2009
twozero = preciseBirdData.query('2000 <= year < 2010')
twozeroFig = pygmt.Figure()
twozeroFig.basemap(region=finalRegion,projection="M8i",frame=["a", '+t"American Golden-Plover (AgBird)- South America (2000-2010)"'])
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
print("American Golden-Plover (AgBird)- South America (2000-2010) Successfully Updated.")

#2010-Present
twoten = preciseBirdData.query('2010 <= year')
twotenFig = pygmt.Figure()
twotenFig.basemap(region=finalRegion,projection="M8i",frame=["a", '+t"American Golden-Plover (AgBird)- South America (2010-Present)"'])
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
print("American Golden-Plover (AgBird)- South America (2010-Present) Successfully Updated.")

#seventies.to_csv('seventies.csv')
#eighties.to_csv('eighties.csv')
#nineties.to_csv('nineties.csv')
#twozero.to_csv('twozero.csv')
#twoten.to_csv('twoten.csv')

#Everything
fig = pygmt.Figure()
fig.basemap(region=finalRegion, projection="M8i", frame=["a", '+t"American Golden-Plover (AgBird)- South America"'])
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
print("American Golden-Plover (AgBird)- South America Successfully Updated.")

xList = []
yList = []
length = len(subsetOne)

#PectoralSandpipers Everything
newDf = pd.read_csv(r"pectoralSandpiperFiltered.csv", encoding='latin1')
#newDf = newDf.query("2000 >= year >= 1970").copy()
pSand = pygmt.Figure()
pSand.basemap(region=finalRegion, projection="M8i", frame=["a", '+t"Pectoral Sandpiper (PsBird)- South America"'])
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
print("Pectoral Sandpiper (PsBird)- South America Successfully Updated.")

#swainsonHawk
hawkDf = pd.read_csv(r"swainsonHawkFiltered.csv")
#hawkDf = hawkDf.query("2000 >= year >= 1970").copy()
sHawk = pygmt.Figure()
sHawk.basemap(region=finalRegion, projection="M8i", frame=["a", '+t"Swainsons Hawk (ShBird)- South America"'])
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
print("Swainsons Hawk (ShBird)- South America Successfully Updated.")


#White Rumped Pectoral
wRump = pd.read_csv(r"whiteRumpedFiltered.csv")
#wRump = wRump.query("2000 >= year >= 1970").copy()
wrp = pygmt.Figure()
wrp.basemap(region=finalRegion, projection="M8i", frame=["a", '+t"Pectoral White-Rumped Sandpiper (WrBird)- South America"'])
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
print("Pectoral White-Rumped Sandpiper (WrBird)- South America Successfully Updated.")

#forktailed flycatcher (control, native south american bird)
fTailed = pd.read_csv(r"forktailedFiltered.csv")
ftd = pygmt.Figure()
ftd.basemap(region = finalRegion, projection = "M8i", frame=["a", '+t"Fork-tailed Flycatcher [Control] (FtBird)- South America"'])
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
print("Fork-tailed Flycatcher [Control] (FtBird)- South America Successfully Updated.")


#shawkworld
sH = pd.read_csv(r"swainsonHawk.csv")
sH = sH.loc[sH['occurrenceStatus']=='PRESENT',['eventDate','individualCount','decimalLatitude','decimalLongitude','day','month','year']].copy()
sH = sH.query("year >= 1970")
sH = sH.query("individualCount <= 10000")
HawkFull = pygmt.Figure()
HawkFull.basemap(region=worldRegion, projection="M8i", frame=["a", '+t"Swainsons Hawk (ShBird)- World"'])
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
print("Swainsons Hawk (ShBird)- World Successfully Updated.")

#fullWorldOfPlover
tB = pd.read_csv(r"plovercsv.csv")
tB = tB.loc[tB['occurrenceStatus']=='PRESENT',['eventDate','individualCount','decimalLatitude','decimalLongitude','day','month','year']].copy()
tB = tB.query("year >= 1970")
tB = tB.query("individualCount <= 10000")
ploverFull = pygmt.Figure()
ploverFull.basemap(region=worldRegion, projection="M8i", frame=["a", '+t"American Golden-Plover (AgBird)- World"'])
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
print("American Golden-Plover (AgBird)- World Successfully Updated.")


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
print("Pectoral Sandpiper (PsBird)- World Successfully Updated.")

#Full World Of White-Rumped Sandpiper
wW = pd.read_csv(r"whiteRumpedRaw.csv")
wW = wW.loc[wW['occurrenceStatus']=='PRESENT',['eventDate','individualCount','decimalLatitude','decimalLongitude','day','month','year']].copy()
wW = wW.query("year >= 1970")
wW = wW.query("individualCount <= 10000")
whiteFull = pygmt.Figure()
whiteFull.basemap(region=worldRegion, projection="M8i", frame=["a", '+t"White-Rumped Sandpiper (WrBird)- World"'])
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
print("White-Rumped Sandpiper (WrBird)- World Successfully Updated.")

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




# #scatterplot
# region = pygmt.info(
#     table=birdData[["decimalLatitude", "decimalLongitude"]],  # x and y columns
#     per_column=True, 
# )

# scatter = pygmt.Figure()

# scatter.basemap(
#     region=region,
#     projection="X10c/10c",
#     frame=[
#         'xafg+l"Latitude (S)"',
#         'yafg+l"Longitude (W)"',
#         'WSen+t"Penguin size at Palmer Station"',
#     ],
# )

# pygmt.makecpt(cmap="inferno", series=(0, 2, 1), color_model="+cAdelie,Chinstrap,Gentoo")

# scatter.plot(
#     # Use bill length and bill depth as x and y data input, respectively
#     x = birdData.decimalLatitude,
#     y = birdData.decimalLongitude,
#     # Vary each symbol size according to another feature (body mass, scaled by 7.5*10e-5)
#     size = 1,
#     # Points colored by categorical number code
#     color=birdData.year(int),
#     # Use colormap created by makecpt
#     cmap=True,
#     # Do not clip symbols that fall close to the map bounds
#     no_clip=True,
#     # Use circles as symbols with size in centimeter units
#     style="cc",
#     # Set transparency level for all symbols to deal with overplotting
#     transparency = 40,
# )

# # Add colorbar legend
# scatter.colorbar()
# scatter.savefig("AgBird_ScatterPlot.png")

print("Program finished.")

