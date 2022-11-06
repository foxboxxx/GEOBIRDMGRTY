import pandas as pd
import numpy as np
import pygmt
import datetime
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from scipy import stats
import seaborn as sb
import math

def lineBreak(num):
    for i in np.arange(0, num):
        print("\n")

def histogramMaker(data, title, filename):
    r = len(data)
    ax = sb.histplot(data=data, x = np.arange(0,r), y="count", bins = 20)
    ax.set(xlabel='Month', ylabel='Population Density', title = title)  
    plt.savefig(filename)

# Code used for creating double y-axis plots comparing any two data
def doubleYAxisPlotMaker(yearMin, yearMax, data1, data2, title, dataLabel1, dataLabel2, 
                         filename, color1, color2):
    t = np.arange(yearMin,yearMax, 1)
    fig, ax1 = plt.subplots()
    # Creating and labeling the first axis
    ax1.set_xlabel('Year')
    ax1.set_ylabel(dataLabel1, color = color1)
    ax1.set_title(title)
    ln1 = ax1.plot(t, data1, color= color1, label = dataLabel1)
    ax1.tick_params(axis='y', labelcolor = color1)
    ax1.grid(color = 'grey', linestyle = "--")
    plt.xticks([2000, 2003, 2006, 2009, 2012, 2015, 2018])

    ax2 = ax1.twinx()  
    # Creating and lebeling the second axis
    ax2.set_ylabel(dataLabel2, color = color2)  
    ln2 = ax2.plot(t, data2, color = color2, label = dataLabel2)
    ax2.set_ylim([0,100])
    ax2.tick_params(axis='y', labelcolor = color2)

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper right")

    fig.tight_layout() 
    fig.savefig(filename)
    print(filename + " successfully uploaded.")
    
# Code used for creating scatter plots    
def scatterPlot(mag, per, title, file, xlabel, colori):
    font = {'fontname':'Helvetica', 'style':'italic'}
    plt.clf()
    sb.regplot(mag, per, 'o',color = colori)
    plt.title(title, **font)
    plt.xlabel(xlabel, **font)
    plt.ylabel("Population Density %", **font)
    plt.grid(color='grey', linestyle = "--", linewidth=0.25, alpha=0.75)
    plt.ylim(0,100) 
    plt.savefig(file)
    
def yearCount(smallSet, bigSet, yearMin, yearMax):
    for i in np.arange(yearMin, yearMax):
        smallSet = smallSet.append(pd.DataFrame({'year': i, 'countS': bigSet.query('year == @i')['individualCount'].sum()}, index = [0]), ignore_index = True)

    return smallSet

def monthCount(smallSet, bigSet, yearMin, yearMax):
    smallSetMonth = pd.DataFrame({})
    for i in np.arange(yearMin, yearMax):
        smallSet = smallSet.append(pd.DataFrame({'year': i, 'countS': bigSet.query('year == @i')['individualCount'].sum()}, index = [0]), ignore_index = True)
        for j in np.arange(1,13):
            smallSetMonth = smallSetMonth.append(pd.DataFrame({'month': j, 'count': bigSet.query('year == @i & month == @j')['individualCount'].sum()}, index = [0]), ignore_index = True)
    return smallSetMonth



# Data for plotting
def plotter(smallData, largeData, start, finish, pltr, label):
    for i in range(start, finish):
        initStart = 12 * i
        initEnd = initStart + 11
        pltr.plot(np.arange(1,13), (smallData.loc[initStart:initEnd]['count'] / (largeData.loc[initStart:initEnd]['count'] + 0.01)) * 100, 'C' + str(i), label = str(2000 + i))
        pltr.set_title(label)
        pltr.set_xlim(1,12)
        pltr.set_ylim(0,100)

def deg2rad(deg):
  return deg * (math.pi/180)

def getDistanceFromLatLonInKm(lat1,lon1,lat2,lon2):
    R = 6371; # Radius of the earth in km
    dLat = deg2rad(lat2 - lat1);  # deg2rad below
    dLon = deg2rad(lon2 - lon1); 
    a = math.sin(dLat/2)**2 + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)); 
    d = R * c; # Distance in km
    return d