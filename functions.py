import pandas as pd
import numpy as np
import pygmt
import datetime
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from scipy import stats
import seaborn as sb

def lineBreak(num):
    for i in np.arange(0, num):
        print("\n")

def histogramMaker(data, title, filename):
    r = len(data)
    ax = sb.histplot(data=data, x = np.arange(0,r), y="count", bins = 20)
    ax.set(xlabel='Month', ylabel='Population Density', title = title)  
    plt.savefig(filename)

    # l = len(data)
    # sb.barplot(x = np.arange(0,l), y = "count", data = data,)
    # plt.savefig("####.png")

def doubleYAxisPlotMaker(yearMin, yearMax, data1, data2, title, dataLabel1, dataLabel2, filename, color1, color2):
    t = np.arange(yearMin,yearMax, 1)
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Year')
    ax1.set_ylabel(dataLabel1, color = color1)
    #5R Analysis of American Golden-Plover 30S60W from 2000-2020
    ax1.set_title(title)
    ln1 = ax1.plot(t, data1, color= color1, label = dataLabel1)
    ax1.tick_params(axis='y', labelcolor = color1)
    ax1.grid(color = 'grey', linestyle = "--")
    # ax1.legend(loc="upper right")

    ax2 = ax1.twinx()  

    ax2.set_ylabel(dataLabel2, color = color2)  # we already handled the x-label with ax1
    ln2 = ax2.plot(t, data2, color = color2, label = dataLabel2)
    ax2.set_ylim([0,100])
    ax2.tick_params(axis='y', labelcolor = color2)
    # ax2.legend(loc="upper right", bbox_to_anchor=(1, 0.90))

    #GOD TIER LEGEND HELPER GOD BLESS
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper right")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(filename)
    print(filename + " successfully uploaded.")

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

def scatterPlot(mag, per, title, file, xlabel):
    font = {'fontname':'Helvetica', 'style':'italic'}
    plt.clf()
    sb.regplot(mag, per, 'o')
    plt.title(title, **font)
    plt.xlabel(xlabel, **font)
    plt.ylabel("Population Density %", **font)
    plt.grid(color='grey', linestyle = "--", linewidth=0.25, alpha=0.75) 
    #plt.text(0.1, 0.9, 'text', size=15, color='purple')
    # m, b = np.polyfit(mag, per, 1)
    # plt.plot(mag, m(mag) + b)


    plt.savefig(file)

# Data for plotting
def plotter(smallData, largeData, start, finish, pltr, label):
    for i in range(start, finish):
        initStart = 12 * i
        initEnd = initStart + 11
        pltr.plot(np.arange(1,13), (smallData.loc[initStart:initEnd]['count'] / (largeData.loc[initStart:initEnd]['count'] + 0.01)) * 100, 'C' + str(i), label = str(2000 + i))
        pltr.set_title(label)
        pltr.set_xlim(1,12)
        pltr.set_ylim(0,100)
