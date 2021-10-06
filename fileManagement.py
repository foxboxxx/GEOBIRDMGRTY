import pandas as pd
import numpy as np
import os
import pygmt
import csv

#Enter the csv file after r and ""
df = pd.read_csv(r"forktailedUnfiltered.csv", encoding='latin1')
print(df)
controlLocation = df[ (df['decimalLatitude'] >= 0) & ((df['decimalLongitude'] <= -90) & df['decimalLongitude'] >= -30)].index
df.drop(controlLocation, inplace = True)
preciseBirdData = df.loc[df['occurrenceStatus']=='PRESENT',['eventDate','individualCount','decimalLatitude','decimalLongitude','day','month','year']].copy()
interest = preciseBirdData.query("year >= 1970")
interest = interest.query("individualCount <= 10000")

interest.to_csv('forktailedFiltered.csv')
print("Done!")