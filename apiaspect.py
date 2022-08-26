import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import statsmodels.graphics.api as smg
import statsmodels.api as sm
from scipy import stats
import seaborn as sb
import requests
import json
# from sorting import lineBreak

# import pygbif
# from pygbif import species as species
# from pygbif import occurrences as occ

# !---> EBIRD TESTING

# print(np.random.rand(10, 12))
# a = np.random.rand(180, 360)
# f, ax = plt.subplots(figsize=(15, 15))
# ax = sb.heatmap(a, linewidths=.5, xticklabels = 1, yticklabels = 1)
# plt.show()

region = 'AR'
token = 'a83ao4aendbm'
urlRecents = "https://api.ebird.org/v2/data/obs/{}/recent".format(region)
urlSpeciesAR = 'https://api.ebird.org/v2/product/spplist/AR'
urlSpeciesCA =  'https://api.ebird.org/v2/product/spplist/CA'
payload={}
headers = {
  'X-eBirdApiToken': '{}'.format(token)
}
ebird = requests.request("GET", urlRecents, headers = headers, data = payload)
ebirdSpeciesAR = requests.request("GET", urlSpeciesAR, headers = headers, data = payload)
ebirdSpeciesCA = requests.request("GET", urlSpeciesCA, headers = headers, data = payload)

# print(ebird.text)
# print(ebirdSpecies.text)
with open('jsonapitest.json', 'w') as outfile:
    outfile.write(ebird.text)
with open('initialstores.json', 'w') as output:
    output.write(ebirdSpeciesAR.text)

# !---> CONVERTING JSON REQUEST DATA TO LIST
print("Argentina:\n")
sightingsListAR = ((ebirdSpeciesAR.text)[1:(len)(ebirdSpeciesAR.text) - 1]).split(",")
sightingsListAR = [x[1:(len)(x) - 1] for x in sightingsListAR] 
print(sightingsListAR)

print("Canada:\n")
sightingsListCA = ((ebirdSpeciesCA.text)[1:(len)(ebirdSpeciesCA.text) - 1]).split(",")
sightingsListCA = [x[1:(len)(x) - 1] for x in sightingsListCA] 
print(sightingsListCA)

matches = []
for x in sightingsListAR: 
    for y in sightingsListCA: 
        if x == y: matches.append(x)
print("Matches:\n")
print(matches)
print("\n")

sciNameList = []
matchesAdv = []
for inst in matches:
    instReq = requests.request("GET", "https://api.ebird.org/v2/ref/taxonomy/ebird?species={}&fmt=json".format(inst), headers = headers, data = payload)
    # print(instReq.text)
    # matchesAdv.append()
    tempL = ((instReq.text)[2:(len)(instReq.text) - 2]).split(",")
    sciName = tempL[0]; sciName = sciName[10:].strip("\"")
    sciNameList.append(sciName)
    print( sciName)

findingDatasetTest = requests.request("GET", 'https://api.gbif.org/v1/dataset/search?speciesKey=2481756', headers = headers, data = payload)
# print("2", findingDatasetTest.text)
# with open('doi.json', 'w') as outfile:
#     json.dump(findingDatasetTest.json(), outfile)

findingDOI = requests.request("GET", "https://api.gbif.org/v1/occurrence/download/dataset/d7dddbf4-2cf0-4f39-9b2a-bb099caae36c")
# print("3", findingDOI.text)
with open('doi.json', 'w') as outfile:
    json.dump(findingDOI.json(), outfile)

# !---> GBIF DATA TESTING

# Secondary Testing:

# dfSciList = pd.DataFrame(index = sciNameList,  columns =['Scientific Names'])
# dfSciList.to_csv('sciNames.csv',)
# print(dfSciList)

username = 'joerob'
password = 'jr90050253'

spList = pd.read_csv('scinames.csv', encoding = 'latin-1')

print('https://api.gbif.org/v1/species/match?name={}&limit=10'.format(sciNameList[0]))
req = requests.get('https://api.gbif.org/v1/species/match?name={}&limit=10'.format(sciNameList[0]))
with open('gbifplaceholder.json', 'w') as outfile:
    json.dump(req.json(), outfile)

taxon_keys = ['acceptedUsageKey', 'usageKey', 'kingdomKey', 'phylumKey','classKey', 'orderKey', 'familyKey', 'genusKey', 'speciesKey']
download_query = {["creator"] : "", ["notificationAddresses"] : [""], ["sendNotification"] : False, ["format"] : "SIMPLE_CSV", ["predicate"] : {
    "type": "in",
    "key": "TAXON_KEY",
    "values": key_list
}}


# <------------->




exit()
response = requests.request("GET", urlRecents, headers = headers, data = payload)

req = requests.get('https://api.gbif.org/v1/species/match?name=Pluvialis dominica&limit=100')
#initial
# req = requests.get('https://api.gbif.org/v1/occurrence/download/statistics&publishingOrgKey=4fa7b334-ce0d-4e88-aaae-2e0c138d049e')
#to file--->
with open('gbifdata.json', 'w') as outfile:
    json.dump(req.json(), outfile)

secondTest = requests.get('https://api.gbif.org/v1/occurrence/search?name=Pluvialis dominica&publishingOrgKey=4fa7b334-ce0d-4e88-aaae-2e0c138d049e&limit=1000')
print(secondTest.text)
# with open('gbifdata.json', 'w') as outfile:
#     json.dump(secondTest.json(), outfile)

agbirdcsv = pd.read_json(secondTest.json()['results'])
print(agbirdcsv)

agbirdcsv.to_csv('apirequest.csv', sep='\t')



# speciesList = requests.get('https://api.gbif.org/v1/dataset/search?country=AR')
# print("list\n")
# print(speciesList.json())

# 10.15468/aomfnb
#d7dddbf4-2cf0-4f39-9b2a-bb099caae36c