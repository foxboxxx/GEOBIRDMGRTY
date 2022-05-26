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
# import pygbif
# from pygbif import species as species
# from pygbif import occurrences as occ

# !---> EBIRD TESTING
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

matchesAdv = []
for inst in matches:
    instReq = requests.request("GET", "https://api.ebird.org/v2/ref/taxonomy/ebird?species={}&fmt=json".format(inst), headers = headers, data = payload)
    # print(instReq.text)
    # matchesAdv.append()
    tempL = dict(((instReq.text)[2:(len)(instReq.text) - 2]).split(","))
    print(tempL)





# !---> GBIF DATA TESTING
response = requests.request("GET", urlRecents, headers = headers, data = payload)

req = requests.get('https://api.gbif.org/v1/species?name=Pluvialis dominica&limit=100')
# req = requests.get('https://api.gbif.org/v1/occurrence/download/statistics&publishingOrgKey=4fa7b334-ce0d-4e88-aaae-2e0c138d049e')
# print(req.status_code)
# print(req.json())
with open('gbifdata.json', 'w') as outfile:
    json.dump(req.json(), outfile)

# speciesList = requests.get('https://api.gbif.org/v1/dataset/search?country=AR')
# print("list\n")
# print(speciesList.json())