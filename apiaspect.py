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
import io
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

def match_species(spList, 
                  nameCol = "scientificName", 
                  speciesApi = "http://api.gbif.org/v1/species/match?verbose=true&name="):
    matched_species = []
    for species in spList.index:
        name = spList.loc[species, nameCol]
        match = requests.get(speciesApi + name)
        print(match)
        
        if match.ok:
            match_result = match.json()
            match_result['inputName'] = spList.loc[species, nameCol]
                    
        if "alternatives" in match_result:
                match_result["has_alternatives"] = True
                for alt in match_result["alternatives"]:
                    alt["inputName"] = spList.loc[species, nameCol]
                    alt["is_alternative"] = True
                    matched_species.append(alt)  # add alternative
                match_result.pop('alternatives')

        matched_species.append(match_result)
    result = pd.DataFrame(matched_species)
    taxon_keys = ['acceptedUsageKey', 'usageKey', 'kingdomKey', 'phylumKey','classKey', 'orderKey', 'familyKey', 'genusKey', 'speciesKey']
    result[taxon_keys] = result[taxon_keys].fillna(0).astype(int)
    result = result.fillna("NULL")
    return result

def create_download_given_query(username, password, download_query, api = "http://api.gbif.org/v1/"):
    headers = {'Content-Type': 'application/json'}
    download_request = requests.post(api + "occurrence/download/request", data = json.dumps(download_query), auth = (username, password), headers = headers)
    if download_request.ok:
        print("Request Good")
    else:
        print(download_request)
        print("request bad")
    return download_request


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

# identifying the scientific names of the birds that I indentified
sciNameList = []
matchesAdv = []
for inst in matches:
    instReq = requests.request("GET", "https://api.ebird.org/v2/ref/taxonomy/ebird?species={}&fmt=json".format(inst), headers = headers, data = payload)
    # print(instReq.text)
    # matchesAdv.append()
    tempL = ((instReq.text)[2:(len)(instReq.text) - 2]).split(",")
    sciName = tempL[0]; sciName = sciName[10:].strip("\"")
    sciNameList.append(sciName)
    print(sciName)

# Testing specific species for whether or not the API works
findingDatasetTest = requests.request("GET", 'https://api.gbif.org/v1/dataset/search?speciesKey=2481756', headers = headers, data = payload)
# print("2", findingDatasetTest.text)
# with open('doi.json', 'w') as outfile:
#     json.dump(findingDatasetTest.json(), outfile)

# Finds the DOi of the requested species using the link below
findingDOI = requests.request("GET", "https://api.gbif.org/v1/occurrence/download/dataset/d7dddbf4-2cf0-4f39-9b2a-bb099caae36c")
# print("3", findingDOI.text)
with open('doi.json', 'w') as outfile:
    json.dump(findingDOI.json(), outfile)

# !---> GBIF DATA TESTING

# print('https://api.gbif.org/v1/species/match?name={}&limit=10'.format(sciNameList[0]))
# req = requests.get('https://api.gbif.org/v1/species/match?name={}&limit=10'.format(sciNameList[0]))
# with open('gbifplaceholder.json', 'w') as outfile:
#     json.dump(req.json(), outfile)  

username = 'joerob'
password = 'jr90050253'

spList = pd.read_csv('scinames.csv', encoding = 'latin-1')
print(spList["Scientific Names"])
print(spList.index)
taxon_keys = match_species(spList, "Scientific Names")
print("done")

key_list = taxon_keys.loc[(taxon_keys["matchType"]=="EXACT") & (taxon_keys["status"]=="ACCEPTED")].usageKey.tolist()

download_query = {}
download_query["creator"] = "joerob"
download_query["notificationAddresses"] = ["robertazzijoseph@gmail.com"]
download_query["sendNotification"] = False # if set to be True, don't forget to add a notificationAddresses above
download_query["format"] = "SIMPLE_CSV"
download_query["predicate"] = {
    "type": "in",
    "key": "TAXON_KEY",
    "values": key_list
}

create_download_given_query(username, password, download_query)


# download_query = {["creator"] : "", ["notificationAddresses"] : [""], ["sendNotification"] : False, ["format"] : "SIMPLE_CSV", ["predicate"] : {
#     "type": "in",
#     "key": "TAXON_KEY",
#     "values": key_list
# }}


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