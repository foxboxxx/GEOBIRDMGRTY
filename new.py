import pandas as pd
rawMagData = "3060fullmag.csv"
filteredMagData = pd.read_csv(rawMagData)
print(filteredMagData)