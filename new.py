import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# rawMagData = "3060fullmag.csv"
# filteredMagData = pd.read_csv(rawMagData)
# print(filteredMagData)

# data = pd.read_csv(r'plovercsv.csv', skiprows = 138000, nrows = 100)
# data.to_csv('b.csv')

# data = {
#   "calories": [420, 380, 390],
#   "duration": [50, 40, 45],
#   "cereal name": ['a', 'b', 'c']
# }

# #load data into a DataFrame object:
# df = pd.DataFrame(data)

# dg = pd.DataFrame([[53335, 4944, 'b']], columns = ['calories', 'duration', 'cereal name'])
# print(df)
# print(dg)

# cerealName = str(dg['cereal name'][0])
# print(cerealName)
# for x in np.arange(0,2):
#     y = df[df['cereal name'] == cerealName].index[0]
#     print(y)
#     df.iat[y,x] = df.iloc[y,x] + dg.iloc[0][x]

# print(df)

# data = [[1, 2, 2], [1, 2, 1], [1, 1, 1]]
# average = [sum(x)/len(x) for x in zip(*data)]
# print(average)

kps = pd.read_csv(r"kpindex.csv")
kps = kps.query('YYY >= 2000').copy()
print(kps)
li = []
for x in np.arange(2000,2020,1):
  l = [x]
  for y in np.arange(0, 7,1):
    l.append((kps.query('@y <= Kp < @y + 1 & YYY == @x'))['Kp'].sum())
    print(l)
  val = x - 2000
  li.append(l)
df = pd.DataFrame(li, columns = ['Year', '0 - 1', '1 - 2', '2 - 3', '3 - 4', '4 - 5', '5 - 6', '6 - 7'])

plt.plot(df["Year"], df["6 - 7"], c = "red")
plt.plot(df["Year"], df["5 - 6"], c = "blue")
plt.plot(df["Year"], df["4 - 5"], c = "green")
plt.show()

