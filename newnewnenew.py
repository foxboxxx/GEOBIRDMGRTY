import pandas as pd

df = pd.DataFrame({'Animal': [50, 50, 50, 51, 51], 'Max Speed': [42, 42, 43, 65, 65], 'INv': [100,100,100,100,100]})

print(df)

d2 = df.groupby(['Animal','Max Speed']).sum()

print(df)
print(d2)

print(d2.iloc[1])