import numpy as np
import pandas as pd

thing = pd.DataFrame({'year':[]})
print(thing)
for i in range(0,2):
    for i in range(0,3):
        thing = thing.append(pd.DataFrame({'year': i}, index = [0]) , ignore_index = True)

print(thing) 
    
