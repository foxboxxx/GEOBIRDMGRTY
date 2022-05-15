import numpy as np
import pandas as pd

l1 = [1,2,4]
l2 = ["ONE", "TWO", "THREE",]

result = zip(l1,l2)
result_set = set(result)
print(result_set)