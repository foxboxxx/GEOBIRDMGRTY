# answer1 = input("Hello! Would you like to play this fun game? (Y/N): ")
# if(answer1.lower() != "y"):
#     print("That's too bad, hope to see you soon!")
#     exit()
# age = int(input("What is your age?: "))
# print("You will turn 100 in " + str(2021 + (100 - age)) + "!")

# number = int(input("Please enter a number: "))
# if number % 2 == 0:
#     print(str(number) + " is even!")
# else:
#     print(str(number) + " is odd!")

# a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
# b = []
# for x in a:
#     if x < 5:
#        b.append(x)
# print(b) 

# number = int(input("Please input a random number: "))
# x = 1
# a = []
# for x in range(1, number + 1, 1):
#     if number % x == 0:
#         a.append(x)
# print(a)
import numpy as np
import pandas as pd

thing = pd.DataFrame({'year':[]})
print(thing)
for i in range(0,2):
    for i in range(0,3):
        thing = thing.append(pd.DataFrame({'year': i}, index = [0]) , ignore_index = True)

print(thing) 
    
