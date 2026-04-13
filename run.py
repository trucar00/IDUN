import pandas as pd

names = ["tom", "jens", "oskar"]
ages = [1, 2, 3]
d = {"name": names, "age": ages}

df = pd.DataFrame(d)
print("saving")

df.to_csv("names.csv")
print("saved")