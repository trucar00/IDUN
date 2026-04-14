import pandas as pd

df = pd.read_csv("../../Data-preMaster/names.csv")

print(df.head())

df.to_csv("saved.csv")