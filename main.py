import pandas as pd
import pyarrow
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet("recipes.parquet")

def get_all_ingredients(df):
    df["NER"] = df["NER"].apply(lambda x: list(map(lambda y : y[1:], x[1:-2].upper().split("\", "))))
    np.save("ingredients.npy", np.concat(df["NER"]))

get_all_ingredients(df)

ingredient = np.load("ingredients.npy", allow_pickle=True)
values, counts = np.unique_counts(ingredient)

mask = np.fromiter(map(len, values), dtype=int) > 2
values, counts = values[mask], counts[mask]

percent = counts / len(ingredient)
mask = percent > 0.0001
values, counts = values[mask][:-2], counts[mask][:-2]
print(len(values))


