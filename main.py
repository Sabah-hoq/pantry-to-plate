# download https://recipenlg.cs.put.poznan.pl/ and convert to recipes.parquet

import pandas as pd
import pyarrow
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet("recipes.parquet")
df["NER"] = df["NER"].apply(lambda x: list(map(lambda y : y[1:], x[1:-2].upper().split("\", "))))

def get_all_ingredients(df):
    np.save("ingredients.npy", np.concat(df["NER"], dtype=object))
    return df["NER"]

#ingredients = get_all_ingredients(df)

def get_all_categories():
    ingredients = np.load("ingredients.npy", allow_pickle=True)
    values, counts = np.unique_counts(ingredients)

    mask = np.fromiter(map(len, values), dtype=int) > 2
    values, counts = values[mask], counts[mask]

    percent = counts / len(ingredients)
    mask = percent > 0.0001
    values, counts = values[mask][:-2], counts[mask][:-2]
    values = values.view(object)

    np.save("categories.npy", values)

    return values

#categories = get_all_categories()

categories = np.load("categories.npy", allow_pickle=True)




