# download https://recipenlg.cs.put.poznan.pl/ and convert to recipes.parquet

import pandas as pd
import pyarrow
import numpy as np
from special import spiceData, spiceModel
import matplotlib.pyplot as plt
from sympy.geometry.entity import translate
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm

# Uncomment lines below to get .parquet file
# df = pd.read_csv('recipes.csv', low_memory=False)
# df.to_parquet('recipes.parquet', index=False)

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
category_dict = dict(zip(categories, range(len(categories))))

df["NER"] = df["NER"].apply(lambda x: list(map(lambda y: category_dict[y] if y in category_dict else -1, x)))
df["compromised"] = df["NER"].apply(lambda x: (np.array(x) == -1).mean())
df = df[(df["compromised"] < 0.2).values]

train_data = np.zeros((len(df), 790),dtype=bool)

train_data[np.arange(len(df)).repeat(df["NER"].apply(len).values), np.concat(df["NER"].values, dtype=int)] = True

train_data = train_data[:, :-1]
np.random.shuffle(train_data)

index_max = int(len(train_data) * 0.75)
train = spiceData(train_data[:index_max], batch_size=512, train=True)
test = spiceData(train_data[index_max:], batch_size=512, train=False)

model = spiceModel()

def print_similar_categories(trained_model):
    trained_model.load_state_dict(torch.load("model_weights.pt"))
    trained_model.eval()

    total = nn.functional.softmax(trained_model(torch.eye(789,dtype=torch.float32)), dim=1)

    sorted_arg = torch.argsort(total, dim=1, descending=True)
    sorted_output = torch.sort(total, dim=1, descending=True)[0][:, :10]

    similar_categories = categories[torch.flatten(sorted_arg[:, :10])].reshape(-1, 10)

    print(sorted_output)
    print(similar_categories)

#print_similar_categories(model)
#STOP

loss_funct = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

for epoch in range(50):
    print("epoch:", epoch)
    train_loss = 0
    test_loss = 0
    model.train()
    for x, y in tqdm(train):
        optimizer.zero_grad()

        pred = model(x)
        loss = loss_funct(pred, y)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
    print("train loss:", train_loss / train.__len__())

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(test):
            pred = model(x)
            loss = loss_funct(pred, y)
            test_loss += loss.item()
    print("test loss:", test_loss / test.__len__())

    train.on_epoch_end()
    test.on_epoch_end()
    scheduler.step()
    print()

torch.save(model.state_dict(), 'model_weights.pt')





