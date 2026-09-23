from cProfile import label
import json
import os
from pickletools import optimize
from pydoc import text
import re
import pickle
from threading import Thread
from turtle import mode, shape, st
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from app import VECTORIZER_FILE
from neural_network import ThreatNetwork
DATA_FILE="data/training.json"
MODEL_FILE="model/model.pth"
VECTORIZER_FILE="model/vectorizer.pkl"
def load_data():
    if not os.path.exists(DATA_FILE):
        return [], []
    with open(DATA_FILE, "r", encoding="utf-8")as file:
        data=json.load(file)
        text=[]
        labels=[]
        for item in data:
            text.append(item["text"])
            label= str(item.get("label", "")).lower()
            if label in ["threat", "thread"]:
                labels.append(1)
            else:
                labels.append(0)
        return text, labels
def train():
    text, labels =load_data()
    if len(text)<10:
        print("Tarvitaan vähintään 10 tarkistuttua esimerkkiä!!")
        return
    vectorizer=TfidfVectorizer(lowercase=True, max_features=3000)
    x = vectorizer.fit_transform(text).toarray()
    x=torch.tensor(x, dtype=torch.float32)
    y=torch.tensor(labels, dtype=torch.long)
    model=ThreatNetwork(x.shape[1])
    criterion= nn.CrossEntropyLoss()
    optimizer= torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(100):
        output=model(x)
        loss=criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 ==0:
            print(f"{epoch}/100 Loss: {loss.item():4f}")
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(),
               MODEL_FILE)
    with open(VECTORIZER_FILE, "wb")as file:
        pickle.dump(vectorizer, file)
    print("malli koulutettu")
if __name__ == "__main__":
    train()