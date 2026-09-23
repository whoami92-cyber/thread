from email.mime import text
from turtle import title
import json
import subprocess
import os
import pickle
from xml.parsers.expat import model
from xml.etree.ElementTree import fromstring
from feedparser import urls
import torch
import urllib
import xml.etree.ElementTree as ET
from collector import collect_articles, collect_articles
from neural_network import ThreatNetwork
DATA_FILE="data/training.json"
MODEL_FILE="model/model.pth"
VECTORIZER_FILE="model/vectorizer.pkl"
def save_and_retrain(text, label):
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)
        data.append({"text": text, "label": label})
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        subprocess.run(["python3", "trainer.py"])
        print("Malli on koulutettu uudelleen ja tallennettu.")
def fetch_latest_news():
    rss_feed_urls =[
        "https://www.iltalehti.fi/rss/uutiset.xml",
        "https://www.yle.fi/rss/uutiset.rss",
        "https://www.mtv.fi/rss/uutiset.xml",
        "https://www.is.fi/rss/tuoreimmat.xml"]
    headlines = []
    for u in rss_feed_urls:
        req=urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                xml_data = response.read()
                root=ET.fromstring(xml_data)
                headlines = []
            for item in root.findall('.//item'):
                title = item.find("title").text
                link = item.find("link").text
                headlines.append(title)
        except Exception as e:
            print(f"Virhe haettaessa uutisia: {e}")
        return headlines
def save_example(text, label):
    os.makedirs("data", exist_ok=True)
    data = []
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
    data.append({"text": text, "label": label})
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
def main():
    if not os.path.exists(DATA_FILE):
        print("virhe, dataa ei löydy, kerää ensin dataa")
        exit()
        with open(VECTORIZER_FILE, "rb") as f:
            vectorizer = pickle.load(f)
            input_size = len(vectorizer.get_feature_names_out())
            model = ThreatNetwork(input_size)
            model.load_state_dict(torch.load(MODEL_FILE))
            model.eval()
    print("""  ===================
                   THREAD AI
               ====================
               BUPLIC INTERNET DATA
               ====================
               """)
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
        print("virhe mallia ei löydy, kouluta ensin malli")
        exit()
    with open (VECTORIZER_FILE, "rb") as f:
        vectorizer = pickle.load(f)
        input_size = len(vectorizer.get_feature_names_out())
        model = ThreatNetwork(input_size)
        model.load_state_dict(torch.load(MODEL_FILE))
        model.eval() 
        print("AI malli ladattu onnistuneesti.")
        headlines=fetch_latest_news()
        print("Viimeisimmät uutiset:")
        for item in headlines:
            print(f"- {item[0]}")
            if isinstance(item, tuple):
                title, link = item
            else:
                title= str(item)
            print(f"- {title}")
            x = vectorizer.transform([title]).toarray()
            x_tensor = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                output = model(x_tensor)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item() * 100
                if prediction == 1:
                    print(f"AI arvioi tekstin olevan UHKA, luottamus: {confidence:.2f}%")
                else:
                    print(f"AI arvioi tekstin olevan NORMAALI, luottamus: {confidence:.2f}%")
                    if confidence < 65:
                        print("Luottamus on alhainen, harkitse tarkistusta.")
                feedback = input("oliko arvio oikein(k/e)? ").strip().lower()
                if feedback == 'k':
                    continue
                if feedback=="e":
                    correct_choice=input("Anna oikea luokitus (1 = THREAD, 0 = NORMAL): ").strip().upper()
                if correct_choice == '1' or correct_choice.upper()=='THREAD':
                    correct_label="THREAD"
                else:
                    correct_label="NORMAL"
                save_example(title, correct_label)
                with open(VECTORIZER_FILE, "rb") as f:
                    vectorizer = pickle.load(f)
                    input_size = len(vectorizer.get_feature_names_out())
                    model = ThreatNetwork(input_size)
                    model.load_state_dict(torch.load(MODEL_FILE))
                    model.eval()
                    continue
                text=input("Anna teksti analysoitavaksi (tai kirjoita 'q' lopettaaksesi): ")
                if text.lower() == "q":
                    break
                if not text.strip():
                    continue
if __name__ == "__main__":
     main ()