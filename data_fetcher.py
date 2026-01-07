
import os
import requests
import pandas as pd
from tqdm import tqdm

API_KEY = "YOUR_API_KEY"
ZOOM =18
SIZE = "224x224"

def fetch_image(lat,lon,idx,save_dir = "data/images"):
     url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&zoom={ZOOM}&size={SIZE}"
        f"&maptype=satellite&key={API_KEY}"
    )
     r = requests.get(url)
     if r.status_code ==200:
          with open(f"{save_dir}/{idx}.png", "wb") as fr:
               fr.write(r.content)

def main():
     os.makedirs("data/images", exist_ok=True)
     df = pd.read_excel("data/raw/train(1).xlsx")

     for idx, row in tqdm(df.iterrows(), total=len(df)):
        fetch_image(row["lat"], row["long"], idx)

if __name__ == "__main__":
    main()