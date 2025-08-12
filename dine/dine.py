import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Step 1: Bounding boxes for each area (lat_min, lat_max, lon_min, lon_max) ---
bounding_boxes = {
    # Downtown core, roughly University to Jarvis, Queen to the Gardiner
    "downtown": {
        "lat_min": 43.6410, "lat_max": 43.6580,
        "lon_min": -79.3900, "lon_max": -79.3750
    },
    # Bounded by Bathurst, Spadina, College, and Dundas
    "kensington": {
        "lat_min": 43.6530, "lat_max": 43.6595,
        "lon_min": -79.4100, "lon_max": -79.3970
    },
    # Centered on Spadina, from Queen St to College St
    "chinatown": {
        "lat_min": 43.6500, "lat_max": 43.6580,
        "lon_min": -79.4010, "lon_max": -79.3950
    },
    # The main strip from University Ave to Bathurst St
    "queenwest": {
        "lat_min": 43.6470, "lat_max": 43.6510,
        "lon_min": -79.4110, "lon_max": -79.3870
    },
    # The area surrounding Trinity Bellwoods Park
    "trinitybellwoods": {
        "lat_min": 43.6460, "lat_max": 43.6545,
        "lon_min": -79.4180, "lon_max": -79.4090
    },
    # The historic, pedestrian-only district
    "distillery": {
        "lat_min": 43.6480, "lat_max": 43.6515,
        "lon_min": -79.3615, "lon_max": -79.3550
    },
    # More accurately Lawrence Park, east of Yonge St
    "lawrence": {
        "lat_min": 43.7180, "lat_max": 43.7350,
        "lon_min": -79.3980, "lon_max": -79.3780
    },
    # Area along the lake, south of the Gardiner Expressway
    "harbourfront": {
        "lat_min": 43.6360, "lat_max": 43.6420,
        "lon_min": -79.4000, "lon_max": -79.3720
    },
    # Broader waterfront area including St. Lawrence Market
    "waterfront": {
        "lat_min": 43.6370, "lat_max": 43.6510,
        "lon_min": -79.3810, "lon_max": -79.3610
    },
    # Bounded by Avenue Rd, Bathurst, Bloor, and Dupont
    "annex": {
        "lat_min": 43.6650, "lat_max": 43.6760,
        "lon_min": -79.4150, "lon_max": -79.3950
    },
    # The Bloor-Yorkville corridor
    "bloor": {
        "lat_min": 43.6640, "lat_max": 43.6720,
        "lon_min": -79.4050, "lon_max": -79.3850
    },
    # East end, along Queen St East
    "leslieville": {
        "lat_min": 43.6580, "lat_max": 43.6680,
        "lon_min": -79.3500, "lon_max": -79.3270
    },
    # Between Leslieville and the Don Valley Parkway
    "riverside": {
        "lat_min": 43.6560, "lat_max": 43.6640,
        "lon_min": -79.3600, "lon_max": -79.3410
    },
    # Centered on College St, from Bathurst to Ossington
    "littleitaly": {
        "lat_min": 43.6540, "lat_max": 43.6600,
        "lon_min": -79.4240, "lon_max": -79.4090
    },
    # Hub around Dundas St W and Lansdowne
    "dundaswest": {
        "lat_min": 43.6500, "lat_max": 43.6590,
        "lon_min": -79.4420, "lon_max": -79.4300
    },
}

# --- Step 2: Area classification function ---
def get_area(lat, lon):
    for area, box in bounding_boxes.items():
        if (
            box["lat_min"] <= lat <= box["lat_max"] and
            box["lon_min"] <= lon <= box["lon_max"]
        ):
            return area
    return "other"

# --- Step 3: Sentiment scoring ---
analyzer = SentimentIntensityAnalyzer()
def get_sentiment_score(text):
    if not isinstance(text, str) or not text.strip():
        return 0.5  # Neutral if text is missing
    vs = analyzer.polarity_scores(text)
    score = (vs["compound"] + 1) / 2  # Convert [-1, 1] to [0, 1]
    return round(score, 3)

# --- Step 4: Apply to your CSV ---
df = pd.read_csv("Dinesafe (2).csv")   # Change to your actual filename
df["area"] = df.apply(
    lambda row: get_area(row["Latitude"], row["Longitude"]), axis=1
)
df["score"] = df["Infraction Details"].apply(get_sentiment_score)

df.to_csv("dine.csv", index=False)
print("Done! Output written to dine.csv .")
