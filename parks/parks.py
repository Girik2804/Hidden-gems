import pandas as pd
import ast
import numpy as np

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

def get_area(lat, lon):
    """Classify park into one of 15 Toronto regions or 'other'"""
    for area, box in bounding_boxes.items():
        if (box["lat_min"] <= lat <= box["lat_max"] and 
            box["lon_min"] <= lon <= box["lon_max"]):
            return area
    return "other"

def parse_coords(geometry_json):
    """Extract lat/lon from geometry column"""
    try:
        geometry = ast.literal_eval(geometry_json)
        coords = geometry.get("coordinates")[0]
        lon, lat = coords
        return lat, lon
    except Exception:
        return None, None

def is_central_location(lat, lon):
    """Check if park is in urban/central area (+2 weight)"""
    central_areas = ["downtown", "kensington", "chinatown", "queenwest", 
                    "trinitybellwoods", "harbourfront", "waterfront", "annex", "bloor"]
    area = get_area(lat, lon)
    return area in central_areas

def amenity_count(amenities):
    """Count number of amenities"""
    if not amenities or pd.isna(amenities) or str(amenities).strip() == '':
        return 0
    return len([a.strip() for a in str(amenities).split(',') if a.strip()])

def calculate_isolation_score(lat, lon, all_coords):
    """Calculate park isolation based on distance to nearest park (+1 weight)"""
    if pd.isna(lat) or pd.isna(lon):
        return 0
    
    min_distance = float('inf')
    for other_lat, other_lon in all_coords:
        if other_lat != lat or other_lon != lon:  # Skip same park
            distance = np.sqrt((lat - other_lat)**2 + (lon - other_lon)**2)
            min_distance = min(min_distance, distance)
    
    # Normalize: more isolated = higher score (max 0.2 for +1 weight)
    if min_distance == float('inf'):
        return 0.1  # Single park case
    return min(min_distance * 100, 0.2)  # Scale appropriately

def calculate_hidden_gem_score(row, all_coords):
    """
    Calculate Hidden Gem Score using your weighted criteria:
    - Amenity Count (+2 weight): More amenities = better
    - Missing Metadata (+1 weight): Less metadata = more hidden  
    - Park Isolation (+1 weight): More isolation = more hidden
    - Urban Location (+2 weight): More central = more likely hidden
    - Low Online Presence (+2 weight): Less search hits = more hidden
    """
    score = 0.0

    # 1. Amenity Count (+2 weight) - max 0.4 points
    count = amenity_count(row.get('AMENITIES', ''))
    score += min(count * 0.1, 0.4)

    # 2. Missing Metadata (+1 weight) - max 0.2 points
    missing_metadata = 0
    phone = row.get('PHONE', '')
    url = row.get('URL', '')
    
    if not phone or pd.isna(phone) or str(phone).strip() == '' or str(phone) == 'None':
        missing_metadata += 1
    if not url or pd.isna(url) or str(url).strip() == '' or str(url) == 'None':
        missing_metadata += 1
    
    score += min(missing_metadata * 0.1, 0.2)

    # 3. Urban Location (+2 weight) - max 0.4 points
    if is_central_location(row['Latitude'], row['Longitude']):
        score += 0.4

    # 4. Park Isolation (+1 weight) - max 0.2 points
    isolation = calculate_isolation_score(row['Latitude'], row['Longitude'], all_coords)
    score += isolation

    # 5. Low Online Presence (+2 weight) - max 0.4 points
    # Using missing metadata as proxy for low online presence
    if missing_metadata >= 2:  # Both phone and URL missing
        score += 0.4
    elif missing_metadata == 1:
        score += 0.2

    return min(score, 1.0)

# Read your CSV
df = pd.read_csv("parks_and_rec.csv")

# Extract coordinates from geometry
df[["Latitude", "Longitude"]] = df["geometry"].apply(
    lambda g: pd.Series(parse_coords(g))
)

# Get all coordinates for isolation calculation
all_coords = [(row['Latitude'], row['Longitude']) 
              for _, row in df.iterrows() if pd.notna(row['Latitude'])]

# Area classification
df["area"] = df.apply(
    lambda row: get_area(row["Latitude"], row["Longitude"]), axis=1
)

# Hidden Gem Score calculation using your methodology
df["score"] = df.apply(
    lambda row: calculate_hidden_gem_score(row, all_coords), axis=1
)

# Output results
df.to_csv("parks.csv", index=False)
print("Done! Output written to parks.csv")
print(f"\nScore distribution:")
print(f"Mean score: {df['score'].mean():.3f}")
print(f"Max score: {df['score'].max():.3f}")
print(f"Min score: {df['score'].min():.3f}")
