import pandas as pd
import numpy as np

bounding_boxes = {
    "downtown": {
        "lat_min": 43.6463, "lat_max": 43.6555,
        "lon_min": -79.3897, "lon_max": -79.3760
    },
    "kensington": {
        "lat_min": 43.6557, "lat_max": 43.6621,
        "lon_min": -79.4073, "lon_max": -79.3995
    },
    "chinatown": {
        "lat_min": 43.6519, "lat_max": 43.6572,
        "lon_min": -79.4037, "lon_max": -79.3963
    },
    "queenwest": {
        "lat_min": 43.6489, "lat_max": 43.6550,
        "lon_min": -79.4128, "lon_max": -79.3912
    },
    "trinitybellwoods": {
        "lat_min": 43.6464, "lat_max": 43.6541,
        "lon_min": -79.4176, "lon_max": -79.4095
    },
    "distillery": {
        "lat_min": 43.6482, "lat_max": 43.6511,
        "lon_min": -79.3612, "lon_max": -79.3551
    },
    "lawrence": {
        "lat_min": 43.7294, "lat_max": 43.7441,
        "lon_min": -79.4268, "lon_max": -79.4057
    },
    "harbourfront": {
        "lat_min": 43.6338, "lat_max": 43.6418,
        "lon_min": -79.3907, "lon_max": -79.3763
    },
    "waterfront": {
        "lat_min": 43.6365, "lat_max": 43.6505,
        "lon_min": -79.3805, "lon_max": -79.3620
    },
    "annex": {
        "lat_min": 43.6662, "lat_max": 43.6728,
        "lon_min": -79.4144, "lon_max": -79.4012
    },
    "bloor": {  # Bloor Corridor
        "lat_min": 43.6616, "lat_max": 43.6707,
        "lon_min": -79.4052, "lon_max": -79.3886
    },
    "leslieville": {
        "lat_min": 43.6596, "lat_max": 43.6676,
        "lon_min": -79.3461, "lon_max": -79.3273
    },
    "riverside": {
        "lat_min": 43.6568, "lat_max": 43.6635,
        "lon_min": -79.3597, "lon_max": -79.3414
    },
    "littleitaly": {
        "lat_min": 43.6535, "lat_max": 43.6600,
        "lon_min": -79.4222, "lon_max": -79.4071
    },
    "dundaswest": {
        "lat_min": 43.6509, "lat_max": 43.6590,
        "lon_min": -79.4411, "lon_max": -79.4302
    },
}

def get_area(lat, lon):
    """Classify library location into one of 15 Toronto regions"""
    if pd.isna(lat) or pd.isna(lon):
        return "other"
    
    for area, box in bounding_boxes.items():
        if (box["lat_min"] <= lat <= box["lat_max"] and 
            box["lon_min"] <= lon <= box["lon_max"]):
            return area
    return "other"

def calculate_community_hub_score(row):
    """
    Calculate Community Hub Score (0-1) based on multiple factors:
    - Service Diversity (30%): Programs and services offered
    - Accessibility & Convenience (25%): Size, parking, workstations  
    - Community Integration (20%): Specialized programs
    - Geographic Accessibility (15%): Central location
    - Historical Significance (10%): Age of library
    """
    score = 0.0
    
    # Skip non-physical branches
    if row['PhysicalBranch'] != 1:
        return 0.5
    
    # 1. Service Diversity (30% weight) - max 0.30 points
    service_count = 0
    services = ['KidsStop', 'LeadingReading', 'CLC', 'DIH', 'TeenCouncil', 
                'YouthHub', 'AdultLiteracyProgram']
    
    for service in services:
        if row.get(service, 0) == 1:
            service_count += 1
    
    service_score = min(service_count * 0.043, 0.30)  # Cap at 0.30
    score += service_score
    
    # 2. Accessibility & Convenience (25% weight) - max 0.25 points
    convenience_score = 0
    
    # Square footage (normalized)
    sq_ft = row.get('SquareFootage', 0)
    if pd.notna(sq_ft) and int(sq_ft) > 0:
        convenience_score += min(int(sq_ft) / 100000, 0.10)  # Max 0.10 for size
    
    # Parking availability
    parking = row.get('PublicParking', 0)
    if pd.notna(parking) and parking != 'shared' and int(parking) > 0:
        convenience_score += min(float(parking) / 200, 0.08)  # Max 0.08 for parking
    
    # Workstations
    workstations = row.get('Workstations', 0)
    if pd.notna(workstations) and workstations > 0:
        convenience_score += min(workstations / 300, 0.07)  # Max 0.07 for workstations
    
    score += min(convenience_score, 0.25)
    
    # 3. Community Integration (20% weight) - max 0.20 points
    # Based on service tier and specialized programs
    service_tier = row.get('ServiceTier', '')
    if service_tier == 'RR':  # Reference/Research
        score += 0.20
    elif service_tier == 'DL':  # District Library
        score += 0.15
    elif service_tier == 'NL':  # Neighborhood Library
        score += 0.10
    
    # 4. Geographic Accessibility (15% weight) - max 0.15 points
    area = get_area(row.get('Lat'), row.get('Long'))
    central_areas = ["downtown", "kensington", "chinatown", "queenwest", 
                    "trinitybellwoods", "harbourfront", "waterfront", "annex", "bloor"]
    
    if area in central_areas:
        score += 0.15
    elif area != "other":
        score += 0.10
    else:
        score += 0.05
    
    # 5. Historical Significance (10% weight) - max 0.10 points
    present_year = row.get('PresentSiteYear', 2023)
    if pd.notna(present_year) and present_year > 1900:
        age = 2023 - present_year
        historical_score = min(age / 1000, 0.10)  # Older = higher score
        score += historical_score
    
    return min(round(score, 3), 1.0)

# Load the CSV file
df = pd.read_csv("tpl-branch-general-information-2023.csv")

print(f"Processing {len(df)} library records...")

# Add area classification
df['area'] = df.apply(
    lambda row: get_area(row.get('Lat'), row.get('Long')), axis=1
)

# Add community hub score
df['score'] = df.apply(calculate_community_hub_score, axis=1)

# Save results
output_filename = "edu.csv"
df.to_csv(output_filename, index=False)

# Display summary statistics
print(f"\nAnalysis Complete! Results saved to: {output_filename}")
print(f"\nArea Distribution:")
print(df['area'].value_counts())
print(f"\nCommunity Hub Score Statistics:")
print(f"Mean score: {df['score'].mean():.3f}")
print(f"Min score: {df['score'].min():.3f}")
print(f"Max score: {df['score'].max():.3f}")
print(f"Standard deviation: {df['score'].std():.3f}")

print(f"\nTop 10 Highest Scoring Libraries:")
top_libraries = df.nlargest(10, 'score')[['BranchName', 'area', 'score', 'ServiceTier']]
for idx, row in top_libraries.iterrows():
    print(f"{row['BranchName']}: {row['score']:.3f} ({row['area']}, {row['ServiceTier']})")
