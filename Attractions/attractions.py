import pandas as pd
import ast

# Toronto region bounding boxes
bounding_boxes = {
    "downtown": {"lat_min": 43.6410, "lat_max": 43.6580, "lon_min": -79.3900, "lon_max": -79.3750},
    "kensington": {"lat_min": 43.6530, "lat_max": 43.6595, "lon_min": -79.4100, "lon_max": -79.3970},
    "chinatown": {"lat_min": 43.6500, "lat_max": 43.6580, "lon_min": -79.4010, "lon_max": -79.3950},
    "queenwest": {"lat_min": 43.6470, "lat_max": 43.6510, "lon_min": -79.4110, "lon_max": -79.3870},
    "trinitybellwoods": {"lat_min": 43.6460, "lat_max": 43.6545, "lon_min": -79.4180, "lon_max": -79.4090},
    "distillery": {"lat_min": 43.6480, "lat_max": 43.6515, "lon_min": -79.3615, "lon_max": -79.3550},
    "lawrence": {"lat_min": 43.7180, "lat_max": 43.7350, "lon_min": -79.3980, "lon_max": -79.3780},
    "harbourfront": {"lat_min": 43.6360, "lat_max": 43.6420, "lon_min": -79.4000, "lon_max": -79.3720},
    "waterfront": {"lat_min": 43.6370, "lat_max": 43.6510, "lon_min": -79.3810, "lon_max": -79.3610},
    "annex": {"lat_min": 43.6650, "lat_max": 43.6760, "lon_min": -79.4150, "lon_max": -79.3950},
    "bloor": {"lat_min": 43.6640, "lat_max": 43.6720, "lon_min": -79.4050, "lon_max": -79.3850},
    "leslieville": {"lat_min": 43.6580, "lat_max": 43.6680, "lon_min": -79.3500, "lon_max": -79.3270},
    "riverside": {"lat_min": 43.6560, "lat_max": 43.6640, "lon_min": -79.3600, "lon_max": -79.3410},
    "littleitaly": {"lat_min": 43.6540, "lat_max": 43.6600, "lon_min": -79.4240, "lon_max": -79.4090},
    "dundaswest": {"lat_min": 43.6500, "lat_max": 43.6590, "lon_min": -79.4420, "lon_max": -79.4300},
}


def get_area(lat, lon):
    for area, box in bounding_boxes.items():
        if box["lat_min"] <= lat <= box["lat_max"] and box["lon_min"] <= lon <= box["lon_max"]:
            return area
    return "other"


def parse_coordinates(geometry_str):
    try:
        geometry = ast.literal_eval(geometry_str)
        coords = geometry.get("coordinates")[0]
        lon, lat = coords
        return lat, lon
    except Exception:
        return None, None


def main():
    src = "Places of Interest and Attractions - 4326 (1).csv"
    df = pd.read_csv(src)

    # Extract coordinates
    if "geometry" in df.columns:
        df[["Latitude", "Longitude"]] = df["geometry"].apply(lambda g: pd.Series(parse_coordinates(g)))
    else:
        df["Latitude"], df["Longitude"] = None, None

    # Classify by area
    df["area"] = df.apply(
        lambda row: get_area(row["Latitude"], row["Longitude"]) if pd.notna(row.get("Latitude")) and pd.notna(row.get("Longitude")) else "other",
        axis=1,
    )

    # Output
    out = "attractions.csv"
    df.to_csv(out, index=False)
    print(f"Done! Wrote {out} with {len(df)} rows.")


if __name__ == "__main__":
    main()