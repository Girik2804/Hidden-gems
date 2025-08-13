import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import math
import os
from streamlit_js_eval import get_geolocation
import plotly.colors
import requests


# Page configuration
st.set_page_config(
    page_title="Toronto Apps", 
    page_icon="💎", 
    layout="wide",
    initial_sidebar_state="collapsed"
)


# category_symbol = {
#     "parks": "triangle-up",    # Documented as working
#     "dine": "square",          # Very reliable 
#     "education": "diamond",    # Well supported
#     "worship": "star"          # Basic symbol
# }

category_symbol = {
    "parks": "circle",    # Documented as working
    "dine": "circle",          # Very reliable 
    "education": "circle",    # Well supported
    "worship": "circle"          # Basic symbol
}


category_color = {
    "parks": "green",
    "dine": "red",
    "education": "black",
    "worship": "purple"
}


# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        # background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background: linear-gradient(135deg, #FF8D29 0%, #D95A4E 50%, #A21A6E 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-title {
        color: #ffffff;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        margin-bottom: 1.5rem;
    }
    
    .place-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .distance-badge {
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .category-tag {
        background: #f0f2f6;
        color: #333;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.7rem;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .filter-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_KEY = "32b5be06a7ba4bc3ac7779ad0a1bfdd6"  # Replace with your actual API key

# Data setup for predefined dataset
placeDict = {
    "Downtown Core": "downtown",
    "Kensington": "kensington", 
    "Chinatown": "chinatown",
    "Queen West": "queenwest",
    "Trinity Bellwoods": "trinitybellwoods",
    "Distillery District": "distillery",
    "Lawrence": "lawrence",
    "Harbourfront": "harbourfront",
    "Waterfront": "waterfront",
    "The Annex": "annex",
    "Bloor Corridor": "bloor",
    "Leslieville": "leslieville",
    "Riverside": "riverside",
    "Little Italy": "littleitaly",
    "Dundas West": "dundaswest"
}

# Mood definitions with recommended place categories
moods = [
    {"name": "Romantic", "emoji": "💕", "desc": "Looking for intimate dining", "categories": ["catering.restaurant", "tourism.attraction", "leisure.park"], "recommended": "dine"},
    {"name": "Anxious", "emoji": "😰", "desc": "Need some calm and peace", "categories": ["leisure.park", "natural.forest", "leisure.spa"], "recommended": "parks"},
    {"name": "Lonely", "emoji": "😔", "desc": "Want spiritual connection", "categories": ["religion.place_of_worship", "activity.community_center"], "recommended": "worship"},
    {"name": "Curious", "emoji": "🤔", "desc": "Ready to learn something new", "categories": ["education.library", "entertainment.museum", "tourism.attraction"], "recommended": "education"},
    {"name": "Hungry", "emoji": "😋", "desc": "Craving delicious food", "categories": ["catering.restaurant", "catering.cafe", "catering.fast_food"], "recommended": "dine"},
    {"name": "Social", "emoji": "😊", "desc": "Want to meet people", "categories": ["catering.bar", "catering.pub", "entertainment.cinema"], "recommended": "dine"},
    {"name": "Contemplative", "emoji": "🧘", "desc": "Seeking inner peace", "categories": ["religion.place_of_worship", "leisure.park", "natural.forest"], "recommended": "worship"},
    {"name": "Energetic", "emoji": "⚡", "desc": "Need outdoor activity", "categories": ["sport.fitness", "leisure.park", "sport.sports_centre"], "recommended": "parks"},
]

neighborhoods = [
    {"name": "Downtown Core", "emoji": "🏙️", "desc": "Urban heart with skyscrapers"},
    {"name": "Kensington", "emoji": "🎨", "desc": "Bohemian market area"},
    {"name": "Chinatown", "emoji": "🏮", "desc": "Vibrant Asian culture"},
    {"name": "Queen West", "emoji": "🎭", "desc": "Trendy arts district"},
    {"name": "Trinity Bellwoods", "emoji": "🌳", "desc": "Hip neighborhood around park"},
    {"name": "Distillery District", "emoji": "🏛️", "desc": "Historic cobblestone streets"},
    {"name": "Lawrence", "emoji": "🏘️", "desc": "Quiet residential area"},
    {"name": "Harbourfront", "emoji": "⚓", "desc": "Waterfront views"},
    {"name": "Waterfront", "emoji": "🌊", "desc": "Scenic lakeside promenades"},
    {"name": "The Annex", "emoji": "📚", "desc": "Intellectual hub near University"},
    {"name": "Bloor Corridor", "emoji": "🛍️", "desc": "Shopping paradise"},
    {"name": "Leslieville", "emoji": "☕", "desc": "Cozy cafes and bookstores"},
    {"name": "Riverside", "emoji": "🏞️", "desc": "Peaceful riverside walks"},
    {"name": "Little Italy", "emoji": "🍝", "desc": "Authentic Italian culture"},
    {"name": "Dundas West", "emoji": "🎪", "desc": "Emerging creative community"}
]

# Destination mapping
destination_info = {
    "parks": {"emoji": "🌲", "name": "Parks", "folder": "parks", "file": "parks.csv"},
    "education": {"emoji": "📚", "name": "Libraries", "folder": "edu", "file": "edu.csv"},
    "worship": {"emoji": "🕊️", "name": "Places of Worship", "folder": "worship", "file": "worship.csv"},
    "dine": {"emoji": "🍽️", "name": "Restaurants", "folder": "dine", "file": "dine.csv"}
}

# Initialize session state
if 'user_location' not in st.session_state:
    st.session_state.user_location = None
if 'selected_mood' not in st.session_state:
    st.session_state.selected_mood = None
if 'selected_neighborhood' not in st.session_state:
    st.session_state.selected_neighborhood = None
if 'places_data' not in st.session_state:
    st.session_state.places_data = []
if 'use_predefined' not in st.session_state:
    st.session_state.use_predefined = False
if 'selected_place_type' not in st.session_state:
    st.session_state.selected_place_type = "cafes"  # Change from "cafes_parks" to "cafes"
if 'show_toronto_map' not in st.session_state:
    st.session_state.show_toronto_map = False
# if 'selected_category' not in st.session_state:
#     st.session_state.selected_category = "parks"
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = 'all'  # Default to show all
    
    
    
    
def load_parking_data():
    """Load and parse parking data from JSON file"""
    try:
        import json
        
        with open('parking/parking.json', 'r') as f:
            parking_data = json.load(f)
        
        # Extract carpark data from the JSON structure
        carparks = parking_data.get('carparks', [])
        
        # Convert to DataFrame with relevant columns
        parking_list = []
        for carpark in carparks:
            # Extract rate details for operating hours
            operating_hours = "24/7"  # Default
            day_max = "N/A"
            night_max = "N/A"
            
            if 'rate_details' in carpark and carpark['rate_details'].get('periods'):
                periods = carpark['rate_details']['periods']
                if periods:
                    first_period = periods[0]
                    if 'rates' in first_period:
                        for rate_info in first_period['rates']:
                            when = rate_info.get('when', '')
                            if 'Day Maximum' in when:
                                day_max = rate_info.get('rate', 'N/A')
                            elif 'Night Maximum' in when:
                                night_max = rate_info.get('rate', 'N/A')
            
            parking_list.append({
                'Park Name': carpark.get('address', 'Unknown'),
                'Parking Spaces': int(carpark.get('capacity', 0)) if carpark.get('capacity', '').isdigit() else 0,
                'Handicap Parking Spaces': 0,  # Not in JSON, defaulting to 0
                'Access': 'Public',  # Assuming public access
                'Lat': float(carpark.get('lat', 0)),
                'Lon': float(carpark.get('lng', 0)),
                'Rate per 30min': carpark.get('rate_half_hour', '0.00'),
                'Carpark Type': carpark.get('carpark_type_str', 'Surface'),
                'Day Maximum': day_max,
                'Night Maximum': night_max,
                'Payment Methods': ', '.join(carpark.get('payment_methods', [])),
                'Operating Hours': operating_hours
            })
        
        parking_df = pd.DataFrame(parking_list)
        return parking_df
        
    except Exception as e:
        st.warning(f"Could not load parking data from JSON: {str(e)}")
        return pd.DataFrame()


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth (km)"""
    from math import radians, cos, sin, asin, sqrt
    
    # Convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  # Radius of earth in kilometers
    return c * r





def add_closest_parking_info(places_df, lat_col, lon_col):
    """Add closest parking information to places dataframe"""
    parking_df = load_parking_data()
    
    if parking_df.empty:
        # Add empty columns if parking data fails to load
        places_df['Closest Parking Name'] = 'N/A'
        places_df['Closest Parking Spaces'] = 0
        places_df['Closest Parking Distance (km)'] = float('inf')
        places_df['Closest Parking Rate'] = 'N/A'
        places_df['Closest Parking Hours'] = 'N/A'
        places_df['Closest Parking Cover'] = 'Open'
        return places_df
    
    closest_parkings = []
    
    for idx, row in places_df.iterrows():
        try:
            place_lat = row[lat_col]
            place_lon = row[lon_col]
            
            # Calculate distances to all parking lots
            distances = parking_df.apply(
                lambda x: haversine_distance(place_lat, place_lon, x['Lat'], x['Lon']), 
                axis=1
            )
            
            # Find closest parking lot
            closest_idx = distances.idxmin()
            closest_parking = parking_df.loc[closest_idx]
            
            # Determine parking cover
            try:
                _t = str(closest_parking['Carpark Type']).lower()
                covered = any(k in _t for k in ['underground','garage','structure','covered','interior','parkade'])
                open_like = any(k in _t for k in ['surface','lot','open','on-street','street'])
                cover_label = 'Covered' if covered and not open_like else ('Open' if open_like and not covered else ('Covered' if 'underground' in _t or 'garage' in _t else 'Open'))
            except Exception:
                cover_label = 'Open'
            
            closest_parkings.append({
                'Closest Parking Name': closest_parking['Park Name'],
                'Closest Parking Spaces': closest_parking['Parking Spaces'],
                'Closest Parking Handicap Spaces': 0,  # Not available in JSON
                'Closest Parking Distance (km)': round(distances[closest_idx], 2),
                'Closest Parking Access': closest_parking['Access'],
                'Closest Parking Rate': f"${closest_parking['Rate per 30min']}/30min",
                'Closest Parking Type': closest_parking['Carpark Type'],
                'Closest Parking Cover': cover_label,
                'Closest Parking Day Max': closest_parking['Day Maximum'],
                'Closest Parking Night Max': closest_parking['Night Maximum'],
                'Closest Parking Payment': closest_parking['Payment Methods']
            })
            
        except Exception as e:
            # Fallback for any errors
            closest_parkings.append({
                'Closest Parking Name': 'N/A',
                'Closest Parking Spaces': 0,
                'Closest Parking Handicap Spaces': 0,
                'Closest Parking Distance (km)': float('inf'),
                'Closest Parking Access': 'Unknown',
                'Closest Parking Rate': 'N/A',
                'Closest Parking Type': 'N/A',
                'Closest Parking Cover': 'Open',
                'Closest Parking Day Max': 'N/A',
                'Closest Parking Night Max': 'N/A',
                'Closest Parking Payment': 'N/A'
            })
    
    # Add closest parking columns to the dataframe
    for key in closest_parkings[0].keys():
        places_df[key] = [cp[key] for cp in closest_parkings]
    
    return places_df



    
    
    

    
    
def calculate_distance_to_places(df, user_lat, user_lon, lat_col, lon_col):
    """Calculate distance from user location to all places in dataframe"""
    import math
    
    def safe_haversine_distance(lat1, lon1, lat2, lon2):
        try:
            # Validate all inputs are numbers
            if not all(isinstance(x, (int, float)) and not pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
                return float('inf')
            
            R = 6371  # Earth's radius in kilometers
            phi1 = math.radians(lat1)
            phi2 = math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlambda = math.radians(lon2 - lon1)
            a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c
        except (ValueError, TypeError, OverflowError):
            return float('inf')
    
    # Calculate distances with error handling
    df['distance_km'] = df.apply(
        lambda row: safe_haversine_distance(user_lat, user_lon, row[lat_col], row[lon_col]), 
        axis=1
    )
    
    # Remove any places with invalid distances
    valid_distances = df['distance_km'] != float('inf')
    df = df[valid_distances]
    
    return df


# ===== Weather utilities (Open-Meteo) =====

def _describe_weather_code(code: int):
    """Return (description, emoji) for Open-Meteo weather_code."""
    try:
        code = int(code)
    except Exception:
        return ("Unknown", "❓")
    if code == 0:
        return ("Clear sky", "☀️")
    if code in {1, 2, 3}:
        return ("Partly cloudy", "⛅")
    if code in {45, 48}:
        return ("Foggy", "🌫️")
    if code in {51, 53, 55}:
        return ("Drizzle", "🌦️")
    if code in {56, 57}:
        return ("Freezing drizzle", "🌧️")
    if code in {61, 63, 65}:
        return ("Rain", "🌧️")
    if code in {66, 67}:
        return ("Freezing rain", "🌧️")
    if code in {71, 73, 75, 77}:
        return ("Snow", "❄️")
    if code in {80, 81, 82}:
        return ("Rain showers", "🌦️")
    if code in {85, 86}:
        return ("Snow showers", "🌨️")
    if code in {95, 96, 99}:
        return ("Thunderstorm", "⛈️")
    return ("Unknown", "❓")


def get_weather_data(latitude: float, longitude: float):
    """Fetch current weather and short-term rain outlook from Open-Meteo (no API key required)."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
            "hourly": "precipitation_probability,precipitation",
            "timezone": "auto",
            "forecast_days": 1,
        }
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        payload = resp.json()
        current = payload.get("current", {}) or {}
        hourly = payload.get("hourly", {}) or {}
        probs = hourly.get("precipitation_probability") or []
        precip_series = hourly.get("precipitation") or []
        # Next few hours outlook (max probability/precip within next 3 hours)
        next3_prob = max(probs[:3]) if len(probs) >= 1 else None
        next3_precip = max(precip_series[:3]) if len(precip_series) >= 1 else None
        desc, emoji = _describe_weather_code(current.get("weather_code"))
        return {
            "temperature": current.get("temperature_2m"),
            "apparent_temperature": current.get("apparent_temperature"),
            "precipitation_now": current.get("precipitation"),
            "wind_speed": current.get("wind_speed_10m"),
            "weather_code": current.get("weather_code"),
            "weather_desc": desc,
            "weather_emoji": emoji,
            "precip_prob_next_3h": next3_prob,
            "precip_next_3h": next3_precip,
        }
    except Exception as e:
        st.warning(f"Weather unavailable: {str(e)}")
        return None


def weather_based_tip(weather: dict) -> dict:
    """Return guidance for categories and parking given weather. Keys: text, parking_tip."""
    if not weather:
        return {"text": "", "parking_tip": ""}
    temp = weather.get("apparent_temperature") or weather.get("temperature")
    precip_prob = weather.get("precip_prob_next_3h") or 0
    code = weather.get("weather_code") or 0

    # Defaults
    suggestion_text = "Nice day for exploring Toronto. Parks and outdoor spots are great."
    parking_tip = "Surface parking is fine."

    try:
        if precip_prob and precip_prob >= 50:
            suggestion_text = "Rain expected soon — consider indoor gems like restaurants, libraries, or places of worship."
            parking_tip = "Prefer covered parking (garage/underground)."
        elif code in {61, 63, 65, 80, 81, 82, 95, 96, 99}:  # rainy/stormy
            suggestion_text = "Wet weather — indoor gems recommended; outdoor parks may be less comfortable."
            parking_tip = "Look for covered parking to stay dry."
        elif temp is not None and temp >= 28:
            suggestion_text = "Hot conditions — enjoy waterfront or shaded parks, or opt for indoor cooling."
            parking_tip = "Choose parking close to entrance or covered if available."
        elif temp is not None and temp <= -5:
            suggestion_text = "Very cold — indoor attractions are more comfortable today."
            parking_tip = "Covered parking helps reduce exposure."
    except Exception:
        pass

    return {"text": suggestion_text, "parking_tip": parking_tip}


def get_place_image(place_name, place_type="restaurant"):
    """Fetch image for a place using reliable image sources with fallbacks"""
    try:
        # Method 1: Use specific Unsplash URLs with better queries
        category_queries = {
            "dine": f"restaurant,food,dining,{place_name.replace(' ', ',')}",
            "parks": f"park,nature,outdoor,toronto,{place_name.replace(' ', ',')}",
            "education": f"library,books,education,{place_name.replace(' ', ',')}",
            "worship": f"church,temple,worship,{place_name.replace(' ', ',')}"
        }
        
        search_query = category_queries.get(place_type, f"{place_name.replace(' ', ',')},toronto")
        
        # Use Unsplash with better search terms
        unsplash_url = f"https://source.unsplash.com/400x250/?{search_query}"
        return unsplash_url
        
    except Exception as e:
        # Enhanced fallback with high-quality stock images
        fallback_images = {
            "dine": "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=400&h=250",
            "parks": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=400&h=250", 
            "education": "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=400&h=250",
            "worship": "https://images.unsplash.com/photo-1518098268026-4e89f1a2cd8e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=400&h=250"
        }
        return fallback_images.get(place_type, "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=250")

















def get_live_places_near_user(user_lat, user_lon, place_type="cafes", mood=None, radius=1000, limit=5):
    """Fetch real-time places from Geoapify API within radius of user location"""
    
    # Define mood-to-categories mapping for live API
    mood_categories = {
        "Anxious": ["leisure.park", "natural.forest", "leisure.spa"],
        "Stressed": ["leisure.park", "leisure.spa", "natural.water"],
        "Lonely": ["religion.place_of_worship", "activity.community_center"],
        "Curious": ["education.library", "entertainment.museum", "tourism.attraction"],
        "Hungry": ["catering.restaurant", "catering.cafe", "catering.fast_food"],
        "Social": ["catering.bar", "catering.pub", "entertainment.cinema"],
        "Contemplative": ["religion.place_of_worship", "leisure.park", "natural.forest"],
        "Energetic": ["sport.fitness", "leisure.park", "sport.sports_centre"],
        "Intellectual": ["education.library", "entertainment.museum", "education.university"],
        "Romantic": ["catering.restaurant", "tourism.attraction", "leisure.park"]
    }
    
    # Determine categories based on mood or place_type
    if mood and mood in mood_categories:
        categories = mood_categories[mood]
    elif place_type == "cafes_parks":
        categories = ["catering.cafe", "leisure.park"]
    elif place_type == "cafes":
        categories = ["catering.cafe"]
    elif place_type == "parks":
        categories = ["leisure.park"]
    elif place_type == "restaurants":
        categories = ["catering.restaurant"]
    elif place_type == "libraries":
        categories = ["education.library"]
    elif place_type == "worship":
        categories = ["religion.place_of_worship"]
    else:
        categories = ["catering.cafe"]
    
    # Build API URL
    categories_str = ",".join(categories)
    url = f"https://api.geoapify.com/v2/places"
    
    params = {
        "categories": categories_str,
        "filter": f"circle:{user_lon},{user_lat},{radius}",
        "bias": f"proximity:{user_lon},{user_lat}",
        "limit": limit,
        "apiKey": API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return parse_geoapify_response(data, user_lat, user_lon)
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return []
    except Exception as e:
        st.error(f"Failed to fetch live places: {str(e)}")
        return []
    
    
    
def load_all_toronto_gems():
    """Load all gems from different directories with color coding for Plotly scatter"""
    all_gems = []
    
    # Define colors for each category (Plotly-compatible)
    category_colors = {
        "parks": "green",
        "dine": "red",
        "education": "black",
        "worship": "purple"
    }
    
    # Define directory mappings with their specific column names
    directories = {
        'parks': {
            'folder': 'parks',
            'file': 'parks.csv',
            'lat_col': 'Latitude',
            'lon_col': 'Longitude',
            'name_col': 'ASSET_NAME',
            'score_col': 'score'
        },
        'dine': {
            'folder': 'dine',
            'file': 'dine.csv',
            'lat_col': 'Latitude',
            'lon_col': 'Longitude',
            'name_col': 'Establishment Name',
            'score_col': 'score'
        },
        'education': {
            'folder': 'edu',
            'file': 'edu.csv',
            'lat_col': 'Lat',  # Different column name!
            'lon_col': 'Long', # Different column name!
            'name_col': 'BranchName',
            'score_col': 'score'
        },
        'worship': {
            'folder': 'worship',
            'file': 'worship.csv',
            'lat_col': 'Latitude',
            'lon_col': 'Longitude',
            'name_col': 'FTH_ORGANIZATION',
            'score_col': None  # Worship data doesn't have scores
        }
    }
    
    for category, config in directories.items():
        try:
            file_path = f"{config['folder']}//{config['file']}"
            df = pd.read_csv(file_path)
            
            # Check if required columns exist
            if config['lat_col'] not in df.columns or config['lon_col'] not in df.columns:
                st.warning(f"Missing lat/lon columns in {file_path}")
                continue
            
            # Apply the same data cleaning as in load_and_display_gems
            original_count = len(df)
            
            # 1. Drop rows with missing critical columns
            required_cols = [config['lat_col'], config['lon_col']]
            if 'area' in df.columns:
                required_cols.append('area')
            df = df.dropna(subset=required_cols)
            
            # 2. Remove rows with zero coordinates (invalid)
            df = df[(df[config['lat_col']] != 0) & (df[config['lon_col']] != 0)]
            
            # 3. Remove rows with coordinates outside Toronto bounds
            df = df[(df[config['lat_col']] >= 43.5) & (df[config['lat_col']] <= 44.0)]
            df = df[(df[config['lon_col']] >= -80.0) & (df[config['lon_col']] <= -79.0)]
            
            # 4. Remove rows with empty or whitespace-only area values (if area column exists)
            if 'area' in df.columns:
                df = df[df['area'].astype(str).str.strip() != '']
                df = df[df['area'].astype(str).str.strip() != 'nan']
                
                # 5. Remove rows with invalid area codes
                valid_areas = list(placeDict.values())
                df = df[df['area'].isin(valid_areas)]
            
            # 6. Apply scoring threshold ≥ 0.8 (NEW ADDITION)
            if config['score_col'] and config['score_col'] in df.columns:
                df = df[df[config['score_col']] >= 0.6]
            
            # 7. Reset index after cleaning
            df = df.reset_index(drop=True)
            
            if df.empty:
                continue
            
            # Standardize column names for consistency
            df_standardized = df.copy()
            df_standardized['latitude'] = df[config['lat_col']]
            df_standardized['longitude'] = df[config['lon_col']]
            df_standardized['name'] = df[config['name_col']]
            
            # Add category and color information
            df_standardized['category'] = category
            df_standardized['color'] = category_colors[category]
            
            # Optionally preserve area and score if present
            if 'area' in df.columns:
                df_standardized['area'] = df['area']
            if config['score_col'] and config['score_col'] in df.columns:
                df_standardized['score'] = df[config['score_col']]
            
            # Keep only the columns we need for the map/search
            base_cols = ['latitude', 'longitude', 'name', 'category', 'color']
            if 'area' in df_standardized.columns:
                base_cols.append('area')
            if 'score' in df_standardized.columns:
                base_cols.append('score')
            df_final = df_standardized[base_cols].copy()
            
            all_gems.append(df_final)
            
        except FileNotFoundError:
            st.warning(f"Could not load {category} data from {file_path}")
            continue
        except KeyError as e:
            st.warning(f"Column {e} not found in {file_path}")
            continue
    
    if all_gems:
        # Combine all dataframes
        combined_df = pd.concat(all_gems, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()






def parse_geoapify_response(data, user_lat, user_lon):
    """Parse Geoapify response into structured place data"""
    places = []
    
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        geom = feature.get('geometry', {})
        coords = geom.get('coordinates', [None, None])
        
        if coords[0] and coords[1]:
            place = {
                'name': props.get('name', 'Unknown Place'),
                'lat': coords[1],
                'lon': coords[0],
                'categories': props.get('categories', []),
                'formatted_address': props.get('formatted', ''),
                'place_id': props.get('place_id', ''),
                'source': 'live_api'
            }
            places.append(place)
    
    return places

def load_and_display_gems(neighborhood=None, destination_type=None, user_lat=None, user_lon=None, sort_by="score_desc"):
    """Load gems data and create scatter plot"""
    try:
        dest_info = destination_info[destination_type]
        file_path = f"{dest_info['folder']}/{dest_info['file']}"
        
        if not os.path.exists(file_path):
            st.error(f"📁 Missing file: {file_path}")
            return None
        
        df = pd.read_csv(file_path)
        
        # Column mapping for different datasets
        if destination_type == "parks":
            lat_col, lon_col, area_col, score_col, name_col = "Latitude", "Longitude", "area", "score", "ASSET_NAME"
        elif destination_type == "education":
            lat_col, lon_col, area_col, score_col, name_col = "Lat", "Long", "area", "score", "BranchName"
        elif destination_type == "worship":
            lat_col, lon_col, area_col, score_col, name_col = "Latitude", "Longitude", "area", None, "FTH_ORGANIZATION"
        elif destination_type == "dine":
            lat_col, lon_col, area_col, score_col, name_col = "Latitude", "Longitude", "area", "score", "Establishment Name"

        # ===== ENHANCED DATA CLEANING =====
        original_count = len(df)
        
        # 1. Drop rows with missing critical columns
        df = df.dropna(subset=[lat_col, lon_col, area_col])
        
        # 2. Remove rows with zero coordinates (invalid)
        df = df[(df[lat_col] != 0) & (df[lon_col] != 0)]
        
        # 3. Remove rows with coordinates outside Toronto bounds
        df = df[(df[lat_col] >= 43.5) & (df[lat_col] <= 44.0)]
        df = df[(df[lon_col] >= -80.0) & (df[lon_col] <= -79.0)]
        
        # 4. Remove rows with empty or whitespace-only area values
        df = df[df[area_col].astype(str).str.strip() != '']
        df = df[df[area_col].astype(str).str.strip() != 'nan']
        
        # 5. Remove rows with obviously invalid area codes (not in our placeDict)
        valid_areas = list(placeDict.values())
        df = df[df[area_col].isin(valid_areas)]
        
        # 6. Reset index after cleaning
        df = df.reset_index(drop=True)
        
        cleaned_count = len(df)
        if cleaned_count < original_count:
            pass
            # st.info(f"🧹 Cleaned dataset: Removed {original_count - cleaned_count} problematic rows, kept {cleaned_count} valid places")
        
        # ===== AREA FILTERING =====
        if neighborhood:
            area_code = placeDict[neighborhood].lower()
            filtered_df = df[df[area_col] == area_code]
            
            if filtered_df.empty:
                st.warning(f"⚠️ No {dest_info['name'].lower()} found in {neighborhood}")
                return None
        else:
            # Show all clean places across Toronto
            filtered_df = df.copy()
            
            
        filtered_df["category"] = destination_type
        
        # ===== SCORE FILTERING =====
        if score_col and score_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[score_col] >= 0.6]
            
        if filtered_df.empty:
            if neighborhood:
                st.warning(f"⚠️ No {dest_info['name'].lower()} with score ≥ 0.6 found in {neighborhood}")
            else:
                st.warning(f"⚠️ No {dest_info['name'].lower()} with score ≥ 0.6 found")
            return None
        
        # ===== DISTANCE CALCULATION (on clean data) =====
        if user_lat and user_lon and len(filtered_df) > 0:
            try:
                filtered_df = calculate_distance_to_places(filtered_df, user_lat, user_lon, lat_col, lon_col)
            except Exception as e:
                st.warning(f"⚠️ Distance calculation failed: {str(e)}")
                filtered_df['distance_km'] = 0.0
                
                
        try:
            filtered_df = add_closest_parking_info(filtered_df, lat_col, lon_col)  # Fixed variable name
        except Exception as e:
            st.warning(f"⚠️ Parking data loading failed: {str(e)}")
                
        
        # ===== SORTING =====
        if sort_by == "distance_asc" and 'distance_km' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('distance_km', ascending=True)
        elif sort_by == "distance_desc" and 'distance_km' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('distance_km', ascending=False)
        elif sort_by == "score_desc" and score_col and score_col in filtered_df.columns:
            filtered_df = filtered_df.sort_values(score_col, ascending=False)
        elif score_col and score_col in filtered_df.columns:
            # Default to score sorting
            filtered_df = filtered_df.sort_values(score_col, ascending=False)
            
        current_category = destination_type
            
        # ===== CREATE MAP =====
        hover_data = {score_col: True} if score_col and score_col in filtered_df.columns else None
                
        
        fig = go.Figure()

        fig.add_trace(go.Scattermapbox(
            lat=filtered_df[lat_col],
            lon=filtered_df[lon_col],
            mode='markers',
            marker=dict(
                size=15,  # Larger size for visibility
                color=category_color.get(current_category, "red"),
                symbol=category_symbol.get(current_category, "circle"),
               
                opacity=0.8            # As suggested in documentation
            ),
            name=destination_type.capitalize(),
            text=filtered_df[name_col],
            hovertemplate='<b>%{text}</b>' +
                        (f'<br>Score: %{{customdata[0]:.2f}}' if score_col and score_col in filtered_df.columns else '') +
                        '<extra></extra>',
            customdata=filtered_df[[score_col]].values if score_col and score_col in filtered_df.columns else None
        ))


    
        if user_lat and user_lon:
            fig.add_trace(go.Scattermapbox(
                lat=[user_lat],
                lon=[user_lon],
                mode='markers',
                marker=dict(
                    size=25,           # Slightly larger for visibility
                    color='blue',      # ✅ Always blue
                    symbol='circle',   # ✅ Always circle
                    opacity=1.0        # ✅ Fully opaque
                ),
                name='You Are Here',
                text=['Your Location'],
                hovertemplate='<b>Your Location</b><extra></extra>',
                showlegend=False
            ))


        
        fig.update_traces(marker=dict(size=15, opacity=1))
        
        if not filtered_df.empty:
            center_lat = filtered_df[lat_col].mean()
            center_lon = filtered_df[lon_col].mean()
            
            fig.update_layout(
                mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=10 if not neighborhood else 13),
                margin={"r":0,"t":60,"l":0,"b":0},
                title_font_size=18,
                title_x=0.5,
                showlegend=False
            )
            
            fig.update_layout(
                mapbox=dict(
                    style="mapbox://styles/mapbox/streets-v11",
                    accesstoken="pk.eyJ1IjoiemVibWFwIiwiYSI6ImNtZTBjMTI4YzAzYnMybHFybGtmOHlsc3oifQ.f6ReOVgpgmwlN4F7ohgaiQ"  # Place your token here
                )
            )

        
        return fig, len(filtered_df), filtered_df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None



# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">The Toronto App</h1>
    <p class="main-subtitle">Discover live places near you based on your mood and location</p>
</div>
""", unsafe_allow_html=True)

# 🔎 Global Search
st.markdown("### 🔎 Search")
s_col1, s_col2 = st.columns([3, 1])
with s_col1:
	search_query = st.text_input("Search by name", key="global_search_query", placeholder="e.g., Trinity Bellwoods, Sushi, Library...", label_visibility="collapsed")
with s_col2:
	search_btn = st.button("Search", key="global_search_button")

if search_btn and search_query:
	try:
		all_df = load_all_toronto_gems()
		if all_df is not None and not all_df.empty:
			mask = all_df['name'].astype(str).str.contains(search_query.strip(), case=False, na=False)
			results = all_df[mask].copy()
			if not results.empty:
				# Add distance if user location is known
				dist_series = None
				if 'user_location' in st.session_state and st.session_state.user_location:
					u_lat, u_lon = st.session_state.user_location
					dist_series = results.apply(lambda r: haversine_distance(u_lat, u_lon, float(r['latitude']), float(r['longitude'])), axis=1)
					results['Distance (km)'] = dist_series
					results = results.sort_values('Distance (km)')
				# Closest parking for each result (best-effort; compute for top 25 to keep it fast)
				try:
					parking_df = load_parking_data()
					def summarize_parking(row):
						try:
							if parking_df is None or parking_df.empty:
								return "N/A"
							lat, lon = float(row['latitude']), float(row['longitude'])
							parking_df['__dist_km'] = parking_df.apply(lambda p: haversine_distance(lat, lon, float(p['Lat']), float(p['Lon'])), axis=1)
							p = parking_df.sort_values('__dist_km').iloc[0]
							# cover label
							pt = str(p.get('Carpark Type','')).lower()
							cover = 'Covered' if any(k in pt for k in ['underground','garage','structure','covered','indoor']) else ('Open' if any(k in pt for k in ['surface','lot','open','outdoor']) else 'Open')
							return f"{p.get('Park Name','Lot')} ({cover}, {p.get('Rate per 30min','$?')}/30min, {p['__dist_km']:.2f} km)"
						except Exception:
							return "N/A"
					results['Closest Parking'] = results.head(25).apply(summarize_parking, axis=1).reindex(results.index).fillna("")
				except Exception:
					results['Closest Parking'] = ""
				# Build display dataframe
				display = pd.DataFrame({
					'Name': results['name'],
					'Area': results['area'].astype(str).str.title() if 'area' in results.columns else "",
					'Category': results['category'].astype(str).str.capitalize(),
					'Score': results['score'].round(2) if 'score' in results.columns else None,
					'Distance (km)': results['Distance (km)'].round(2) if 'Distance (km)' in results.columns else None,
					'Closest Parking': results['Closest Parking'],
					'Navigate': results.apply(lambda r: f"https://www.google.com/maps/search/?api=1&query={r['latitude']},{r['longitude']}", axis=1)
				})
				st.dataframe(
					display,
					column_config={
						"Navigate": st.column_config.LinkColumn(
							"Navigate",
							display_text="Navigate with Maps"
						)
					},
					use_container_width=True
				)
			else:
				st.info("No matches found.")
		else:
			st.info("No data available to search.")
	except Exception as e:
		st.warning(f"Search failed: {str(e)}")

# Get user's live location
try:
    location = get_geolocation()
    
    if location is not None:
        user_lat = location['coords']['latitude']
        user_lon = location['coords']['longitude']
        st.session_state.user_location = (user_lat, user_lon)
        
        # ===== Weather at your location =====
        weather = get_weather_data(user_lat, user_lon)
        if weather:
            st.session_state.current_weather = weather
            colw1, colw2, colw3 = st.columns([1, 2, 2])
            with colw1:
                st.metric(
                    label=f"{weather['weather_emoji']} {weather['weather_desc']}",
                    value=f"{round(weather['temperature'],1)}°C" if isinstance(weather.get('temperature'), (int, float)) else "--",
                    delta=None
                )
            with colw2:
                tip = weather_based_tip(weather)
                if tip.get('text'):
                    st.info(tip['text'])
            with colw3:
                if tip and tip.get('parking_tip'):
                    st.success(f"🚗 Parking: {tip['parking_tip']}")
        
        # Filters Section
        # st.markdown("## 🎛️ Filters")
        
        # Add toggle for map mode
        col_toggle1, col_toggle2, col_toggle3 = st.columns([1, 1, 1])
        with col_toggle2:
            # st.text("🔄 Toggle Map Mode")
            map_mode = st.toggle(
                "🗺️ Show Toronto Hidden Gems Map", 
                value=st.session_state.show_toronto_map, 
                key="map_mode_toggle",
                help="Toggle between live places near you vs. curated hidden gems across Toronto"
            )
            
        st.markdown("<hr style='border: none; border-top;'>", unsafe_allow_html=True)


        # Handle toggle state
        st.session_state.show_toronto_map = map_mode

        # Update use_predefined based on toggle OR mood selection
        # Update use_predefined based on ONLY toggle (not mood)
        if st.session_state.show_toronto_map:
            st.session_state.use_predefined = True
        else:
            st.session_state.use_predefined = False

        # Create three columns for side-by-side filters
        col1, col2, col3 = st.columns(3)

        # Column 1: Mood Filter
        with col1:
            st.markdown("### 🎭 Mood Filter")
            mood_options = [
                "None",
                "Romantic", "Anxious", "Lonely", "Curious", 
                "Hungry", "Social", "Contemplative", "Energetic"
            ]
            
            current_mood = "None"
            if hasattr(st.session_state, 'selected_mood') and st.session_state.selected_mood:
                current_mood = st.session_state.selected_mood
            
            selected_mood = st.selectbox(
                "Choose your mood:",
                mood_options,
                index=mood_options.index(current_mood),
                key="mood_filter_select"
            )
            
            # Update session state
            if selected_mood == "None":
                st.session_state.selected_mood = None
            else:
                st.session_state.selected_mood = selected_mood

        # Column 2: Neighbourhood Filter
        with col2:
            st.markdown("### 🏘️ Neighbourhood Filter")
            neighborhood_options = ["All Areas"] + list(placeDict.keys())
            
            current_neighborhood = "All Areas"
            if hasattr(st.session_state, 'selected_neighborhood') and st.session_state.selected_neighborhood:
                current_neighborhood = st.session_state.selected_neighborhood
            
            selected_neighborhood = st.selectbox(
                "Choose area:",
                neighborhood_options,
                index=neighborhood_options.index(current_neighborhood),
                key="neighborhood_filter_select"
            )
            
            # Update session state
            if selected_neighborhood == "All Areas":
                st.session_state.selected_neighborhood = None
            else:
                st.session_state.selected_neighborhood = selected_neighborhood

        # Column 3: Place Type Filter
        # Column 3: Place Type Filter (Live API) OR Category Filter (Toronto Map)
        # Column 3: Place Type Filter (Live API) OR Category Filter (Toronto Map)
        with col3:
            if st.session_state.show_toronto_map:
                # Toronto Map Mode - Category Filter with All Option
                st.markdown("### 🏷️ Category")
                category_options = {
                    "🌟 All Categories": "all",  # New option
                    "🌲 Parks": "parks",
                    "🍽️ Restaurants": "dine", 
                    "📚 Libraries": "education",
                    "🕊️ Places of Worship": "worship"
                }
                
                # Set current selection based on session state
                current_category = "🌟 All Categories"
                for key, value in category_options.items():
                    if value == st.session_state.selected_category:
                        current_category = key
                        break
                
                selected_category_display = st.selectbox(
                    "Choose gem category:",
                    list(category_options.keys()),
                    index=list(category_options.keys()).index(current_category),
                    key="category_filter_select"
                )
                
                # Update session state
                st.session_state.selected_category = category_options[selected_category_display]
                
            else:
                # Live API Mode - Place Type Filter  
                st.markdown("### 📍 Place Type Filter")
                place_type_options = {
                    "All Types": None,
                    "Cafés": "cafes",
                    "Parks": "parks", 
                    "Restaurants": "restaurants",
                    "Libraries": "libraries",
                    "Places of Worship": "worship"
                }
                
                current_place_type = "All Types"
                if hasattr(st.session_state, 'selected_place_type') and st.session_state.selected_place_type:
                    for key, value in place_type_options.items():
                        if value == st.session_state.selected_place_type:
                            current_place_type = key
                            break
                
                selected_place_type_display = st.selectbox(
                    "Choose place type:",
                    list(place_type_options.keys()),
                    index=list(place_type_options.keys()).index(current_place_type),
                    key="place_type_filter_select"
                )
                
                # Update session state
                st.session_state.selected_place_type = place_type_options[selected_place_type_display]

        
        # Auto-load places based on current settings
        if st.session_state.show_toronto_map:
            # Toronto Map Display Logic
            if st.session_state.selected_category == "all":
    # Check if mood is selected for filtering
                if st.session_state.selected_mood:
                    # MOOD FILTERING: Load only the category that matches the selected mood
                    selected_mood_data = next(mood for mood in moods if mood["name"] == st.session_state.selected_mood)
                    recommended_destination = selected_mood_data['recommended']
                    
                    # Load gems for the mood-recommended category only
                    result = load_and_display_gems(
                        st.session_state.selected_neighborhood, 
                        recommended_destination,
                        user_lat=user_lat,
                        user_lon=user_lon,
                        sort_by="score_desc"
                    )
                    
                    if result:
                        fig, count, filtered_data = result
                        
                        # Display mood-filtered results
                        st.markdown(f"### 🎭 {destination_info[recommended_destination]['name']} for {st.session_state.selected_mood} Mood")
                        st.success(f"🎉 Found {count} places perfect for your {st.session_state.selected_mood.lower()} mood!")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display the data table
                        # Display the data table
                        if not filtered_data.empty:
                            st.markdown("#### 📍 Places Details")
                            
                            # Get appropriate columns for navigation
                            if recommended_destination == "parks":
                                lat_col, lon_col = "Latitude", "Longitude"
                            elif recommended_destination == "education":
                                lat_col, lon_col = "Lat", "Long"
                            elif recommended_destination == "worship":
                                lat_col, lon_col = "Latitude", "Longitude"
                            elif recommended_destination == "dine":
                                lat_col, lon_col = "Latitude", "Longitude"
                            
                            # ✅ Safely check which columns exist before accessing them
                            available_columns = filtered_data.columns.tolist()
                            display_columns = []
                            column_mapping = {}
                            
                            # Map the actual column names from your datasets
                            if 'ASSET_NAME' in available_columns:  # Parks
                                display_columns.append('ASSET_NAME')
                                column_mapping['ASSET_NAME'] = 'Name'
                            elif 'BranchName' in available_columns:  # Education
                                display_columns.append('BranchName')
                                column_mapping['BranchName'] = 'Name'
                            elif 'FTH_ORGANIZATION' in available_columns:  # Worship
                                display_columns.append('FTH_ORGANIZATION')
                                column_mapping['FTH_ORGANIZATION'] = 'Name'
                            elif 'Establishment Name' in available_columns:  # Dine
                                display_columns.append('Establishment Name')
                                column_mapping['Establishment Name'] = 'Name'
                            
                            # Add other available columns
                            if 'score' in available_columns:
                                display_columns.append('score')
                                column_mapping['score'] = 'Score'
                            if 'distance_km' in available_columns:
                                display_columns.append('distance_km')
                                column_mapping['distance_km'] = 'Distance (km)'
                            if 'category' in available_columns:
                                display_columns.append('category')
                                column_mapping['category'] = 'Category'
                            
                            if display_columns:
                                display_df = filtered_data[display_columns].copy()
                                display_df = display_df.rename(columns=column_mapping)
                                
                                # Round numeric columns if they exist
                                # Round numeric columns if they exist
                                if 'Score' in display_df.columns:
                                    display_df['Score'] = display_df['Score'].round(2)
                                if 'Distance (km)' in display_df.columns:
                                    display_df['Distance (km)'] = display_df['Distance (km)'].round(2)

                                # ✅ ADD PARKING INFORMATION - NEW
                                # Enhanced parking display with JSON data
                                # Enhanced parking display with JSON data
                                if 'Closest Parking Name' in filtered_data.columns:
                                    display_df['Closest Parking'] = filtered_data.apply(
                                        lambda row: f"{row['Closest Parking Name']} ({row['Closest Parking Spaces']} spaces, {row['Closest Parking Distance (km)']}km, {row['Closest Parking Rate']})", 
                                        axis=1
                                    )
                                    display_df['Parking Cover'] = filtered_data['Closest Parking Cover']
                                    display_df['Parking Prices(based on hours)'] = filtered_data.apply(
                                        lambda row: f"Day: {row['Closest Parking Day Max']} | Night: {row['Closest Parking Night Max']}", 
                                        axis=1
                                    )
                                    display_df['Payment Methods'] = filtered_data['Closest Parking Payment']

                                # ✅ KEEP NAVIGATION COLUMN
                                if lat_col in filtered_data.columns and lon_col in filtered_data.columns:
                                    display_df['Navigate with Maps'] = filtered_data.apply(
                                        lambda row: f"https://www.google.com/maps/search/?api=1&query={row[lat_col]},{row[lon_col]}", 
                                        axis=1
                                    )

                                # Display with navigation buttons
                                st.dataframe(
                                    display_df,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "Navigate with Maps": st.column_config.LinkColumn(
                                            "Navigate with Maps",
                                            help="Click to open in Google Maps",
                                            validate="^https://[a-z]",
                                            max_chars=100,
                                            display_text="🗺️ Navigate with Maps"
                                        )
                                    }
                                )


                            else:
                                # Fallback: show available columns for debugging
                                st.write("Available columns:", available_columns)
                                st.dataframe(filtered_data.head())


                        
                    else:
                        st.info(f"💡 No {destination_info[recommended_destination]['name'].lower()} places found for your {st.session_state.selected_mood.lower()} mood in this area. Try a different mood or location.")
                
                else:
                    # NO MOOD SELECTED: Show all categories as before
                    gems_df = load_all_toronto_gems()
                    
                    # Weather-aware guidance (only shows if we have user location weather)
                    try:
                        if 'user_location' in st.session_state and st.session_state.user_location:
                            w = st.session_state.get('current_weather') or get_weather_data(st.session_state.user_location[0], st.session_state.user_location[1])
                            tip = weather_based_tip(w) if w else None
                            if tip and tip.get('text') and w:
                                st.info(f"{w['weather_emoji']} {w['weather_desc']} — {tip['text']}")
                    except Exception:
                        pass
                    
                    if not gems_df.empty:
                        # Filter by neighborhood if selected
                        if st.session_state.selected_neighborhood and st.session_state.selected_neighborhood != "All Toronto":
                            gems_df = gems_df[gems_df['neighborhood'] == st.session_state.selected_neighborhood]
                    
                        if not gems_df.empty:
                            st.markdown(f"### 🌟 All Toronto Gems ({len(gems_df)} total)")
                            
                            # Create map figure for all categories
                            categories = gems_df['category'].unique()
                            colors = plotly.colors.qualitative.Plotly
                            
                            traces = []
                            for i, category in enumerate(categories):
                                category_data = gems_df[gems_df['category'] == category]
                                
                                trace = go.Scattermapbox(
                                    lat=category_data['latitude'],
                                    lon=category_data['longitude'],
                                    mode='markers',
                                    marker=dict(
                                        size=10,
                                        color=colors[i % len(colors)],
                                    ),
                                    text=category_data['name'],
                                    # customdata=category_data['category'],
                                    # hovertemplate='<b>%{text}</b><br>Category: %{customdata}<extra></extra>',
                                    name=category,
                                    showlegend=True
                                )
                                traces.append(trace)

                            fig = go.Figure(data=traces)

                            fig.update_layout(
                                mapbox=dict(
                                    style="open-street-map",
                                    zoom=11,
                                    center=dict(
                                        lat=gems_df['latitude'].mean(),
                                        lon=gems_df['longitude'].mean()
                                    )
                                ),
                                height=600,
                                margin=dict(r=0, t=0, l=0, b=0),
                                legend=dict(
                                    title="Categories",
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.01
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display summary stats
                            # Display summary stats (only show metrics for available data)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Places", len(gems_df))
                            with col2:
                                st.metric("Categories", gems_df['category'].nunique())

                            
                            # Display the data table
                            # Display the data table (only use available columns)
                            # Display the data table (only use available columns)
                            st.markdown("#### 📍 All Places")

                            # Only use columns that actually exist in gems_df
                            available_columns = gems_df.columns.tolist()
                            display_columns = []
                            column_mapping = {}

                            # Build display based on what's available
                            if 'name' in available_columns:
                                display_columns.append('name')
                                column_mapping['name'] = 'Name'
                            if 'category' in available_columns:
                                display_columns.append('category')
                                column_mapping['category'] = 'Category'

                            if display_columns:
                                display_df = gems_df[display_columns].copy()
                                display_df = display_df.rename(columns=column_mapping)
                                
                                # ✅ ADD NAVIGATION COLUMN FOR ALL CATEGORIES
                                # ✅ ADD NAVIGATION COLUMN FOR ALL CATEGORIES
                                if 'latitude' in gems_df.columns and 'longitude' in gems_df.columns:
                                    # Add parking info
                                    try:
                                        gems_df = add_closest_parking_info(gems_df, 'latitude', 'longitude')
                                        
                                        if 'Closest Parking Name' in gems_df.columns:
                                            display_df['Closest Parking'] = gems_df.apply(
                                                lambda row: f"{row['Closest Parking Name']} ({row['Closest Parking Spaces']} spaces, {row['Closest Parking Distance (km)']}km, {row['Closest Parking Rate']})", 
                                                axis=1
                                            )
                                            display_df['Parking Cover'] = gems_df['Closest Parking Cover']
                                    except Exception as e:
                                        st.warning(f"Could not add parking info: {str(e)}")
                                    
                                    # ✅ KEEP NAVIGATION
                                    display_df['Navigate with Maps'] = gems_df.apply(
                                        lambda row: f"https://www.google.com/maps/search/?api=1&query={row['latitude']},{row['longitude']}", 
                                        axis=1
                                    )

                                # Display with navigation buttons
                                st.dataframe(
                                    display_df,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "Navigate with Maps": st.column_config.LinkColumn(
                                            "Navigate with Maps",
                                            help="Click to open in Google Maps",
                                            validate="^https://[a-z]",
                                            max_chars=100,
                                            display_text="🗺️ Navigate with Maps"
                                        )
                                    }
                                )

                            else:
                                # Fallback: show raw data if column mapping fails
                                st.dataframe(gems_df, use_container_width=True, hide_index=True)


                        else:
                            st.info("No places found in the selected neighborhood.")
                    else:
                        st.warning("No data available to display.")

                
            else:
                if st.session_state.selected_mood:
                    selected_mood_data = next(mood for mood in moods if mood["name"] == st.session_state.selected_mood)
                    recommended_destination = selected_mood_data['recommended']
                    sort_by = "score_desc"  # Default for mood-based
                else:
                    recommended_destination = st.session_state.selected_category
                    sort_by = st.session_state.get('selected_sort', 'score_desc')

                # Call function with user location and sorting
                result = load_and_display_gems(
                    st.session_state.selected_neighborhood, 
                    recommended_destination,
                    user_lat=user_lat,  # Pass user location
                    user_lon=user_lon,  # Pass user location
                    sort_by=sort_by
                )

                
                # Show title based on whether neighborhood is selected
                # Show title based on whether neighborhood is selected
                if st.session_state.selected_neighborhood:
                    if st.session_state.selected_mood:
                        st.markdown(f"## 🗺️ {destination_info[recommended_destination]['name']} for {st.session_state.selected_mood} Mood in {st.session_state.selected_neighborhood}")
                    else:
                        st.markdown(f"## 🗺️ {destination_info[recommended_destination]['name']} in {st.session_state.selected_neighborhood}")
                else:
                    if st.session_state.selected_mood:
                        st.markdown(f"## 🗺️ All {destination_info[recommended_destination]['name']} for {st.session_state.selected_mood} Mood")
                    else:
                        st.markdown(f"## 🗺️ Toronto Hidden Gems - {destination_info[recommended_destination]['name']}")

                
                # result = load_and_display_gems(st.session_state.selected_neighborhood, recommended_destination)
                
                if result:
                    fig, count, filtered_data = result
                    
                    # Update success message
                    if st.session_state.selected_mood:
                        if st.session_state.selected_neighborhood:
                            st.success(f"🎉 Found {count} places perfect for your {st.session_state.selected_mood.lower()} mood in {st.session_state.selected_neighborhood}!")
                        else:
                            st.success(f"🎉 Found {count} places perfect for your {st.session_state.selected_mood.lower()} mood across all areas!")
                    else:
                        if st.session_state.selected_neighborhood:
                            st.success(f"🎉 Found {count} hidden gems in {st.session_state.selected_neighborhood}!")
                        else:
                            st.success(f"🎉 Found {count} hidden gems across Toronto!")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display places table for predefined dataset
                    st.markdown("## 🔥 Top Places" + (f" in {st.session_state.selected_neighborhood}" if st.session_state.selected_neighborhood else " (All Areas)"))

                    # Get appropriate columns
                    if recommended_destination == "parks":
                        lat_col, lon_col, name_col = "Latitude", "Longitude", "ASSET_NAME"
                    elif recommended_destination == "education":
                        lat_col, lon_col, name_col = "Lat", "Long", "BranchName"
                    elif recommended_destination == "worship":
                        lat_col, lon_col, name_col = "Latitude", "Longitude", "FTH_ORGANIZATION"
                    elif recommended_destination == "dine":
                        lat_col, lon_col, name_col = "Latitude", "Longitude", "Establishment Name"

                    # Ensure distance is calculated BEFORE slicing data
                    if user_lat and user_lon and 'distance_km' not in filtered_data.columns:
                        filtered_data = calculate_distance_to_places(filtered_data, user_lat, user_lon, lat_col, lon_col)

                    # Show more places if all areas, fewer if filtered by neighborhood
                    # display_count = 10 if not st.session_state.selected_neighborhood else 5
                    # display_gems = filtered_data.head(display_count).copy()
                    display_gems = filtered_data.copy()

                    # Create display dataframe
                    df_display = pd.DataFrame()

                    # Add area column if showing all areas
                    if not st.session_state.selected_neighborhood:
                        if 'area' in display_gems.columns:
                            df_display['Area'] = display_gems['area'].apply(lambda x: next(k for k, v in placeDict.items() if v == x))

                    # Add score if available
                    if 'score' in display_gems.columns:
                        df_display['Score'] = display_gems['score'].round(2)

                    # Add name column (FIXED: removed quotes around name_col)
                    if name_col in display_gems.columns:
                        df_display['Name'] = display_gems[name_col]

                    # Add distance if available
                    # Add distance if available
                    if 'distance_km' in display_gems.columns:
                        df_display['Distance (km)'] = display_gems['distance_km'].round(2)

                    # ✅ ADD PARKING COLUMNS HERE
                    # Enhanced parking display with JSON data
                    # Add parking and navigation columns
                    if 'Closest Parking Name' in display_gems.columns:
                        df_display['Closest Parking'] = display_gems.apply(
                            lambda row: f"{row['Closest Parking Name']} ({row['Closest Parking Spaces']} spaces, {row['Closest Parking Distance (km)']}km, {row['Closest Parking Rate']})", 
                            axis=1
                        )
                        df_display['Parking Cover'] = display_gems['Closest Parking Cover']
                        df_display['Parking Prices(based on hours)'] = display_gems.apply(
                            lambda row: f"Day: {row['Closest Parking Day Max']} | Night: {row['Closest Parking Night Max']}", 
                            axis=1
                        )

                    # ✅ KEEP NAVIGATION COLUMN
                    df_display['Navigate with Maps'] = display_gems.apply(
                        lambda row: f"https://www.google.com/maps/search/?api=1&query={row[lat_col]},{row[lon_col]}", 
                        axis=1
                    )

                    # Display with navigation buttons
                    st.dataframe(
                        df_display,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Navigate with Maps": st.column_config.LinkColumn(
                                "Navigate with Maps",
                                help="Click to open in Google Maps",
                                validate="^https://[a-z]",
                                max_chars=100,
                                display_text="🗺️ Navigate with Maps"
                            )
                        }
                    )

                    # Weather-aware parking recommendation under the table
                    try:
                        w = st.session_state.get('current_weather')
                        if w:
                            precip_prob = w.get('precip_prob_next_3h') or 0
                            code = w.get('weather_code') or 0
                            # Lower threshold and always show suggestions (includes drizzle)
                            bad_weather = (precip_prob >= 30) or (code in {51,53,55,56,57,61,63,65,80,81,82,95,96,99})

                            # Find a suggested parking near you based on cover preference
                            suggested = None
                            try:
                                pdf = load_parking_data()
                                if not pdf.empty and 'Lat' in pdf.columns and 'Lon' in pdf.columns:
                                    def _cover_label(t):
                                        t = str(t).lower()
                                        covered = any(k in t for k in ['underground','garage','structure','covered','interior','parkade'])
                                        open_like = any(k in t for k in ['surface','lot','open','on-street','street'])
                                        return 'Covered' if covered and not open_like else ('Open' if open_like and not covered else ('Covered' if 'underground' in t or 'garage' in t else 'Open'))
                                    pdf = pdf.copy()
                                    pdf['Cover'] = pdf['Carpark Type'].apply(_cover_label)
                                    pdf['dist_km'] = pdf.apply(lambda r: haversine_distance(user_lat, user_lon, float(r['Lat']), float(r['Lon'])), axis=1)
                                    pref = 'Covered' if bad_weather else 'Open'
                                    filtered = pdf[pdf['Cover'] == pref]
                                    if filtered.empty:
                                        filtered = pdf
                                    suggested = filtered.loc[filtered['dist_km'].idxmin()]
                            except Exception:
                                suggested = None

                            if bad_weather:
                                if suggested is not None:
                                    _text = f"☔ Recommended parking: Covered. Suggested: {suggested['Park Name']} — {suggested['Cover']}, {suggested['dist_km']:.2f} km, ${suggested['Rate per 30min']}/30min; Day {suggested['Day Maximum']}, Night {suggested['Night Maximum']}."
                                else:
                                    _text = "☔ Recommended parking: Covered (garage/underground) due to current/expected weather."
                                st.markdown(f"""
                                <div style="font-size:0.9rem; padding:8px 12px; border-radius:8px; background:#fff3cd; color:#664d03; border:1px solid #ffe69c;">
                                    {_text}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                if suggested is not None:
                                    _text = f"☀️ Suggested parking: Open/surface. Suggested: {suggested['Park Name']} — {suggested['Cover']}, {suggested['dist_km']:.2f} km, ${suggested['Rate per 30min']}/30min; Day {suggested['Day Maximum']}, Night {suggested['Night Maximum']}."
                                else:
                                    _text = "☀️ Suggested parking: Open/surface is fine in sunny/clear conditions."
                                st.markdown(f"""
                                <div style="font-size:0.9rem; padding:8px 12px; border-radius:8px; background:#e7f1ff; color:#084298; border:1px solid #b6d4fe;">
                                    {_text}
                                </div>
                                """, unsafe_allow_html=True)
                    except Exception:
                        pass

                else:
                    if st.session_state.selected_neighborhood:
                        st.info("💡 Try selecting a different neighborhood or category combination.")
                    else:
                        st.info("💡 No places found for this category. Try a different category.")
                
        else:
    # Use live API (default behavior) - now with mood support
            with st.spinner("🔍 Searching for places near you..."):
                # Determine what to search for - mood overrides place type
                if st.session_state.selected_mood:
                    places = get_live_places_near_user(
                        user_lat, user_lon, 
                        mood=st.session_state.selected_mood,
                        limit=5
                    )
                    search_description = f"mood-based ({st.session_state.selected_mood.lower()})"
                else:
                    places = get_live_places_near_user(
                        user_lat, user_lon, 
                        place_type=st.session_state.selected_place_type,
                        limit=5
                    )
                    place_type_names = {
                        "cafes_parks": "cafés & parks",
                        "cafes": "cafés",
                        "parks": "parks",
                        "restaurants": "restaurants",
                        "libraries": "libraries", 
                        "worship": "places of worship"
                    }
                    search_description = place_type_names.get(st.session_state.selected_place_type, "mixed")
                
                # 🚨 DEFINE current_category RIGHT HERE - BEFORE using it 🚨
                if st.session_state.selected_mood:
                    mood_to_category = {
                        "Romantic": "dine",
                        "Anxious": "parks", 
                        "Lonely": "worship",
                        "Curious": "education",
                        "Hungry": "dine",
                        "Social": "dine",
                        "Contemplative": "worship",
                        "Energetic": "parks"
                    }
                    current_category = mood_to_category.get(st.session_state.selected_mood, "dine")
                else:
                    place_type_to_category = {
                        "cafes": "dine",
                        "cafes_parks": "dine",
                        "parks": "parks", 
                        "restaurants": "dine",
                        "libraries": "education",
                        "worship": "worship"
                    }
                    current_category = place_type_to_category.get(st.session_state.selected_place_type, "dine")
                
                st.session_state.places_data = places
                
                if places:
                    if st.session_state.selected_mood:
                        st.success(f"🎉 Found {len(places)} places perfect for your {st.session_state.selected_mood.lower()} mood within 1km!")
                        st.info(f"💡 **Live mood-based search** - Enable 'Toronto Hidden Gems Map' for city-wide results")
                    else:
                        st.success(f"🎉 Found {len(places)} {search_description} within 1km!")
                    
                    # Display map
                    if st.session_state.selected_mood:
                        st.markdown(f"## 🗺️ Live Places for {st.session_state.selected_mood} Mood")
                    else:
                        st.markdown("## 🗺️ Places Near You")
                    
                    if places:
                        place_lats = [p['lat'] for p in places]
                        place_lons = [p['lon'] for p in places]
                        place_names = [p['name'] for p in places]
                        
                        title_text = f"🗺️ Live {st.session_state.selected_mood} Mood Places" if st.session_state.selected_mood else f"🗺️ Live Places Near You"
                        
                        # Create figure with custom symbols support
                        fig = go.Figure()

                        # Add places markers with custom symbols
                        fig.add_trace(go.Scattermapbox(
                            lat=place_lats,
                            lon=place_lons,
                            mode='markers',
                            marker=dict(
        size=15,  # Larger size for visibility
        color=category_color.get(current_category, "red"),
        symbol=category_symbol.get(current_category, "circle"),
        
        opacity=0.8            # As suggested in documentation
    ),
                            # marker=dict(
                            #     size=18,
                            #     color=category_color.get(current_category, "red"),  # Dynamic color based on category
                            #     symbol=category_symbol.get(current_category, "circle"),  # Dynamic symbol based on category
                            # ),
                            name=f'{current_category.title()} Places',
                            text=place_names,
                            hovertemplate='<b>%{text}</b><extra></extra>',
                            showlegend=False
                        ))

                        # Set up the mapbox layout with your token
                        fig.update_layout(
                            mapbox=dict(
                                style="mapbox://styles/mapbox/streets-v11",
                                accesstoken="pk.eyJ1IjoiemVibWFwIiwiYSI6ImNtZTBjMTI4YzAzYnMybHFybGtmOHlsc3oifQ.f6ReOVgpgmwlN4F7ohgaiQ",
                                center=dict(lat=sum(place_lats)/len(place_lats), lon=sum(place_lons)/len(place_lons)),
                                zoom=14
                            ),
                            height=600,
                            title=title_text,
                            title_font_size=18,
                            title_x=0.5,
                            margin={"r":0,"t":60,"l":0,"b":0},
                            showlegend=False
                        )

                        
                        # Add user location marker
                        fig.add_trace(go.Scattermapbox(
                            lat=[user_lat],
                            lon=[user_lon],
                            mode='markers',
                            marker=dict(
                                size=25,           # ✅ Larger for visibility
                                color='blue',      # ✅ Always blue
                                symbol='circle',   # ✅ Always circle
                                opacity=1.0        # ✅ Fully opaque
                            ),
                            name='You Are Here',
                            text=['Your Current Location'],
                            hovertemplate='<b>Your Location</b><extra></extra>',
                            showlegend=False
                        ))

                        
                        fig.add_trace(go.Scattermapbox(
                            lat=place_lats,
                            lon=place_lons,
                            mode='markers',
                            marker=dict(
        size=15,  # Larger size for visibility
        color=category_color.get(current_category, "red"),
        symbol=category_symbol.get(current_category, "circle"),
        opacity=0.8            # As suggested in documentation
    ),
                            # marker=dict(
                            #     size=15, 
                            #     color=category_color.get(current_category, "red"),
                            #     symbol=category_symbol.get(current_category, "circle")
                            # ),
                            name='Nearby Places',
                            text=place_names,
                            # hovertemplate='<b>%{text}</b><br>Click for details<extra></extra>',
                            showlegend=False
                        ))
                        
                        # Add Mapbox token and styling
                        fig.update_layout(
                            mapbox=dict(
                                style="mapbox://styles/mapbox/streets-v11",
                                accesstoken="pk.eyJ1IjoiemVibWFwIiwiYSI6ImNtZTBjMTI4YzAzYnMybHFybGtmOHlsc3oifQ.f6ReOVgpgmwlN4F7ohgaiQ",  # Replace with your token
                                center=dict(lat=user_lat, lon=user_lon), 
                                zoom=13
                            ),
                            margin={"r":0,"t":60,"l":0,"b":0},
                            title_font_size=18,
                            title_x=0.5,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                                        
                    # Display places table with navigation links
                    if st.session_state.selected_mood:
                        st.markdown(f"## 🔥 Top Places for {st.session_state.selected_mood} Mood")
                    else:
                        st.markdown("## 🔥 Top Places Near You")

                    # Create display dataframe with navigation links
                    # Create enhanced display dataframe with parking info
                    places_data = []

                    # First, create a temporary dataframe for Live API places to use parking function
                    live_places_df = pd.DataFrame([{
                        'lat': place['lat'],
                        'lon': place['lon'],
                        'name': place['name'],
                        'address': place['formatted_address'],
                        'categories': ', '.join([cat.replace("_", " ").title() for cat in place['categories'][:2]])
                    } for place in places])

                    # Add parking information
                    if not live_places_df.empty:
                        live_places_df = add_closest_parking_info(live_places_df, 'lat', 'lon')

                    # Create display data with parking info
                    # Create display data with enhanced parking info
                    # Create display data with enhanced parking info
                    # Create display data with enhanced parking info and navigation
                    for idx, place in enumerate(places):
                        parking_info = ""
                        parking_hours = ""
                        payment_methods = ""
                        
                        if not live_places_df.empty and idx < len(live_places_df):
                            row = live_places_df.iloc[idx]
                            parking_name = row['Closest Parking Name']
                            parking_spaces = row['Closest Parking Spaces']
                            parking_distance = row['Closest Parking Distance (km)']
                            parking_rate = row['Closest Parking Rate']
                            parking_day_max = row['Closest Parking Day Max']
                            parking_night_max = row['Closest Parking Night Max']
                            
                            parking_info = f"{parking_name} ({parking_spaces} spaces, {parking_distance}km, {parking_rate})"
                            parking_hours = f"Day: {parking_day_max} | Night: {parking_night_max}"
                            payment_methods = row['Closest Parking Payment']
                        
                        # Attach brief weather label for quick decision-making
                        weather_label = ""
                        try:
                            if 'user_location' in st.session_state and st.session_state.user_location:
                                w = st.session_state.get('current_weather') or get_weather_data(st.session_state.user_location[0], st.session_state.user_location[1])
                                if w:
                                    weather_label = f"{w['weather_emoji']} {w['weather_desc']}"
                        except Exception:
                            pass
                        
                        places_data.append({
                            'Place Name': place['name'],
                            'Address': place['formatted_address'],
                            'Categories': ', '.join([cat.replace("_", " ").title() for cat in place['categories'][:2]]),
                            'Closest Parking': parking_info,
                            'Parking Cover': row['Closest Parking Cover'] if not live_places_df.empty and idx < len(live_places_df) and 'Closest Parking Cover' in row else '',
                            'Parking Prices (based on hours)': parking_hours,
                            'Payment Methods': payment_methods,
                            'Weather Now': weather_label,
                            'Navigate with Maps': f"https://www.google.com/maps/search/?api=1&query={place['lat']},{place['lon']}"  # ✅ KEEP THIS
                        })

                    df_display = pd.DataFrame(places_data)

                    # ✅ KEEP THE NAVIGATION COLUMN CONFIG
                    st.dataframe(
                        df_display,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Navigate with Maps": st.column_config.LinkColumn(
                                "Navigate with Maps",
                                help="Click to open in Google Maps",
                                validate="^https://[a-z]",
                                max_chars=100,
                                display_text="🗺️ Navigate with Maps"
                            )
                        }
                    )

                    # Weather-aware parking recommendation under the table
                    try:
                        w = st.session_state.get('current_weather')
                        if w:
                            precip_prob = w.get('precip_prob_next_3h') or 0
                            code = w.get('weather_code') or 0
                            # Lower threshold and always show suggestions (includes drizzle)
                            bad_weather = (precip_prob >= 30) or (code in {51,53,55,56,57,61,63,65,80,81,82,95,96,99})

                            # Find a suggested parking near you based on cover preference
                            suggested = None
                            try:
                                pdf = load_parking_data()
                                if not pdf.empty and 'Lat' in pdf.columns and 'Lon' in pdf.columns:
                                    def _cover_label(t):
                                        t = str(t).lower()
                                        covered = any(k in t for k in ['underground','garage','structure','covered','interior','parkade'])
                                        open_like = any(k in t for k in ['surface','lot','open','on-street','street'])
                                        return 'Covered' if covered and not open_like else ('Open' if open_like and not covered else ('Covered' if 'underground' in t or 'garage' in t else 'Open'))
                                    pdf = pdf.copy()
                                    pdf['Cover'] = pdf['Carpark Type'].apply(_cover_label)
                                    pdf['dist_km'] = pdf.apply(lambda r: haversine_distance(user_lat, user_lon, float(r['Lat']), float(r['Lon'])), axis=1)
                                    pref = 'Covered' if bad_weather else 'Open'
                                    filtered = pdf[pdf['Cover'] == pref]
                                    if filtered.empty:
                                        filtered = pdf
                                    suggested = filtered.loc[filtered['dist_km'].idxmin()]
                            except Exception:
                                suggested = None

                            if bad_weather:
                                if suggested is not None:
                                    _text = f"☔ Recommended parking: Covered. Suggested: {suggested['Park Name']} — {suggested['Cover']}, {suggested['dist_km']:.2f} km, ${suggested['Rate per 30min']}/30min; Day {suggested['Day Maximum']}, Night {suggested['Night Maximum']}."
                                else:
                                    _text = "☔ Recommended parking: Covered (garage/underground) due to current/expected weather."
                                st.markdown(f"""
                                <div style="font-size:0.9rem; padding:8px 12px; border-radius:8px; background:#fff3cd; color:#664d03; border:1px solid #ffe69c;">
                                    {_text}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                if suggested is not None:
                                    _text = f"☀️ Suggested parking: Open/surface. Suggested: {suggested['Park Name']} — {suggested['Cover']}, {suggested['dist_km']:.2f} km, ${suggested['Rate per 30min']}/30min; Day {suggested['Day Maximum']}, Night {suggested['Night Maximum']}."
                                else:
                                    _text = "☀️ Suggested parking: Open/surface is fine in sunny/clear conditions."
                                st.markdown(f"""
                                <div style="font-size:0.9rem; padding:8px 12px; border-radius:8px; background:#e7f1ff; color:#084298; border:1px solid #b6d4fe;">
                                    {_text}
                                </div>
                                """, unsafe_allow_html=True)
                    except Exception:
                        pass

                
                else:
                    if st.session_state.selected_mood:
                        st.warning(f"⚠️ No places found within 1km for your {st.session_state.selected_mood.lower()} mood. Try enabling 'Toronto Hidden Gems Map' for city-wide results!")
                    else:
                        st.warning("⚠️ No places found within 1km. Try a different filter or check back later!")


    else:
        st.error("🚫 Location access denied or unavailable. Please enable location services in your browser and refresh the page.")

except Exception as e:
    st.error(f"Error getting location: {str(e)}")
    st.info("💡 Make sure to allow location access when prompted by your browser.")

# Display current status
if st.session_state.places_data or st.session_state.use_predefined:
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.places_data and not st.session_state.use_predefined:
            st.metric("📍 Places Found", len(st.session_state.places_data))
        elif st.session_state.selected_neighborhood:
            st.metric("📍 Area", st.session_state.selected_neighborhood)
    with col2:
        if st.session_state.selected_mood:
            st.metric("🎭 Current Mood", st.session_state.selected_mood)
        elif st.session_state.selected_place_type:
            place_type_names = {
                "cafes_parks": "Cafés & Parks",
                "cafes": "Cafés",
                "parks": "Parks",
                "restaurants": "Restaurants", 
                "libraries": "Libraries",
                "worship": "Worship Places"
            }
            # st.metric("🏪 Place Type", place_type_names.get(st.session_state.selected_place_type, "Mixed"))

# ===== Quick Itinerary Builder (placed above Trending) =====
st.markdown("---")
st.markdown("## ⚡ Quick Itinerary Builder")

# Inputs
it_col1, it_col2, it_col3 = st.columns([1, 1, 1])
with it_col1:
	mood_names = ["None"] + [m["name"] for m in moods]
	default_mood = st.session_state.selected_mood if st.session_state.selected_mood in mood_names else "None"
	it_mood = st.selectbox("Mood (optional)", mood_names, index=mood_names.index(default_mood))
with it_col2:
	it_time = st.selectbox("Time", ["Now", "Morning", "Afternoon", "Evening"])
with it_col3:
	it_radius_km = st.slider("Max walking/driving radius (km)", 0.5, 5.0, 2.0, 0.5)

# Additional planning filters
pf_col1, pf_col2 = st.columns([1, 1])
with pf_col1:
	neighborhood_options = ["Any"] + list(placeDict.keys())
	it_neighborhood = st.selectbox("Neighborhood", neighborhood_options)
with pf_col2:
	it_attraction_type = st.selectbox("Attraction Type", ["Any", "Parks", "Libraries"])

build_btn = st.button("Build Itinerary")
new_btn = st.button("New options")

if build_btn or new_btn:
	try:
		# Determine origin: user location if available; else Toronto downtown
		origin_lat, origin_lon = 43.6532, -79.3832
		if 'user_location' in st.session_state and st.session_state.user_location:
			origin_lat, origin_lon = st.session_state.user_location

		# Persist and reuse filters for new options
		if 'itinerary_filters' not in st.session_state:
			st.session_state.itinerary_filters = None
		if 'itinerary_iter' not in st.session_state:
			st.session_state.itinerary_iter = 0
		if build_btn or not st.session_state.itinerary_filters:
			st.session_state.itinerary_filters = {
				'mood': it_mood,
				'time': it_time,
				'radius_km': it_radius_km,
				'neighborhood': it_neighborhood,
				'attraction_type': it_attraction_type,
			}
			offset = 0
			st.session_state.itinerary_iter = 0
		else:
			offset = st.session_state.itinerary_iter
			# Overwrite local vars with persisted filters to keep same settings
			it_mood = st.session_state.itinerary_filters['mood']
			it_time = st.session_state.itinerary_filters['time']
			it_radius_km = st.session_state.itinerary_filters['radius_km']
			it_neighborhood = st.session_state.itinerary_filters['neighborhood']
			it_attraction_type = st.session_state.itinerary_filters['attraction_type']

		# Weather-aware hints and decisions
		weather = None
		if 'current_weather' in st.session_state and st.session_state.current_weather:
			weather = st.session_state.current_weather
		elif 'user_location' in st.session_state and st.session_state.user_location:
			try:
				weather = get_weather_data(origin_lat, origin_lon)
				st.session_state.current_weather = weather
			except Exception:
				weather = None
		bad_codes = {51,53,55,56,57,61,63,65,80,81,82,95,96,99}
		precip_prob = (weather or {}).get('precip_prob_next_3h') or 0
		code = (weather or {}).get('weather_code') or 0
		bad_weather = (precip_prob and precip_prob >= 50) or (code in bad_codes)
		if weather:
			wt = weather_based_tip(weather)
			if wt.get('text'):
				st.info(f"{wt['text']}")

		# Load dine data
		dine_path = f"{destination_info['dine']['folder']}/{destination_info['dine']['file']}"
		dine_df = pd.read_csv(dine_path)
		dine_df = dine_df.dropna(subset=["Latitude", "Longitude"]).copy()
		if 'score' in dine_df.columns:
			dine_df = dine_df[dine_df['score'] >= 0.6]
		# Neighborhood filter for dine
		if it_neighborhood != "Any" and 'area' in dine_df.columns:
			area_code = placeDict[it_neighborhood].lower()
			dine_df = dine_df[dine_df['area'] == area_code]
		# Compute distance
		dine_df['distance_km'] = dine_df.apply(
			lambda r: haversine_distance(origin_lat, origin_lon, float(r['Latitude']), float(r['Longitude'])), axis=1
		)
		# Sort: higher score first, then closer
		if 'score' in dine_df.columns:
			df_sel = dine_df.sort_values(by=['score', 'distance_km'], ascending=[False, True])
		else:
			df_sel = dine_df.sort_values(by=['distance_km'], ascending=[True])

		eat_choice = None
		if not df_sel.empty:
			# If mood is Hungry or Romantic, bias closer; else prefer highest score; then rotate by offset
			if it_mood in ("Hungry", "Romantic") and 'distance_km' in df_sel.columns and 'score' in df_sel.columns:
				df_sel = df_sel.sort_values(by=['distance_km','score'], ascending=[True, False])
			eat_choice = df_sel.iloc[offset % len(df_sel)]

		# Prepare parking suggestion near the chosen restaurant with weather preference
		parking_info = None
		if eat_choice is not None:
			# Load parking dataset and prefer covered during bad weather, else open
			try:
				pdf = load_parking_data()
				if not pdf.empty:
					def _cover_label(t):
						_t = str(t).lower()
						if any(k in _t for k in ["underground","garage","structure","covered","indoor"]):
							return "Covered"
						if any(k in _t for k in ["surface","lot","open","outdoor"]):
							return "Open"
						return "Open"
					pdf['Cover'] = pdf['Carpark Type'].apply(_cover_label)
					base_lat, base_lon = float(eat_choice['Latitude']), float(eat_choice['Longitude'])
					pdf['dist_km'] = pdf.apply(lambda r: haversine_distance(base_lat, base_lon, float(r['Lat']), float(r['Lon'])), axis=1)
					if bad_weather:
						cands = pdf[pdf['Cover'] == 'Covered']
						if cands.empty:
							cands = pdf
					else:
						cands = pdf[pdf['Cover'] == 'Open']
						if cands.empty:
							cands = pdf
					suggested = cands.sort_values('dist_km', ascending=True)
					if not suggested.empty:
						row = suggested.iloc[offset % len(suggested)].to_dict()
						parking_info = {
							'Closest Parking Name': row.get('Park Name'),
							'Closest Parking Distance (km)': row.get('dist_km'),
							'Closest Parking Rate': f"${row.get('Rate per 30min')}/30min",
							'Closest Parking Cover': row.get('Cover'),
							'Closest Parking Day Max': row.get('Day Maximum'),
							'Closest Parking Night Max': row.get('Night Maximum')
						}
			except Exception:
				parking_info = None

		# Nearby attractions: parks or libraries depending on filter and weather
		attractions = []
		try:
			base_lat, base_lon = (float(eat_choice['Latitude']), float(eat_choice['Longitude'])) if eat_choice is not None else (origin_lat, origin_lon)
			# Decide dataset
			preferred = it_attraction_type
			if preferred == "Any":
				preferred = "Libraries" if bad_weather else "Parks"
			if preferred == "Libraries":
				edu_path = f"{destination_info['education']['folder']}/{destination_info['education']['file']}"
				edu_df = pd.read_csv(edu_path)
				edu_df = edu_df.dropna(subset=["Lat", "Long"]).copy()
				if 'score' in edu_df.columns:
					edu_df = edu_df[edu_df['score'] >= 0.6]
				if it_neighborhood != "Any" and 'area' in edu_df.columns:
					area_code = placeDict[it_neighborhood].lower()
					edu_df = edu_df[edu_df['area'] == area_code]
				edu_df['distance_km'] = edu_df.apply(lambda r: haversine_distance(base_lat, base_lon, float(r['Lat']), float(r['Long'])), axis=1)
				edu_df = edu_df.sort_values(by=['distance_km'], ascending=[True])
				rows = list(edu_df.itertuples(index=False))
				for i in range(2):
					if len(rows) == 0:
						break
					j = (offset + i) % len(rows)
					r = rows[j]
					attractions.append({
						'ASSET_NAME': getattr(r, 'BranchName') if hasattr(r, 'BranchName') else 'Library',
						'score': getattr(r, 'score') if hasattr(r, 'score') else None,
						'distance_km': getattr(r, 'distance_km') if hasattr(r, 'distance_km') else None,
						'Latitude': getattr(r, 'Lat'),
						'Longitude': getattr(r, 'Long')
					})
			else:
				parks_path = f"{destination_info['parks']['folder']}/{destination_info['parks']['file']}"
				parks_df = pd.read_csv(parks_path)
				parks_df = parks_df.dropna(subset=["Latitude", "Longitude"]).copy()
				if 'score' in parks_df.columns:
					parks_df = parks_df[parks_df['score'] >= 0.6]
				if it_neighborhood != "Any" and 'area' in parks_df.columns:
					area_code = placeDict[it_neighborhood].lower()
					parks_df = parks_df[parks_df['area'] == area_code]
				parks_df['distance_km'] = parks_df.apply(lambda r: haversine_distance(base_lat, base_lon, float(r['Latitude']), float(r['Longitude'])), axis=1)
				parks_df = parks_df.sort_values(by=['distance_km'], ascending=[True])
				rows = list(parks_df.itertuples(index=False))
				for i in range(2):
					if len(rows) == 0:
						break
					j = (offset + i) % len(rows)
					r = rows[j]
					attractions.append({
						'ASSET_NAME': getattr(r, 'ASSET_NAME') if hasattr(r, 'ASSET_NAME') else 'Attraction',
						'score': getattr(r, 'score') if hasattr(r, 'score') else None,
						'distance_km': getattr(r, 'distance_km') if hasattr(r, 'distance_km') else None,
						'Latitude': getattr(r, 'Latitude'),
						'Longitude': getattr(r, 'Longitude')
					})
		except Exception:
			attractions = []

		# Increment iterator when generating new options
		if new_btn:
			st.session_state.itinerary_iter = st.session_state.itinerary_iter + 1

		# Render results
		st.markdown("### Your Itinerary")
		cols_res = st.columns([1, 1])
		with cols_res[0]:
			if eat_choice is not None:
				eat_name = eat_choice.get('Establishment Name', 'Top Restaurant')
				eat_score = eat_choice.get('score', None)
				eat_dist = eat_choice.get('distance_km', None)
				eat_lat = float(eat_choice['Latitude']); eat_lon = float(eat_choice['Longitude'])
				maps_url = f"https://www.google.com/maps/search/?api=1&query={eat_lat},{eat_lon}"
				score_text = f" (DineSafe score {eat_score:.2f})" if pd.notna(eat_score) else ""
				dist_text = f" — {eat_dist:.2f} km away" if pd.notna(eat_dist) else ""
				st.markdown(f"**🍽️ Place to eat:** [{eat_name}]({maps_url}){score_text}{dist_text}")
			else:
				st.warning("No nearby restaurants found.")

			if parking_info:
				p_name = parking_info.get('Closest Parking Name', 'N/A')
				p_dist = parking_info.get('Closest Parking Distance (km)', None)
				p_rate = parking_info.get('Closest Parking Rate', 'N/A')
				p_cover = parking_info.get('Closest Parking Cover', 'Open')
				p_day = parking_info.get('Closest Parking Day Max', 'N/A')
				p_night = parking_info.get('Closest Parking Night Max', 'N/A')
				pdist_text = f" — {p_dist:.2f} km" if isinstance(p_dist, (int, float)) else ""
				pref_emoji = "☔" if bad_weather else "☀️"
				pref_text = "(weather: prefer covered)" if bad_weather else "(weather: open is fine)"
				st.markdown(f"**🅿️ Parking:** {p_name}{pdist_text} · {p_cover} · {p_rate} · Day {p_day}, Night {p_night} {pref_emoji} {pref_text}")
			else:
				st.info("Parking suggestion unavailable.")
		with cols_res[1]:
			if attractions:
				st.markdown("**📍 Nearby attractions:**")
				for a in attractions:
					a_name = a.get('ASSET_NAME', a.get('BranchName', 'Attraction'))
					a_score = a.get('score', None)
					a_dist = a.get('distance_km', None)
					a_lat = float(a.get('Latitude'))
					a_lon = float(a.get('Longitude'))
					a_url = f"https://www.google.com/maps/search/?api=1&query={a_lat},{a_lon}"
					s_text = f" (score {a_score:.2f})" if pd.notna(a_score) else ""
					d_text = f" — {a_dist:.2f} km" if pd.notna(a_dist) else ""
					st.markdown(f"- [{a_name}]({a_url}){s_text}{d_text}")
			else:
				st.info("No nearby attractions found within the selected radius.")

	except Exception as e:
		st.warning(f"Could not build itinerary: {str(e)}")

# Top 5 Trending Places in Toronto This Week
st.markdown("---")
st.markdown("## 🔥 Top 5 Trending Places in Toronto This Week")
st.markdown("*Updated July 30, 2025*")

# Create trending places data
trending_places = [
    {
        "name": "CN Tower EdgeWalk",
        "type": "🏗️ Iconic Attraction",
        "description": "Toronto's most thrilling attraction - walk around the CN Tower's exterior ledge 116 stories above the ground",
        "why_trending": "Summer season brings perfect weather for the ultimate adrenaline rush",
        "image": "https://images.unsplash.com/photo-1517935706615-2717063c2225?w=400&h=200&fit=crop&crop=center",
        "maps_url": "https://www.google.com/maps/search/?api=1&query=43.6426,-79.3871"
    },
    {
        "name": "Summerlicious Restaurants",
        "type": "🍽️ Dining Experience", 
        "description": "220+ Toronto restaurants offering 3-course prix fixe menus from $20-$75",
        "why_trending": "Limited time event (July 4-20) featuring Toronto's hottest restaurants",
        "image": "https://images.unsplash.com/photo-1414235077428-338989a2e8c0?w=400&h=200&fit=crop&crop=center",
        "maps_url": "https://www.google.com/maps/search/?api=1&query=Toronto+restaurants"
    },
    {
        "name": "Ripley's Aquarium of Canada",
        "type": "🐠 Family Attraction",
        "description": "Walk through glass tunnels surrounded by sharks, rays, and 450+ marine species",
        "why_trending": "Perfect indoor escape from summer heat with Instagram-worthy moments",
        "image": "https://images.unsplash.com/photo-1544551763-46a013bb70d5?w=400&h=200&fit=crop&crop=center",
        "maps_url": "https://www.google.com/maps/search/?api=1&query=43.6426,-79.3858"
    },
    {
        "name": "Beaches Jazz Festival",
        "type": "🎵 Live Music",
        "description": "100+ free live performances, street dancing, and authentic food along Queen St East",
        "why_trending": "37th annual festival running July 4-27 with massive street parties",
        "image": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=400&h=200&fit=crop&crop=center",
        "maps_url": "https://www.google.com/maps/search/?api=1&query=Queen+Street+East+Beaches+Toronto"
    },
    {
        "name": "Casa Loma Gardens",
        "type": "🏰 Historic Castle",
        "description": "Gothic Revival mansion with blooming summer gardens and panoramic city views",
        "why_trending": "Peak garden season with Instagram-perfect European castle vibes",
        "image": "https://casaloma.ca/wp-content/uploads/2018/04/Gardens4-1.jpg",
        "maps_url": "https://www.google.com/maps/search/?api=1&query=43.6780,-79.4094"
    }
]

# Display trending places in a grid
for i in range(0, len(trending_places), 1):
    place = trending_places[i]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display image
        st.markdown(f"""
        <div style="
            background-image: url('{place['image']}');
            background-size: cover;
            background-position: center;
            height: 150px;
            border-radius: 10px;
            margin-bottom: 1rem;
        "></div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Display place info
        st.markdown(f"""
        <div style="padding-left: 1rem;">
            <h4 style="color: #667eea; margin-bottom: 0.5rem; font-size: 1.2rem;">
                #{i+1} {place['name']}
            </h4>
            <p style="color: #888; font-size: 0.9rem; margin-bottom: 0.5rem;">
                {place['type']}
            </p>
            <p style="color: #333; font-size: 0.95rem; margin-bottom: 0.8rem; line-height: 1.4;">
                {place['description']}
            </p>
            <p style="color: #667eea; font-size: 0.85rem; font-weight: bold; margin-bottom: 1rem;">
                🔥 Why it's trending: {place['why_trending']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation button
        st.link_button(
            f"📍 Visit {place['name']}", 
            url=place['maps_url'],
            use_container_width=True
        )
    
    st.markdown("---")




