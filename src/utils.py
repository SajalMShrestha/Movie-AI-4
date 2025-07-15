"""
Utility functions and constants for the movie recommendation system.
"""

import streamlit as st
import requests
import concurrent.futures
from sentence_transformers import SentenceTransformer
from tmdbv3api import Movie
import torch

# Global configuration constants
RECOMMENDATION_WEIGHTS = {
    "mood_tone": 0.15,
    "genre_similarity": 0.10,
    "cast_crew": 0.10,
    "narrative_style": 0.10,       # Increased from 0.08
    "ratings": 0.05,
    "trending_factor": 0.07,
    "release_year": 0.05,
    "discovery_boost": 0.10,       # Increased from 0.05
    "age_alignment": 0.0,
    "embedding_similarity": 0.28    # Reduced from 0.35
}

STREAMING_PLATFORM_PRIORITY = {
    "netflix": 1.0,
    "disney_plus": 0.9,
    "hbo_max": 0.85,
    "hulu": 0.8,
    "prime_video": 0.75,
    "apple_tv": 0.7,
    "peacock": 0.6,
    "paramount_plus": 0.5
}

MOOD_TONE_MAP = {
    "uplifting": {"Comedy", "Romance", "Music", "Adventure", "Family", "Animation"},
    "dark_gritty": {"Crime", "Thriller", "Mystery", "Horror"},
    "thought_provoking": {"Science Fiction", "Documentary", "History", "Fantasy"},
    "high_intensity": {"Action", "War", "Horror", "Thriller"},
    "emotional": {"Drama", "Romance", "History", "War"},
    "nostalgic": {"Western", "History", "War"},
    "neutral": {"TV Movie"}
}

def calculate_user_mood_preferences(favorite_movies_info):
    """Calculate user's mood preferences dynamically from their favorites."""
    mood_counts = {}
    total_movies = len(favorite_movies_info)
    
    if total_movies == 0:
        return {"uplifting": 0.5, "emotional": 0.3, "high_intensity": 0.2}
    
    for movie_info in favorite_movies_info:
        movie_genres = set(movie_info.get('genres', []))
        
        for mood, genre_set in MOOD_TONE_MAP.items():
            if movie_genres & genre_set:
                mood_counts[mood] = mood_counts.get(mood, 0) + 1
    
    # Convert counts to normalized weights
    mood_weights = {}
    for mood, count in mood_counts.items():
        mood_weights[mood] = count / total_movies
    
    # Ensure all moods have minimal weight for discovery
    for mood in MOOD_TONE_MAP.keys():
        if mood not in mood_weights:
            mood_weights[mood] = 0.05
    
    return mood_weights

def get_mood_score(genres, user_mood_weights):
    """Calculate mood score using dynamic user preferences."""
    if not user_mood_weights:
        return 0.5
    
    movie_moods = set()
    genre_names = set()
    
    # Extract genre names
    for g in genres:
        genre_name = g.get('name', '') if isinstance(g, dict) else getattr(g, 'name', '')
        if genre_name:
            genre_names.add(genre_name)
    
    # Find matching moods
    for mood, genre_set in MOOD_TONE_MAP.items():
        if genre_names & genre_set:
            movie_moods.add(mood)
    
    if not movie_moods:
        return 0.1
    
    # Calculate average preference across matched moods
    total_preference = sum(user_mood_weights.get(mood, 0.05) for mood in movie_moods)
    base_score = total_preference / len(movie_moods)
    
    return min(base_score, 1.0)

@st.cache_resource
def get_embedding_model():
    """Get the sentence transformer model for embeddings."""
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

def fetch_similar_movie_details(m_id, fetch_cache=None):
    """
    Fetch detailed movie information including credits and generate embeddings.
    
    Args:
        m_id: Movie ID from TMDB
        fetch_cache: Cache dictionary to store results
    
    Returns:
        Tuple of (movie_id, (movie_details, embedding)) or (movie_id, None)
    """
    if fetch_cache is None:
        fetch_cache = {}
    
    if m_id in fetch_cache:
        return m_id, fetch_cache[m_id]
    
    try:
        movie_api = Movie()
        m_details = movie_api.details(m_id)
        m_credits = movie_api.credits(m_id)

        # Robust genre, cast, director extraction
        genres = []
        genres_list = getattr(m_details, 'genres', [])
        for g in genres_list:
            if isinstance(g, dict):
                name = g.get('name', '')
            else:
                name = getattr(g, 'name', '')
            if name:
                genres.append(name)

        cast_list_raw = m_credits.get('cast', []) if isinstance(m_credits, dict) else getattr(m_credits, 'cast', [])
        crew_list = m_credits.get('crew', []) if isinstance(m_credits, dict) else getattr(m_credits, 'crew', [])
        
        if hasattr(cast_list_raw, '__iter__'):
            m_details.cast = list(cast_list_raw)[:3] if cast_list_raw else []
        else:
            m_details.cast = []

        directors = []
        for c in crew_list:
            is_director = False
            name = ''
            if isinstance(c, dict):
                is_director = c.get('job', '') == 'Director'
                name = c.get('name', '')
            else:
                is_director = getattr(c, 'job', '') == 'Director'
                name = getattr(c, 'name', '')
            
            if is_director and name:
                directors.append(name)
        
        m_details.directors = directors
        m_details.plot = getattr(m_details, 'overview', '') or ''

        # Skip if plot is missing or too short
        if not m_details.plot or len(m_details.plot.split()) < 5:
            fetch_cache[m_id] = None
            return m_id, None

        # Import narrative analysis function
        from .narrative_analysis import infer_narrative_style
        m_details.narrative_style = infer_narrative_style(m_details.plot)

        # Generate embedding
        embedding_model = get_embedding_model()
        embedding = embedding_model.encode(m_details.plot, convert_to_tensor=True)

        fetch_cache[m_id] = (m_details, embedding)
        return m_id, (m_details, embedding)

    except Exception as e:
        fetch_cache[m_id] = None
        return m_id, None

def get_trending_popularity(api_key):
    """Fetch and normalize trending popularity scores."""
    try:
        url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get("results", [])
            if not data:
                return {}
            max_pop = max([m.get("popularity", 1) for m in data])
            return {m["id"]: m.get("popularity", 0) / max_pop for m in data}
    except:
        return {}

def estimate_user_age(years):
    """Estimate user age based on release years of favorite movies."""
    if not years:
        return 30
    from datetime import datetime
    median = sorted(years)[len(years)//2]
    return datetime.now().year - median + 18

def fetch_multiple_movie_details(movie_ids, fetch_cache):
    """Fetch details for multiple movies using threading."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_id = {
            executor.submit(fetch_similar_movie_details, mid, fetch_cache): mid 
            for mid in movie_ids[:50]
        }
        for future in concurrent.futures.as_completed(future_to_id):
            mid = future_to_id[future]
            try:
                result = future.result()
                if result and result[1]:
                    results[mid] = result[1]
            except:
                pass
    return results