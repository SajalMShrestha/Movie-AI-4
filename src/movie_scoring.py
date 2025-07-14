"""
Enhanced movie recommendation scoring and candidate pool building with multi-page support.
"""

import streamlit as st
import requests
import concurrent.futures
import numpy as np
import torch
from datetime import datetime
from tmdbv3api import Movie
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sentence_transformers.util import cos_sim
import time
import random

from .utils import (
    RECOMMENDATION_WEIGHTS, get_embedding_model, get_mood_score, 
    get_trending_popularity, estimate_user_age, fetch_similar_movie_details
)
from .narrative_analysis import infer_narrative_style, infer_mood_from_plot, compute_narrative_similarity
from .franchise_detection import apply_final_franchise_limit
from .movie_search import fuzzy_search_movies

def build_enhanced_candidate_pool(favorite_genre_ids, favorite_cast_ids, favorite_director_ids, 
                                favorite_years, tmdb_api_key, target_pool_size=300):
    """
    Build an enhanced pool of candidate movies using multi-page fetching and diverse strategies.
    
    Args:
        favorite_genre_ids: Set of favorite genre IDs
        favorite_cast_ids: Set of favorite cast member IDs
        favorite_director_ids: Set of favorite director IDs
        favorite_years: List of favorite movie years
        tmdb_api_key: TMDB API key
        target_pool_size: Target size for the candidate pool (default: 300)
    
    Returns:
        Set of candidate movie IDs
    """
    candidate_movie_ids = set()
    
    def safe_api_call(url, params, strategy_name, max_retries=3):
        """Make API call with retry logic and rate limiting."""
        for attempt in range(max_retries):
            try:
                time.sleep(0.1)  # Rate limiting
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt
                    st.warning(f"Rate limited for {strategy_name}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    st.warning(f"API error for {strategy_name}: {response.status_code}")
                    return None
            except Exception as e:
                if attempt == max_retries - 1:
                    st.warning(f"Error in {strategy_name} after {max_retries} attempts: {e}")
                time.sleep(1)
        return None
    
    def fetch_multiple_pages(base_url, base_params, strategy_name, pages=3, movies_per_page=20):
        """Fetch multiple pages for a given strategy."""
        strategy_movies = set()
        
        for page in range(1, pages + 1):
            params = base_params.copy()
            params["page"] = page
            
            data = safe_api_call(base_url, params, f"{strategy_name} (page {page})")
            if data and "results" in data:
                movies = data["results"]
                strategy_movies.update([m["id"] for m in movies[:movies_per_page]])
                
                # Stop if we get fewer results than expected (last page)
                if len(movies) < 20:
                    break
            else:
                break
                
        return strategy_movies
    
    # Strategy 1: Enhanced Genre Discovery (60-90 movies)
    st.write("🎭 Building genre-based candidates...")
    genre_movies = set()
    
    for genre_id in list(favorite_genre_ids)[:4]:  # Increased from 3 to 4
        # High popularity, recent movies
        base_params = {
            "api_key": tmdb_api_key,
            "with_genres": str(genre_id),
            "sort_by": "popularity.desc",
            "vote_count.gte": 50,
            "primary_release_date.gte": "2015-01-01"
        }
        recent_movies = fetch_multiple_pages(
            "https://api.themoviedb.org/3/discover/movie", 
            base_params, 
            f"Genre {genre_id} (Recent Popular)", 
            pages=2
        )
        genre_movies.update(recent_movies)
        
        # High-rated classics in this genre
        base_params.update({
            "sort_by": "vote_average.desc",
            "vote_count.gte": 200,
            "vote_average.gte": 7.5,
            "primary_release_date.gte": "1990-01-01",
            "primary_release_date.lte": "2020-12-31"
        })
        classic_movies = fetch_multiple_pages(
            "https://api.themoviedb.org/3/discover/movie", 
            base_params, 
            f"Genre {genre_id} (Classics)", 
            pages=2
        )
        genre_movies.update(classic_movies)
    
    candidate_movie_ids.update(genre_movies)
    st.write(f"   Added {len(genre_movies)} genre-based movies")
    
    # Strategy 2: Enhanced Cast Discovery (50-70 movies)
    st.write("🎬 Building cast-based candidates...")
    cast_movies = set()
    
    for person_id in list(favorite_cast_ids)[:6]:  # Increased from 5 to 6
        # Popular movies with this actor
        base_params = {
            "api_key": tmdb_api_key,
            "with_cast": str(person_id),
            "sort_by": "popularity.desc",
            "vote_count.gte": 30
        }
        popular_cast_movies = fetch_multiple_pages(
            "https://api.themoviedb.org/3/discover/movie", 
            base_params, 
            f"Cast {person_id} (Popular)", 
            pages=2
        )
        cast_movies.update(popular_cast_movies)
        
        # Highest-rated movies with this actor
        base_params.update({
            "sort_by": "vote_average.desc",
            "vote_count.gte": 100,
            "vote_average.gte": 7.0
        })
        rated_cast_movies = fetch_multiple_pages(
            "https://api.themoviedb.org/3/discover/movie", 
            base_params, 
            f"Cast {person_id} (Rated)", 
            pages=1
        )
        cast_movies.update(rated_cast_movies)
    
    candidate_movie_ids.update(cast_movies)
    st.write(f"   Added {len(cast_movies)} cast-based movies")
    
    # Strategy 3: Enhanced Director Discovery (40-60 movies)
    st.write("🎯 Building director-based candidates...")
    director_movies = set()
    
    for person_id in list(favorite_director_ids)[:4]:  # Increased from 3 to 4
        base_params = {
            "api_key": tmdb_api_key,
            "with_crew": str(person_id),
            "sort_by": "popularity.desc",
            "vote_count.gte": 30
        }
        popular_director_movies = fetch_multiple_pages(
            "https://api.themoviedb.org/3/discover/movie", 
            base_params, 
            f"Director {person_id} (Popular)", 
            pages=2
        )
        director_movies.update(popular_director_movies)
        
        # Critically acclaimed works
        base_params.update({
            "sort_by": "vote_average.desc",
            "vote_count.gte": 50,
            "vote_average.gte": 7.0
        })
        acclaimed_director_movies = fetch_multiple_pages(
            "https://api.themoviedb.org/3/discover/movie", 
            base_params, 
            f"Director {person_id} (Acclaimed)", 
            pages=1
        )
        director_movies.update(acclaimed_director_movies)
    
    candidate_movie_ids.update(director_movies)
    st.write(f"   Added {len(director_movies)} director-based movies")
    
    # Strategy 4: Enhanced Temporal Discovery (40-60 movies)
    st.write("📅 Building year-based candidates...")
    temporal_movies = set()
    
    if favorite_years:
        decades = set()
        for year in favorite_years:
            decade_start = (year // 10) * 10
            decades.add(decade_start)
        
        for decade_start in list(decades)[:3]:  # Increased coverage
            # Popular movies from this decade
            base_params = {
                "api_key": tmdb_api_key,
                "primary_release_date.gte": f"{decade_start}-01-01",
                "primary_release_date.lte": f"{decade_start + 9}-12-31",
                "sort_by": "popularity.desc",
                "vote_count.gte": 100
            }
            decade_popular = fetch_multiple_pages(
                "https://api.themoviedb.org/3/discover/movie", 
                base_params, 
                f"Decade {decade_start}s (Popular)", 
                pages=2
            )
            temporal_movies.update(decade_popular)
            
            # Highly-rated movies from this decade
            base_params.update({
                "sort_by": "vote_average.desc",
                "vote_count.gte": 200,
                "vote_average.gte": 7.0
            })
            decade_rated = fetch_multiple_pages(
                "https://api.themoviedb.org/3/discover/movie", 
                base_params, 
                f"Decade {decade_start}s (Rated)", 
                pages=1
            )
            temporal_movies.update(decade_rated)
    
    candidate_movie_ids.update(temporal_movies)
    st.write(f"   Added {len(temporal_movies)} temporal-based movies")
    
    # Strategy 5: Enhanced Multi-Criteria Discovery (30-50 movies)
    st.write("🔀 Building multi-criteria candidates...")
    multi_criteria_movies = set()
    
    try:
        # Combine top genres and cast
        top_genres = list(favorite_genre_ids)[:2]
        top_cast = list(favorite_cast_ids)[:3]
        
        for genre_combo in range(len(top_genres)):
            for cast_combo in range(len(top_cast)):
                base_params = {
                    "api_key": tmdb_api_key,
                    "with_genres": str(top_genres[genre_combo]),
                    "with_cast": str(top_cast[cast_combo]),
                    "sort_by": "popularity.desc",
                    "vote_count.gte": 20
                }
                combo_movies = fetch_multiple_pages(
                    "https://api.themoviedb.org/3/discover/movie", 
                    base_params, 
                    "Multi-criteria", 
                    pages=1,
                    movies_per_page=10
                )
                multi_criteria_movies.update(combo_movies)
                
                if len(multi_criteria_movies) >= 50:
                    break
            if len(multi_criteria_movies) >= 50:
                break
    except Exception as e:
        st.warning(f"Error with multi-criteria discovery: {e}")
    
    candidate_movie_ids.update(multi_criteria_movies)
    st.write(f"   Added {len(multi_criteria_movies)} multi-criteria movies")
    
    # Strategy 6: Enhanced Trending and Awards (30-40 movies)
    st.write("🔥 Building trending and award-winning candidates...")
    trending_movies = set()
    
    # Current trending
    data = safe_api_call(
        "https://api.themoviedb.org/3/trending/movie/week", 
        {"api_key": tmdb_api_key}, 
        "Trending Weekly"
    )
    if data and "results" in data:
        trending_movies.update([m["id"] for m in data["results"][:20]])
    
    # Award season favorites (high-rated recent releases)
    base_params = {
        "api_key": tmdb_api_key,
        "sort_by": "vote_average.desc",
        "vote_count.gte": 500,
        "vote_average.gte": 7.5,
        "primary_release_date.gte": "2020-01-01"
    }
    award_movies = fetch_multiple_pages(
        "https://api.themoviedb.org/3/discover/movie", 
        base_params, 
        "Award Contenders", 
        pages=2
    )
    trending_movies.update(award_movies)
    
    candidate_movie_ids.update(trending_movies)
    st.write(f"   Added {len(trending_movies)} trending/award movies")
    
    # Strategy 7: Discovery Serendipity (20-30 movies)
    st.write("🎲 Building serendipitous discoveries...")
    serendipity_movies = set()
    
    # Random high-quality movies from adjacent genres
    all_genres = [28, 12, 16, 35, 80, 99, 18, 10751, 14, 36, 27, 10402, 9648, 10749, 878, 10770, 53, 10752, 37]
    adjacent_genres = [g for g in all_genres if g not in favorite_genre_ids]
    
    for genre_id in random.sample(adjacent_genres, min(3, len(adjacent_genres))):
        base_params = {
            "api_key": tmdb_api_key,
            "with_genres": str(genre_id),
            "sort_by": "vote_average.desc",
            "vote_count.gte": 300,
            "vote_average.gte": 7.0
        }
        serendipity_genre_movies = fetch_multiple_pages(
            "https://api.themoviedb.org/3/discover/movie", 
            base_params, 
            f"Serendipity Genre {genre_id}", 
            pages=1,
            movies_per_page=10
        )
        serendipity_movies.update(serendipity_genre_movies)
    
    candidate_movie_ids.update(serendipity_movies)
    st.write(f"   Added {len(serendipity_movies)} serendipitous movies")
    
    # Strategy 8: Deep Cuts and Hidden Gems (20-30 movies)
    st.write("💎 Building hidden gems...")
    hidden_gems = set()
    
    for genre_id in list(favorite_genre_ids)[:2]:
        # Lower popularity but high ratings (hidden gems)
        base_params = {
            "api_key": tmdb_api_key,
            "with_genres": str(genre_id),
            "sort_by": "vote_average.desc",
            "vote_count.gte": 100,
            "vote_count.lte": 1000,  # Not too popular
            "vote_average.gte": 7.5
        }
        gems = fetch_multiple_pages(
            "https://api.themoviedb.org/3/discover/movie", 
            base_params, 
            f"Hidden Gems {genre_id}", 
            pages=2,
            movies_per_page=10
        )
        hidden_gems.update(gems)
    
    candidate_movie_ids.update(hidden_gems)
    st.write(f"   Added {len(hidden_gems)} hidden gems")
    
    # Remove duplicates and apply intelligent limiting
    final_candidates = list(candidate_movie_ids)
    
    # If we have too many candidates, prioritize based on recency and quality
    if len(final_candidates) > target_pool_size:
        st.write(f"🎯 Optimizing pool from {len(final_candidates)} to {target_pool_size} movies...")
        
        # This would ideally involve fetching basic info and scoring, 
        # but for now we'll randomly sample to avoid bias
        final_candidates = random.sample(final_candidates, target_pool_size)
    
    st.write(f"✅ Built candidate pool with {len(final_candidates)} movies")
    
    return set(final_candidates)

def identify_taste_clusters(favorite_embeddings, favorite_movies_info):
    """
    Identify distinct taste clusters from user's favorite movies.
    
    Args:
        favorite_embeddings: List of movie embeddings
        favorite_movies_info: List of movie info dictionaries
    
    Returns:
        Tuple of (cluster_centers, cluster_labels)
    """
    if len(favorite_embeddings) <= 2:
        return None, None
    
    # Convert embeddings to numpy array
    embeddings_array = torch.stack(favorite_embeddings).cpu().numpy()
    
    # Determine optimal number of clusters
    n_clusters = min(3, max(2, len(favorite_embeddings) // 2))
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    cluster_centers = kmeans.cluster_centers_
    
    # Convert back to torch tensors
    cluster_centers_torch = [torch.from_numpy(center) for center in cluster_centers]
    
    return cluster_centers_torch, cluster_labels

def compute_multi_cluster_similarity(candidate_embedding, cluster_centers):
    """
    Compute similarity to multiple cluster centers.
    
    Args:
        candidate_embedding: Embedding of candidate movie
        cluster_centers: List of cluster center embeddings
    
    Returns:
        Float: Maximum similarity to any cluster
    """
    if cluster_centers is None:
        return 0.0
    
    max_similarity = 0.0
    for center in cluster_centers:
        similarity = float(cos_sim(candidate_embedding, center))
        max_similarity = max(max_similarity, similarity)
    
    return max_similarity

def analyze_taste_diversity(favorite_embeddings, favorite_genres, favorite_years):
    """
    Analyze how diverse the user's taste is.
    
    Args:
        favorite_embeddings: List of movie embeddings
        favorite_genres: Set of favorite genres
        favorite_years: List of favorite years
    
    Returns:
        Dictionary with diversity metrics and taste profile
    """
    diversity_metrics = {
        "genre_diversity": len(favorite_genres) / 5.0,
        "temporal_spread": 0.0,
        "embedding_variance": 0.0,
        "taste_profile": "focused"
    }
    
    # Temporal spread
    if len(favorite_years) > 1:
        year_range = max(favorite_years) - min(favorite_years)
        diversity_metrics["temporal_spread"] = min(year_range / 50.0, 1.0)
    
    # Embedding variance
    if len(favorite_embeddings) > 1:
        embeddings_array = torch.stack(favorite_embeddings).cpu().numpy()
        pairwise_similarities = sklearn_cosine_similarity(embeddings_array)
        mask = ~np.eye(pairwise_similarities.shape[0], dtype=bool)
        avg_similarity = pairwise_similarities[mask].mean()
        diversity_metrics["embedding_variance"] = 1.0 - avg_similarity
    
    # Determine taste profile
    overall_diversity = (diversity_metrics["genre_diversity"] + 
                        diversity_metrics["temporal_spread"] + 
                        diversity_metrics["embedding_variance"]) / 3.0
    
    if overall_diversity < 0.3:
        diversity_metrics["taste_profile"] = "focused"
    elif overall_diversity < 0.6:
        diversity_metrics["taste_profile"] = "diverse"
    else:
        diversity_metrics["taste_profile"] = "eclectic"
    
    return diversity_metrics

def compute_score(m, cluster_centers, diversity_metrics, favorite_genres, favorite_actors, 
                 user_prefs, trending_scores, favorite_narrative_styles, candidate_movies, normalized_embedding_scores=None):
    """
    Compute recommendation score for a movie.
    Returns: (total_score, score_breakdown_dict)
    """
    try:
        narrative = getattr(m, 'narrative_style', {})
        score_components = {}  # Track each component
        
        # Genre similarity
        genres = set()
        genres_list = getattr(m, 'genres', [])
        for g in genres_list:
            if isinstance(g, dict):
                name = g.get('name', '')
            else:
                name = getattr(g, 'name', '')
            if name:
                genres.add(name)
        
        genre_score = RECOMMENDATION_WEIGHTS['genre_similarity'] * (len(genres & favorite_genres) / max(len(favorite_genres),1))
        score_components['genre_similarity'] = genre_score
        
        # Cast and crew similarity
        cast_names = set()
        cast_list = getattr(m, 'cast', [])
        for actor in cast_list:
            if isinstance(actor, dict):
                name = actor.get('name', '')
            else:
                name = getattr(actor, 'name', '')
            if name:
                cast_names.add(name)
        
        directors = getattr(m, 'directors', [])
        director_names = set(directors) if isinstance(directors, list) else set()
        
        cast_dir = cast_names | director_names
        cast_score = RECOMMENDATION_WEIGHTS['cast_crew'] * (len(cast_dir & favorite_actors) / max(len(favorite_actors),1))
        score_components['cast_crew'] = cast_score
        
        # Release year scoring
        recency_score = 0
        try:
            release_date = getattr(m, 'release_date', None)
            if release_date:
                year_diff = datetime.now().year - int(release_date[:4])
                if year_diff<=2: recency_score = RECOMMENDATION_WEIGHTS['release_year']*0.2
                elif year_diff<=5: recency_score = RECOMMENDATION_WEIGHTS['release_year']*0.1
                elif year_diff<=10: recency_score = RECOMMENDATION_WEIGHTS['release_year']*0.05
                elif year_diff<=20: recency_score = RECOMMENDATION_WEIGHTS['release_year']*0.02
        except (ValueError, TypeError, AttributeError):
            pass
        score_components['recency'] = recency_score
        
        # Ratings score
        vote_average = getattr(m, 'vote_average', 0) or 0
        ratings_score = RECOMMENDATION_WEIGHTS['ratings'] * (vote_average/10)
        score_components['ratings'] = ratings_score
        
        # Mood/tone score
        movie_genres = getattr(m, 'genres', [])
        mood_score = RECOMMENDATION_WEIGHTS['mood_tone'] * get_mood_score(movie_genres, user_prefs['preferred_moods'])
        score_components['mood_tone'] = mood_score

        # Narrative style score
        plot = getattr(m, 'plot', '') or getattr(m, 'overview', '') or ''
        narrative = infer_narrative_style(plot)
        narrative_match_score = compute_narrative_similarity(narrative, favorite_narrative_styles)
        narrative_score = RECOMMENDATION_WEIGHTS['narrative_style'] * narrative_match_score
        score_components['narrative_style'] = narrative_score

        # Embedding similarity score
        embedding_score = 0
        movie_id = getattr(m, 'id', None)
        
        # Use normalized embedding scores if available
        if normalized_embedding_scores and movie_id in normalized_embedding_scores:
            embedding_score = RECOMMENDATION_WEIGHTS['embedding_similarity'] * normalized_embedding_scores[movie_id]
        elif movie_id and movie_id in candidate_movies:
            candidate_data = candidate_movies[movie_id]
            if len(candidate_data) >= 2 and candidate_data[1] is not None:
                candidate_embedding = candidate_data[1]
                
                # Use multi-cluster similarity for diverse tastes
                if cluster_centers and diversity_metrics['taste_profile'] in ['diverse', 'eclectic']:
                    embedding_sim_score = compute_multi_cluster_similarity(candidate_embedding, cluster_centers)
                else:
                    # For focused tastes, use average embedding
                    embedding_model = get_embedding_model()
                    if hasattr(user_prefs, 'favorite_embeddings') and user_prefs['favorite_embeddings']:
                        avg_embedding = torch.mean(torch.stack(user_prefs['favorite_embeddings']), dim=0)
                        embedding_sim_score = float(cos_sim(candidate_embedding, avg_embedding))
                    else:
                        embedding_sim_score = 0.0
                
                embedding_score = RECOMMENDATION_WEIGHTS['embedding_similarity'] * embedding_sim_score
        score_components['embedding_similarity'] = embedding_score

        # Trending factor
        movie_trend_score = trending_scores.get(getattr(m, 'id', 0), 0)
        mood_match_score = get_mood_score(movie_genres, user_prefs['preferred_moods'])
        genre_overlap_score = len(genres & favorite_genres) / max(len(favorite_genres), 1)

        # Adjust trending boost based on taste diversity
        if diversity_metrics['taste_profile'] == 'eclectic':
            trending_weight = RECOMMENDATION_WEIGHTS['trending_factor'] * 1.5
        elif diversity_metrics['taste_profile'] == 'focused':
            if mood_match_score > 0.5 and genre_overlap_score > 0.4:
                trending_weight = RECOMMENDATION_WEIGHTS['trending_factor']
            else:
                trending_weight = 0
        else:
            if mood_match_score > 0.3 and genre_overlap_score > 0.2:
                trending_weight = RECOMMENDATION_WEIGHTS['trending_factor']
            else:
                trending_weight = 0
        
        trending_score = trending_weight * movie_trend_score
        score_components['trending'] = trending_score

        # Discovery boost for eclectic users
        discovery_score = 0
        if diversity_metrics['taste_profile'] == 'eclectic':
            if 0.1 < genre_overlap_score < 0.5:
                discovery_score = RECOMMENDATION_WEIGHTS['discovery_boost'] * 1.5
        score_components['discovery_boost'] = discovery_score

        # Age penalty for very old movies
        age_penalty = 0
        try:
            if release_date:
                release_year = int(release_date[:4])
                if datetime.now().year - release_year > 20:
                    age_penalty = -0.03
        except (ValueError, TypeError):
            pass
        score_components['age_penalty'] = age_penalty

        # Age alignment scoring
        age_alignment_score = 0
        try:
            if release_date:
                release_year = int(release_date[:4])
                user_age_at_release = user_prefs['estimated_age'] - (datetime.now().year - release_year)
                if 15 <= user_age_at_release <= 25:
                    age_alignment_score = RECOMMENDATION_WEIGHTS['age_alignment']
                elif 10 <= user_age_at_release < 15 or 25 < user_age_at_release <= 30:
                    age_alignment_score = RECOMMENDATION_WEIGHTS['age_alignment'] * 0.5
        except (ValueError, TypeError):
            pass
        score_components['age_alignment'] = age_alignment_score
        
        # Calculate total score
        total_score = max(sum(score_components.values()), 0)

        # Create score breakdown dictionary
        score_breakdown = {
            'genre_similarity': RECOMMENDATION_WEIGHTS['genre_similarity'] * (len(genres & favorite_genres) / max(len(favorite_genres),1)),
            'cast_crew': RECOMMENDATION_WEIGHTS['cast_crew'] * (len(cast_dir & favorite_actors) / max(len(favorite_actors),1)),
            'ratings': RECOMMENDATION_WEIGHTS['ratings'] * (vote_average/10),
            'mood_tone': RECOMMENDATION_WEIGHTS['mood_tone'] * get_mood_score(movie_genres, user_prefs['preferred_moods']),
            'embedding_similarity': embedding_score if 'embedding_score' in locals() else 0,
            'trending': trending_weight * movie_trend_score if 'trending_weight' in locals() and 'movie_trend_score' in locals() else 0,
            'recency_bonus': recency_score if 'recency_score' in locals() else 0
        }

        return total_score, score_breakdown
        
    except Exception as e:
        st.warning(f"Error computing score for movie: {e}")
        return 0, {}

def recommend_movies(favorite_titles, debug=False):
    """
    Main recommendation function that processes user's favorite movies and returns recommendations.
    
    Args:
        favorite_titles: List of user's favorite movie titles
        debug: Boolean flag to enable debug output for scoring breakdowns
    
    Returns:
        Tuple of (recommendations, candidate_movies)
    """
    # Check cache first
    cache_key = "|".join(sorted(favorite_titles))
    
    if cache_key in st.session_state.recommendation_cache:
        cached_result = st.session_state.recommendation_cache[cache_key]
        st.write(f"✅ Using cached results")
        return cached_result
    
    # Initialize collections
    favorite_genres = set()
    favorite_actors = set()
    favorite_directors = set()
    favorite_genre_ids = set()
    favorite_cast_ids = set()
    favorite_director_ids = set()
    plot_moods, favorite_years = set(), []
    favorite_narrative_styles = {"tone": [], "complexity": [], "genre_indicator": [], "setting_context": []}
    favorite_embeddings = []
    favorite_movies_info = []

    # Process favorite movies with fuzzy search fallback
    valid_movies_found = []
    failed_searches = []
    movie_api = Movie()

    for title in favorite_titles:
        try:
            search_result = movie_api.search(title)
            
            if search_result:
                valid_movies_found.append((title, search_result[0]))
            else:
                # Try fuzzy search for this title
                st.write(f"🔍 Trying fuzzy search for '{title}'...")
                fuzzy_results = fuzzy_search_movies(title, max_results=3, similarity_threshold=0.7)
                
                if fuzzy_results:
                    best_match = fuzzy_results[0]
                    st.write(f"📝 Using '{best_match['title']}' as match for '{title}' ({best_match['similarity']:.0%} similarity)")
                    
                    corrected_search = movie_api.search(best_match['title'])
                    if corrected_search:
                        valid_movies_found.append((title, corrected_search[0]))
                    else:
                        failed_searches.append(title)
                else:
                    failed_searches.append(title)
                    
        except Exception as e:
            st.warning(f"Error processing {title}: {e}")
            failed_searches.append(title)

    # Show search results
    if failed_searches:
        st.warning(f"⚠️ Could not find matches for: {', '.join(failed_searches)}")
        st.info("💡 Try using more common titles or check spelling for better results")

    if len(valid_movies_found) < 3:
        st.error("❌ Need at least 3 valid movies to generate good recommendations")
        st.info("💡 Please add more movies or try different titles")
        return [], {}

    # Process valid movies and extract features
    for original_title, search_result in valid_movies_found:
        try:
            movie_id = search_result.id
            
            # Check cache
            if movie_id in st.session_state.movie_details_cache:
                details = st.session_state.movie_details_cache[movie_id]
                credits = st.session_state.movie_credits_cache[movie_id]
            else:
                details = movie_api.details(movie_id)
                credits = movie_api.credits(movie_id)
                st.session_state.movie_details_cache[movie_id] = details
                st.session_state.movie_credits_cache[movie_id] = credits
            
            # Extract features (genres, cast, directors, etc.)
            movie_info = {"title": original_title, "genres": [], "year": None}
            
            # Process genres
            genres_list = getattr(details, 'genres', [])
            for g in genres_list:
                if isinstance(g, dict):
                    name = g.get('name', '')
                    if hasattr(g, 'id'):
                        favorite_genre_ids.add(g.id)
                    elif 'id' in g:
                        favorite_genre_ids.add(g['id'])
                else:
                    name = getattr(g, 'name', '')
                    if hasattr(g, 'id'):
                        favorite_genre_ids.add(g.id)
                if name:
                    favorite_genres.add(name)
                    movie_info["genres"].append(name)

            # Process cast and crew
            cast_list_raw = credits.get('cast', []) if isinstance(credits, dict) else getattr(credits, 'cast', [])
            crew_list = credits.get('crew', []) if isinstance(credits, dict) else getattr(credits, 'crew', [])
            
            if hasattr(cast_list_raw, '__iter__'):
                cast_list = list(cast_list_raw)[:3] if cast_list_raw else []
            else:
                cast_list = []
            
            for c in cast_list:
                if isinstance(c, dict):
                    name = c.get('name', '')
                    cast_id = c.get('id', 0)
                else:
                    name = getattr(c, 'name', '')
                    cast_id = getattr(c, 'id', 0)
                if name:
                    favorite_actors.add(name)
                if cast_id:
                    favorite_cast_ids.add(cast_id)

            for c in crew_list:
                is_director = False
                name = ''
                person_id = 0
                if isinstance(c, dict):
                    is_director = c.get('job', '') == 'Director'
                    name = c.get('name', '')
                    person_id = c.get('id', 0)
                else:
                    is_director = getattr(c, 'job', '') == 'Director'
                    name = getattr(c, 'name', '')
                    person_id = getattr(c, 'id', 0)
                
                if is_director and name:
                    favorite_directors.add(name)
                if is_director and person_id:
                    favorite_director_ids.add(person_id)

            # Process plot and narrative
            overview = getattr(details, 'overview', '') or ''
            plot_moods.add(infer_mood_from_plot(overview))
            narr_style = infer_narrative_style(overview)
            for key in favorite_narrative_styles:
                favorite_narrative_styles[key].append(narr_style.get(key, ""))
            
            # Process release date
            release_date = getattr(details, 'release_date', None)
            if release_date:
                try:
                    year = int(release_date[:4])
                    favorite_years.append(year)
                    movie_info["year"] = year
                except (ValueError, TypeError):
                    pass
            
            # Generate embedding
            embedding_model = get_embedding_model()
            emb = embedding_model.encode(overview, convert_to_tensor=True)
            favorite_embeddings.append(emb)
            favorite_movies_info.append(movie_info)
                
        except Exception as e:
            st.warning(f"Error processing {original_title}: {e}")
            continue

    # Build candidate pool and analyze taste
    from tmdbv3api import TMDb
    tmdb = TMDb()
    
    st.write("🔍 Building enhanced candidate movie pool...")
    candidate_movie_ids = build_enhanced_candidate_pool(
        favorite_genre_ids, favorite_cast_ids, favorite_director_ids, 
        favorite_years, tmdb.api_key, target_pool_size=300
    )

    # Convert to list for processing
    candidate_movie_ids = list(candidate_movie_ids)

    # Analyze taste diversity
    diversity_metrics = analyze_taste_diversity(favorite_embeddings, favorite_genres, favorite_years)
    
    # Identify taste clusters
    cluster_centers, cluster_labels = identify_taste_clusters(favorite_embeddings, favorite_movies_info)

    # Set up user preferences
    user_prefs = {
        "preferred_moods": plot_moods,
        "estimated_age": estimate_user_age(favorite_years),
        "taste_diversity": diversity_metrics,
        "favorite_embeddings": favorite_embeddings
    }

    # Fetch candidate movie details
    candidate_movies = {}
    fetch_cache = st.session_state.fetch_cache
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_similar_movie_details, mid, fetch_cache): mid for mid in candidate_movie_ids}
        for fut in concurrent.futures.as_completed(futures):
            try:
                result = fut.result()
                if result is None:
                    continue
                mid, payload = result
                if payload is None:
                    continue
                m, embedding = payload
                if m is None or embedding is None:
                    continue
                vote_count = getattr(m, 'vote_count', 0)
                if vote_count < 20:
                    continue
                candidate_movies[mid] = (m, embedding)
            except Exception as e:
                st.warning(f"Error processing candidate movie: {e}")
                continue

    # Update session cache
    st.session_state.fetch_cache.update(fetch_cache)

    if not candidate_movies:
        st.warning("No candidate movies with valid plots or embeddings were found.")
        return [], {}

    # Get trending scores
    trending_scores = get_trending_popularity(tmdb.api_key)
    
    if debug:
        st.write(f"🔍 **Trending Scores Diagnostic:**")
        st.write(f"   - Total movies with trending scores: {len(trending_scores)}")
        
        # Show movies with non-zero trending scores
        non_zero_trending = {k: v for k, v in trending_scores.items() if v > 0}
        st.write(f"   - Movies with non-zero trending: {len(non_zero_trending)}")
        
        if non_zero_trending:
            st.write("   - Non-zero trending movies:")
            for movie_id, score in list(non_zero_trending.items())[:5]:
                st.write(f"     Movie ID {movie_id}: {score:.4f}")

    # STEP 1: First pass - collect all raw embedding similarities
    raw_embedding_scores = {}
    for movie_obj, embedding in candidate_movies.values():
        if movie_obj is None or embedding is None:
            continue
        
        movie_id = getattr(movie_obj, 'id', None)
        if movie_id and movie_id in candidate_movies:
            candidate_data = candidate_movies[movie_id]
            if len(candidate_data) >= 2 and candidate_data[1] is not None:
                candidate_embedding = candidate_data[1]
                
                # Calculate raw embedding similarity
                if cluster_centers and diversity_metrics['taste_profile'] in ['diverse', 'eclectic']:
                    raw_embedding_sim = compute_multi_cluster_similarity(candidate_embedding, cluster_centers)
                else:
                    if hasattr(user_prefs, 'favorite_embeddings') and user_prefs['favorite_embeddings']:
                        avg_embedding = torch.mean(torch.stack(user_prefs['favorite_embeddings']), dim=0)
                        raw_embedding_sim = float(cos_sim(candidate_embedding, avg_embedding))
                    else:
                        raw_embedding_sim = 0.0
                
                raw_embedding_scores[movie_id] = raw_embedding_sim

    # STEP 2: Normalize embedding scores to 0-1 range
    if raw_embedding_scores:
        all_embedding_scores = list(raw_embedding_scores.values())
        min_emb, max_emb = min(all_embedding_scores), max(all_embedding_scores)
        
        # Avoid division by zero
        if max_emb > min_emb:
            normalized_embedding_scores = {
                movie_id: (raw_score - min_emb) / (max_emb - min_emb) 
                for movie_id, raw_score in raw_embedding_scores.items()
            }
        else:
            normalized_embedding_scores = {movie_id: 0.5 for movie_id in raw_embedding_scores.keys()}
        
        if debug:
            st.write(f"🔧 **Embedding Normalization:**")
            st.write(f"   - Raw range: {min_emb:.4f} to {max_emb:.4f}")
            st.write(f"   - Normalized range: 0.0000 to 1.0000")
    else:
        normalized_embedding_scores = {}

    # STEP 3: Second pass - compute final scores with normalized embeddings
    scored = []
    score_breakdowns = {}

    for movie_obj, embedding in candidate_movies.values():
        if movie_obj is None or embedding is None:
            continue
        try:
            result = compute_score(
                movie_obj, cluster_centers, diversity_metrics, favorite_genres, 
                favorite_actors, user_prefs, trending_scores, favorite_narrative_styles, 
                candidate_movies, normalized_embedding_scores  # Pass normalized scores
            )

            # Handle both old and new return formats
            if isinstance(result, tuple):
                score, breakdown = result
            else:
                score = result
                breakdown = {}

            vote_count = getattr(movie_obj, 'vote_count', 0)
            vote_bonus = min(vote_count, 500) / 50000
            score += vote_bonus

            # Store breakdown for debug
            if debug and isinstance(result, tuple):
                breakdown['vote_bonus'] = vote_bonus
                movie_id = getattr(movie_obj, 'id', 0)
                if not hasattr(recommend_movies, 'score_breakdowns'):
                    recommend_movies.score_breakdowns = {}
                recommend_movies.score_breakdowns[movie_id] = breakdown

            scored.append((movie_obj, score))
            
            if debug:
                # Show mood scores for a broader sample
                movie_genres = getattr(movie_obj, 'genres', [])
                raw_mood_score = get_mood_score(movie_genres, user_prefs['preferred_moods'])
                movie_title = getattr(movie_obj, 'title', 'Unknown')
                
                # Debug first 20 movies instead of just top 10
                if len(scored) < 20:
                    st.write(f"🎭 {movie_title}: raw_mood_score = {raw_mood_score:.3f}")
            
        except Exception as e:
            st.warning(f"Error scoring movie {getattr(movie_obj, 'title', 'Unknown')}: {e}")
            continue

    scored.sort(key=lambda x:x[1], reverse=True)
    
    # Apply diversity and filtering
    top = []
    low_votes = 0
    used_genres = set()
    favorite_titles_set = {title.lower() for title in favorite_titles}

    def calculate_genre_overlap(movie_genres, used_genres):
        """Helper to calculate genre overlap penalty."""
        overlap_count = len(movie_genres & used_genres)
        return overlap_count / max(len(movie_genres), 1)

    for m, s in scored:
        vote_count = getattr(m, 'vote_count', 0)
        movie_title = getattr(m, 'title', 'Unknown Title')
        
        # Skip if this movie is in the user's favorites
        if movie_title.lower() in favorite_titles_set:
            continue
        
        # Get movie genres for diversity
        movie_genres = set()
        genres_list = getattr(m, 'genres', [])
        for g in genres_list:
            if isinstance(g, dict):
                name = g.get('name', '')
            else:
                name = getattr(g, 'name', '')
            if name:
                movie_genres.add(name)
        
        # Calculate final score with diversity penalty
        final_score = s
        
        # Apply diversity enforcement after first 5 selections
        if len(top) > 5:
            genre_overlap_penalty = calculate_genre_overlap(movie_genres, used_genres)
            final_score = s - (genre_overlap_penalty * 0.1)
            
            if debug:
                st.write(f"🎨 **Diversity Check for {movie_title}:**")
                st.write(f"   - Movie genres: {list(movie_genres)}")
                st.write(f"   - Used genres: {list(used_genres)}")
                st.write(f"   - Overlap penalty: {genre_overlap_penalty:.3f}")
                st.write(f"   - Original score: {s:.4f}")
                st.write(f"   - Final score: {final_score:.4f}")
        
        # For eclectic users, ensure genre diversity (keep existing logic)
        if diversity_metrics['taste_profile'] == 'eclectic' and len(top) >= 3:
            genre_overlap = len(movie_genres & used_genres) / max(len(movie_genres), 1)
            if genre_overlap > 0.7:
                continue
        
        if vote_count < 100:
            if low_votes >= 2: 
                continue
            low_votes += 1
        
        top.append((movie_title, final_score))
        used_genres.update(movie_genres)
        
        # Debug final recommendations as they're selected
        if debug:
            movie_id = getattr(m, 'id', 0)
            trending_score = trending_scores.get(movie_id, 0)
            st.write(f"🎯 **Recommendation #{len(top)}**: {movie_title}")
            st.write(f"   - Final score: {final_score:.4f}")
            st.write(f"   - Trending score: {trending_score:.4f}")
            st.write(f"   - Vote count: {vote_count}")
            
            # Show score breakdown
            if hasattr(recommend_movies, 'score_breakdowns') and movie_id in recommend_movies.score_breakdowns:
                breakdown = recommend_movies.score_breakdowns[movie_id]
                st.write("   - **Score breakdown:**")
                for component, value in breakdown.items():
                    st.write(f"     • {component}: {value:.4f}")
            
            # Show what year it is for recency bias check
            try:
                release_date = getattr(m, 'release_date', None)
                if release_date:
                    year = release_date[:4]
                    st.write(f"   - Year: {year}")
            except:
                pass
            st.write("---")
        
        if len(top) == 10: 
            break

    # Apply final franchise limiting
    franchise_limited_top = apply_final_franchise_limit(top, candidate_movies, max_per_franchise=1)

    # Cache and return result
    result = (franchise_limited_top, candidate_movies)
    st.session_state.recommendation_cache[cache_key] = result
    return result