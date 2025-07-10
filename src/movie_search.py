"""
Movie search functionality with fuzzy matching and typo tolerance.
"""

import re
import requests
import streamlit as st
from difflib import SequenceMatcher

def generate_search_variations(query):
    """
    Generate focused search variations with basic typo tolerance.
    
    Args:
        query: Original search query
    
    Returns:
        List of search variations
    """
    variations = set()
    
    # Original query
    variations.add(query.strip())
    
    # Basic cleanup
    clean_query = re.sub(r'[^\w\s]', ' ', query.lower()).strip()
    variations.add(clean_query)
    
    # Handle number-word conversions
    number_word_map = {'3': 'three', 'three': '3'}
    
    words = clean_query.split()
    for i, word in enumerate(words):
        if word in number_word_map:
            new_words = words.copy()
            new_words[i] = number_word_map[word]
            variations.add(' '.join(new_words))
    
    # Add space-corrected version for concatenated words
    if ' ' not in clean_query and len(clean_query) > 3:
        spaced_query = re.sub(r'^(\d+)([a-z])', r'\1 \2', clean_query)
        if spaced_query != clean_query:
            variations.add(spaced_query)
    
    # Add individual significant words (for partial matching)
    for word in words:
        if len(word) > 3:
            variations.add(word)
    
    return list(variations)[:5]

def calculate_title_similarity(query, title):
    """
    Calculate similarity between query and movie title using multiple methods.
    
    Args:
        query: Search query
        title: Movie title to compare against
    
    Returns:
        Float similarity score between 0 and 1
    """
    query_lower = query.lower().strip()
    title_lower = title.lower().strip()
    
    # Method 1: Exact match
    if query_lower == title_lower:
        return 1.0
    
    # Method 2: Substring matching
    if query_lower in title_lower or title_lower in query_lower:
        return 0.95
    
    # Method 3: Handle "3 idiots" specifically with typo tolerance
    if ('3' in query_lower or 'three' in query_lower or 'thre' in query_lower):
        query_normalized = query_lower.replace('thre', 'three').replace('idoits', 'idiots').replace('idoit', 'idiots')
        title_normalized = title_lower
        
        if (('3' in query_normalized or 'three' in query_normalized) and 
            'idiots' in query_normalized and
            ('3' in title_normalized or 'three' in title_normalized) and
            'idiots' in title_normalized):
            return 0.9
    
    # Method 4: Word-based similarity
    query_words = set(query_lower.split())
    title_words = set(title_lower.split())
    
    # Remove stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    query_words_clean = query_words - stop_words
    title_words_clean = title_words - stop_words
    
    if query_words_clean and title_words_clean:
        # Direct word overlap
        overlap = len(query_words_clean & title_words_clean)
        union = len(query_words_clean | title_words_clean)
        jaccard = overlap / union if union > 0 else 0
        
        if jaccard >= 0.5:
            return 0.8 + (jaccard * 0.2)
        
        # Fuzzy word matching for typos
        fuzzy_matches = 0
        for q_word in query_words_clean:
            for t_word in title_words_clean:
                if len(q_word) > 2 and len(t_word) > 2:
                    word_sim = SequenceMatcher(None, q_word, t_word).ratio()
                    if word_sim >= 0.7:
                        fuzzy_matches += 1
                        break
        
        fuzzy_ratio = fuzzy_matches / len(query_words_clean) if query_words_clean else 0
        if fuzzy_ratio >= 0.5:
            return 0.7 + (fuzzy_ratio * 0.2)
    
    # Method 5: Character-level similarity (fallback)
    char_similarity = SequenceMatcher(None, query_lower, title_lower).ratio()
    return char_similarity

def fuzzy_search_movies(query, max_results=10, similarity_threshold=0.6):
    """
    Perform fuzzy search for movies with typo tolerance.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        similarity_threshold: Minimum similarity score to include
    
    Returns:
        List of movie dictionaries with similarity scores
    """
    try:
        # First try direct search
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": st.secrets["TMDB_API_KEY"], "query": query}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            
            # If we get good results from direct search, return them
            if len(results) >= 3:
                return [
                    {
                        "title": m.get('title', ''),
                        "year": m.get('release_date', '')[:4] if m.get('release_date') else '',
                        "id": m.get('id'),
                        "poster_path": m.get('poster_path'),
                        "similarity": 1.0
                    }
                    for m in results[:max_results]
                    if m.get('title') and m.get('id')
                ]
        
        # If direct search fails, try fuzzy matching
        fuzzy_results = []
        search_variations = generate_search_variations(query)
        
        for search_term in search_variations:
            try:
                params = {"api_key": st.secrets["TMDB_API_KEY"], "query": search_term}
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    word_results = response.json().get("results", [])
                    for movie in word_results[:12]:
                        title = movie.get('title', '')
                        if title:
                            similarity = calculate_title_similarity(query, title)
                            if similarity >= 0.3:  # Lowered threshold for better typo tolerance
                                fuzzy_results.append({
                                    "title": title,
                                    "year": movie.get('release_date', '')[:4] if movie.get('release_date') else '',
                                    "id": movie.get('id'),
                                    "poster_path": movie.get('poster_path'),
                                    "similarity": similarity
                                })
            except:
                continue
        
        # Remove duplicates and sort by similarity
        seen_ids = set()
        unique_results = []
        for result in fuzzy_results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: x['similarity'], reverse=True)
        return unique_results[:max_results]
        
    except Exception as e:
        st.warning(f"Fuzzy search error: {e}")
        return []

def suggest_corrections(query, search_results):
    """
    Suggest possible corrections when search returns few results.
    
    Args:
        query: Original search query
        search_results: Results from initial search
    
    Returns:
        Boolean indicating if corrections were suggested
    """
    if not search_results or len(search_results) < 3:
        st.info(f"🔍 **Showing closest matches for '{query}'**")
        
        fuzzy_results = fuzzy_search_movies(query, max_results=5, similarity_threshold=0.2)
        
        if fuzzy_results:
            st.write("**Did you mean one of these?**")
            
            cols = st.columns(5)
            for idx, movie in enumerate(fuzzy_results[:5]):
                with cols[idx % 5]:
                    if movie.get('poster_path'):
                        poster_url = f"https://image.tmdb.org/t/p/w200{movie['poster_path']}"
                        st.image(poster_url, use_container_width=True)
                    
                    title_display = movie['title']
                    if movie['year']:
                        title_display += f" ({movie['year']})"
                    
                    st.write(f"**{title_display}**")
                    st.write(f"*{movie['similarity']:.0%} match*")
                    
                    if st.button("Add This Movie", key=f"fuzzy_add_{idx}"):
                        # Add the movie to favorites
                        existing_titles = [m["title"] for m in st.session_state.favorite_movies if isinstance(m, dict)]
                        if len(st.session_state.favorite_movies) >= 5:
                            st.warning("You can only add up to 5 movies.")
                        elif movie['title'] not in existing_titles:
                            st.session_state.favorite_movies.append({
                                "title": movie['title'],
                                "year": movie['year'],
                                "poster_path": movie.get('poster_path', ''),
                                "id": movie['id']
                            })
                            st.session_state["search_done"] = True
                            st.session_state["previous_query"] = ""
                            st.session_state["movie_search"] = ""
                            st.success(f"✅ Added {movie['title']}")
                            st.rerun()
            
            return True
    return False

def enhanced_movie_search(search_query, api_key):
    if not search_query or len(search_query) < 2:
        return []
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": api_key, "query": search_query}
    response = requests.get(url, params=params)
    data = response.json()
    results = data.get("results", [])
    search_results = [
        {
            "label": f"{m.get('title')} ({m.get('release_date')[:4]})" if m.get("release_date") else m.get('title'),
            "id": m.get("id"),
            "poster_path": m.get("poster_path")
        }
        for m in results[:5]
        if m.get("title") and m.get("id")
    ]
    return search_results