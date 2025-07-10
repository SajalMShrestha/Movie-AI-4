"""
Main Streamlit application for movie recommendations.
"""

import streamlit as st
import uuid
from tmdbv3api import TMDb, Movie
import requests

# Import our modular components
from src.movie_search import enhanced_movie_search
from src.movie_scoring import recommend_movies
from src.feedback_system import (
    initialize_feedback_csv, get_or_create_numeric_session_id,
    record_feedback_to_sheet, record_final_comments_to_sheet
)

# Page configuration
st.set_page_config(
    page_title="Screen or Skip",
    page_icon="🎬"
)

# Initialize TMDb
tmdb = TMDb()
tmdb.api_key = st.secrets["TMDB_API_KEY"]
tmdb.language = 'en'
tmdb.debug = True

# Initialize session state variables
def initialize_session_state():
    """Initialize all required session state variables."""
    session_vars = {
        "favorite_movies": [],
        "selected_movie": None,
        "recommendations": None,
        "candidates": None,
        "recommend_triggered": False,
        "favorite_movie_posters": {},
        "movie_details_cache": {},
        "movie_credits_cache": {},
        "fetch_cache": {},
        "recommendation_cache": {},
        "session_id": str(uuid.uuid4()),
        "search_done": False,
        "previous_query": ""
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

# Initialize everything
initialize_session_state()
initialize_feedback_csv()
numeric_id, session_uuid = get_or_create_numeric_session_id()
st.session_state.numeric_session_id = numeric_id

def main():
    """Main application function."""
    st.title("🎬 Screen or Skip (S<>S)")
    
    # Movie search section
    search_query = st.text_input(
        "Search (Add your 5 favorite movies to get personalized recommendations!)", 
        key="movie_search"
    )

    # Reset search_done when user types a different movie
    if search_query != st.session_state["previous_query"]:
        st.session_state["search_done"] = False
        st.session_state["previous_query"] = search_query

    # Use the refactored search logic
    search_results = enhanced_movie_search(search_query, st.secrets["TMDB_API_KEY"])

    # Show Top 5 matches
    if search_results:
        st.markdown("### Top 5 Matches")
        cols = st.columns(5)
        for idx, movie in enumerate(search_results):
            with cols[idx]:
                poster_url = f"https://image.tmdb.org/t/p/w200{movie['poster_path']}" if movie.get("poster_path") else None
                if poster_url:
                    st.image(poster_url)
                st.write(f"**{movie['label']}**")
                if st.button("Add Movie", key=f"add_{idx}"):
                    clean_title = movie["label"].split(" (", 1)[0]
                    movie_id = movie["id"]

                    existing_titles = [m["title"] for m in st.session_state.favorite_movies if isinstance(m, dict)]
                    if len(st.session_state.favorite_movies) >= 5:
                        st.warning("You can only add up to 5 movies.")
                    elif clean_title not in existing_titles:
                        st.session_state.favorite_movies.append({
                            "title": clean_title,
                            "year": movie["label"].split("(", 1)[1].replace(")", "") if "(" in movie["label"] else "",
                            "poster_path": movie.get("poster_path", ""),
                            "id": movie_id
                        })
                        st.session_state["search_done"] = True
                        st.session_state["previous_query"] = ""
                        st.success(f"✅ Added {clean_title}")
                        st.rerun()

    # Show selected movies section
    if st.session_state.favorite_movies:
        st.subheader("🎥 Your Selected Movies (5 max)")
        
        cols = st.columns(5)
        for i, movie in enumerate(st.session_state.favorite_movies):
            with cols[i % 5]:
                title = movie["title"]
                year = movie.get("year", "")
                poster = movie.get("poster_path")
                
                if poster:
                    poster_url = f"https://image.tmdb.org/t/p/w200{poster}"
                    st.image(poster_url)
                else:
                    st.write("🎬 No poster")
                
                st.write(f"**{title}**")
                if year:
                    st.write(f"({year})")
                
                if st.button(f"Remove", key=f"remove_{i}"):
                    st.session_state.favorite_movies.pop(i)
                    st.rerun()

        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("❌ Clear All"):
                st.session_state.favorite_movies = []
                st.session_state.recommendations = None
                st.session_state.candidates = None
                st.session_state.recommend_triggered = False
                st.rerun()

        with col2:
            if st.button("🎬 Get Recommendations", type="primary"):
                if len(st.session_state.favorite_movies) != 5:
                    st.warning("Please select exactly 5 movies to get recommendations.")
                else:
                    with st.spinner("Finding personalized movie recommendations..."):
                        favorite_titles = [m["title"] for m in st.session_state.favorite_movies if isinstance(m, dict)]
                        try:
                            recs, candidate_movies = recommend_movies(favorite_titles)
                            st.session_state.recommendations = recs
                            st.session_state.candidates = candidate_movies
                            st.session_state.recommend_triggered = True
                        except Exception as e:
                            st.error(f"❌ Failed to generate recommendations: {e}")
                            import traceback
                            st.error(traceback.format_exc())

    # Display recommendations and feedback
    if st.session_state.recommend_triggered:
        if not st.session_state.recommendations:
            st.warning("⚠️ No recommendations could be generated. Please try different favorite movies.")
            st.info("Tip: Make sure your selected movies have plot summaries and at least some popularity.")
        else:
            st.subheader("🌟 Your Top 10 Movie Recommendations")

            # Collect feedback for all movies
            user_feedback = {}

            for idx, (title, score) in enumerate(st.session_state.recommendations, 1):
                # Find the movie object from candidates
                movie_obj = None
                for m, _ in st.session_state.candidates.values():
                    if m and getattr(m, 'title', '') == title:
                        movie_obj = m
                        break
                
                if movie_obj is None:
                    continue
                
                st.markdown(f"### {idx}. {movie_obj.title}")
                
                # Create columns for poster and details
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if movie_obj.poster_path:
                        st.image(f"https://image.tmdb.org/t/p/w300{movie_obj.poster_path}", width=150)
                    else:
                        st.write("🎬 No poster")
                
                with col2:
                    # Show year
                    release_year = "N/A"
                    try:
                        if hasattr(movie_obj, 'release_date') and movie_obj.release_date:
                            release_year = movie_obj.release_date[:4]
                    except:
                        pass
                    st.write(f"**Year:** {release_year}")
                    
                    # Show plot
                    overview = getattr(movie_obj, 'overview', None) or getattr(movie_obj, 'plot', None) or "No description available."
                    st.write(f"**Plot:** {overview}")

                # Feedback section
                fb_key = f"watch_{idx}"
                liked_key = f"liked_{idx}"

                response = st.radio(
                    "Would you watch this?", 
                    ["Yes", "No", "Already watched"], 
                    key=fb_key, 
                    index=None,
                    horizontal=True
                )

                liked = None
                if response == "Already watched":
                    liked = st.radio(
                        "Did you like it?", 
                        ["Yes", "No"], 
                        key=liked_key, 
                        index=None,
                        horizontal=True
                    )

                # Capture movie metadata
                movie_genres = []
                genres_list = getattr(movie_obj, 'genres', [])
                for g in genres_list:
                    if isinstance(g, dict):
                        name = g.get('name', '')
                    else:
                        name = getattr(g, 'name', '')
                    if name:
                        movie_genres.append(name)
                
                user_feedback[idx] = {
                    "movie": movie_obj.title,
                    "movie_id": movie_obj.id,
                    "movie_genres": " | ".join(movie_genres),
                    "movie_year": release_year,
                    "recommendation_rank": idx,
                    "recommendation_score": score,
                    "response": response,
                    "liked": liked,
                }
                
                st.markdown("---")

            # Final Comments Section
            st.markdown("---")
            st.subheader("💬 Final Comments")
            st.write("Share any additional thoughts about the recommendations!")

            final_comments = st.text_area(
                "Your feedback helps me improve the recommendation system:",
                placeholder="Did the recommendations match your taste? Any movies you were surprised to see?",
                height=100,
                key="final_comments_text"
            )

            if final_comments:
                char_count = len(final_comments)
                st.caption(f"Characters: {char_count}")

            # Email input
            st.markdown("---")
            st.subheader("📧 Email")
            save_email = st.text_input(
                "Enter your email to save your recommendations:",
                placeholder="your.email@example.com",
                key="save_profile_email"
            )

            # Submit all responses
            if st.button("Submit All Responses", type="primary"):
                success_count = 0
                total_responses = 0
                
                # Get user profile data
                user_top_5 = " | ".join([m["title"] for m in st.session_state.favorite_movies])
                
                # Extract genres from user's selected movies
                favorite_genres = set()
                for movie in st.session_state.favorite_movies:
                    try:
                        movie_id = movie.get("id")
                        if movie_id and movie_id in st.session_state.movie_details_cache:
                            details = st.session_state.movie_details_cache[movie_id]
                            genres_list = getattr(details, 'genres', [])
                            for g in genres_list:
                                if isinstance(g, dict):
                                    name = g.get('name', '')
                                else:
                                    name = getattr(g, 'name', '')
                                if name:
                                    favorite_genres.add(name)
                    except Exception as e:
                        st.warning(f"Error processing movie: {e}")
                        continue
                
                user_favorite_genres = " | ".join(list(favorite_genres)[:5])
                user_taste_profile = "diverse"  # Default
                
                # Process movie feedback responses
                for index, feedback in user_feedback.items():
                    if feedback["response"]:  # Only save if user provided a response
                        total_responses += 1
                        
                        recommendation_reason = f"Genre match: {feedback['movie_genres']}" if feedback['movie_genres'] else "Algorithm recommendation"
                        
                        if record_feedback_to_sheet(
                            numeric_session_id=st.session_state.numeric_session_id,
                            uuid_session_id=st.session_state.session_id,
                            user_top_5_movies=user_top_5,
                            user_taste_profile=user_taste_profile,
                            user_favorite_genres=user_favorite_genres,
                            recommendation_rank=feedback["recommendation_rank"],
                            movie_id=feedback["movie_id"],
                            movie_title=feedback["movie"],
                            movie_genres=feedback["movie_genres"],
                            movie_year=feedback["movie_year"],
                            recommendation_score=feedback["recommendation_score"],
                            recommendation_reason=recommendation_reason,
                            would_watch=feedback["response"],
                            liked_if_seen=feedback["liked"] or "",
                            user_email=save_email.strip() if save_email else ""
                        ):
                            success_count += 1
                
                # Save final comments if provided
                comments_saved = False
                if final_comments and final_comments.strip():
                    comments_saved = record_final_comments_to_sheet(
                        numeric_session_id=st.session_state.numeric_session_id,
                        uuid_session_id=st.session_state.session_id,
                        user_top_5_movies=user_top_5,
                        user_taste_profile=user_taste_profile,
                        user_favorite_genres=user_favorite_genres,
                        final_comments=final_comments.strip(),
                        user_email=save_email.strip() if save_email else ""
                    )
                
                # Show results
                if success_count == total_responses and total_responses > 0:
                    if comments_saved or not final_comments.strip():
                        st.success(f"✅ All {success_count} movie responses saved successfully!")
                        if comments_saved:
                            st.success("✅ Your final comments were also saved!")
                        st.balloons()
                    else:
                        st.success(f"✅ All {success_count} movie responses saved!")
                        st.warning("⚠️ Final comments failed to save, but movie feedback was recorded.")
                elif success_count > 0:
                    st.warning(f"⚠️ {success_count}/{total_responses} movie responses saved. Some failed to save.")
                    if comments_saved:
                        st.success("✅ Your final comments were saved!")
                else:
                    if total_responses == 0:
                        st.warning("⚠️ Please provide at least one movie response before submitting.")
                    else:
                        st.error("❌ Failed to save any movie responses. Please check your Google Sheets setup.")
                        if comments_saved:
                            st.success("✅ Your final comments were saved!")

if __name__ == "__main__":
    main()