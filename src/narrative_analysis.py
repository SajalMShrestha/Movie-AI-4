"""
Narrative analysis and text processing for movie plots.
"""

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import word_tokenize
from nltk.corpus import stopwords
import textstat
from collections import Counter

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

# Safe auto-download of punkt tokenizer if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Keywords for genre classification
ACTION_KEYWORDS = {"chase", "fight", "escape", "war", "battle", "mission", "rescue"}
EMOTION_KEYWORDS = {"love", "friendship", "betrayal", "grief", "romance", "relationship", "emotional"}
CONCEPT_KEYWORDS = {"reality", "consciousness", "philosophy", "dream", "mystery", "existential"}
REALISTIC_KEYWORDS = {"city", "suburb", "historical", "town", "village", "real-life", "everyday", "ordinary"}
FANTASY_KEYWORDS = {"galaxy", "kingdom", "future", "space", "magical", "supernatural", "alien", "fantasy", "alternate"}

def preprocess_text(plot):
    """
    Tokenize and preprocess text for analysis.
    
    Args:
        plot: Movie plot text
    
    Returns:
        List of preprocessed words
    """
    tokens = word_tokenize(plot.lower())
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    return words

def infer_tone(plot):
    """
    Infer the emotional tone of a movie plot.
    
    Args:
        plot: Movie plot text
    
    Returns:
        String: 'positive', 'negative', or 'neutral'
    """
    sentiment = sia.polarity_scores(plot)
    if sentiment['compound'] >= 0.3:
        return "positive"
    elif sentiment['compound'] <= -0.3:
        return "negative"
    else:
        return "neutral"

def infer_narrative_complexity(plot):
    """
    Determine narrative complexity based on readability and vocabulary.
    
    Args:
        plot: Movie plot text
    
    Returns:
        String: 'complex' or 'simple'
    """
    readability = textstat.flesch_kincaid_grade(plot)
    words = preprocess_text(plot)
    unique_words_ratio = len(set(words)) / max(len(words), 1)

    # Adjusted complexity thresholds
    if readability >= 9 or unique_words_ratio >= 0.5:
        return "complex"
    else:
        return "simple"

def infer_genre_indicator(plot):
    """
    Classify plot focus based on keywords.
    
    Args:
        plot: Movie plot text
    
    Returns:
        String: Genre indicator category
    """
    words = set(preprocess_text(plot))
    if words & ACTION_KEYWORDS:
        return "action-oriented"
    elif words & EMOTION_KEYWORDS:
        return "emotion/character-oriented"
    elif words & CONCEPT_KEYWORDS:
        return "idea/concept-oriented"
    else:
        return "general"

def infer_setting_context(plot):
    """
    Determine setting context of the plot.
    
    Args:
        plot: Movie plot text
    
    Returns:
        String: Setting context category
    """
    words = set(preprocess_text(plot))
    if words & FANTASY_KEYWORDS:
        return "fantastical/surreal"
    elif words & REALISTIC_KEYWORDS:
        return "realistic"
    else:
        return "neutral"

def infer_narrative_style(plot):
    """
    Comprehensive narrative style analysis.
    
    Args:
        plot: Movie plot text
    
    Returns:
        Dictionary with narrative style components
    """
    if not plot:
        return {
            "tone": "neutral",
            "complexity": "simple",
            "genre_indicator": "general",
            "setting_context": "neutral"
        }

    return {
        "tone": infer_tone(plot),
        "complexity": infer_narrative_complexity(plot),
        "genre_indicator": infer_genre_indicator(plot),
        "setting_context": infer_setting_context(plot)
    }

def compute_narrative_similarity(candidate_style, reference_styles):
    """
    Compare narrative styles between candidate and reference movies.
    
    Args:
        candidate_style: Narrative style dict for candidate movie
        reference_styles: Reference styles from user's favorite movies
    
    Returns:
        Float: Similarity score between 0 and 1
    """
    similarity = 0
    for key in candidate_style:
        if not reference_styles[key]: 
            continue
        dominant = Counter(reference_styles[key]).most_common(1)[0][0]
        if candidate_style[key] == dominant:
            similarity += 1
    return similarity / len(candidate_style)

def construct_enriched_description(movie_details, credits, keywords=None):
    """
    Create enriched text description for embedding generation.
    
    Args:
        movie_details: Movie details object
        credits: Movie credits object
        keywords: Optional movie keywords
    
    Returns:
        String: Enriched description text
    """
    title = getattr(movie_details, 'title', 'Unknown')
    genres = []
    genres_list = getattr(movie_details, 'genres', [])
    for g in genres_list:
        if isinstance(g, dict):
            name = g.get('name', '')
        else:
            name = getattr(g, 'name', '')
        if name:
            genres.append(name)
    
    # Handle credits safely
    if isinstance(credits, dict):
        cast_list_raw = credits.get('cast', [])
        crew_list = credits.get('crew', [])
    else:
        cast_list_raw = getattr(credits, 'cast', [])
        crew_list = getattr(credits, 'crew', [])

    # Safely slice cast_list
    if hasattr(cast_list_raw, '__iter__'):
        cast_list = list(cast_list_raw)[:3] if cast_list_raw else []
    else:
        cast_list = []
    
    favorite_actors = set()
    favorite_directors = set()

    # Collect actor and director names
    for c in cast_list:
        if isinstance(c, dict):
            name = c.get('name', '')
        else:
            name = getattr(c, 'name', '')
        if name:
            favorite_actors.add(name)

    # Process director names
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
            favorite_directors.add(name)

    tagline = getattr(movie_details, 'tagline', '')
    overview = getattr(movie_details, 'overview', '')
    keyword_list = []
    if keywords:
        for k in keywords:
            if isinstance(k, dict):
                name = k.get('name', '')
            else:
                name = getattr(k, 'name', '')
            if name:
                keyword_list.append(name)

    enriched_text = f"{title} is a {', '.join([g for g in genres if g])} movie"
    if favorite_directors:
        enriched_text += f" directed by {', '.join([d for d in favorite_directors if d])}"
    if favorite_actors:
        enriched_text += f", starring {', '.join([c for c in favorite_actors if c])}"
    enriched_text += ". "
    if tagline:
        enriched_text += f"Tagline: {tagline}. "
    if keyword_list:
        enriched_text += f"Keywords: {', '.join([k for k in keyword_list if k])}. "
    enriched_text += f"Plot: {overview}"

    return enriched_text

def infer_mood_from_plot(plot):
    """
    Infer mood category from plot text.
    
    Args:
        plot: Movie plot text
    
    Returns:
        String: Mood category
    """
    sentiment = sia.polarity_scores(plot)
    if sentiment['compound'] >= 0.4:
        return 'feel_good'
    elif sentiment['compound'] <= -0.3:
        return 'melancholic'
    else:
        return 'cerebral'