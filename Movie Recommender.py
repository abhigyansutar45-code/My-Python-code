"""
MOVIE RECOMMENDATION SYSTEM
===========================
Beginner-friendly content-based recommendation system.

Core method:
    Movie metadata -> cleaned tags -> TF-IDF -> cosine similarity -> Top-N

The script can:
    1. Load data/movies.csv when available.
    2. Otherwise use a small clearly-labelled educational demo dataset.
    3. Clean and combine movie metadata.
    4. Build a TF-IDF representation.
    5. Generate recommendations dynamically with cosine similarity.
    6. Search for movies.
    7. Filter recommendations by genre/year/rating.
    8. Explain recommendation similarity using shared metadata tokens.
    9. Create meaningful visualizations when requested.
   10. Save/load the recommender with joblib.

Expected CSV columns (flexible):
    movie_id, title, genres, overview, keywords, cast, director,
    production, release_year, rating

All recommendation results are calculated from the supplied metadata.
No movie-to-movie recommendation list is hardcoded.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", category=FutureWarning)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "movies.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "movie_recommender.joblib"


# -----------------------------------------------------------------------------
# EDUCATIONAL DEMO DATA
# -----------------------------------------------------------------------------
# This is deliberately small and is only a fallback when no CSV is supplied.
# It is labelled as demonstration data and is not presented as a real dataset.
DEMO_MOVIES = [
    {
        "movie_id": 1,
        "title": "Interstellar",
        "genres": "Science Fiction Adventure Drama",
        "overview": "Explorers travel through a wormhole in space to search for a future home for humanity.",
        "keywords": "space exploration wormhole astronaut future survival science",
        "cast": "Matthew McConaughey Anne Hathaway Jessica Chastain",
        "director": "Christopher Nolan",
        "production": "Warner Bros",
        "release_year": 2014,
        "rating": 8.7,
    },
    {
        "movie_id": 2,
        "title": "The Martian",
        "genres": "Science Fiction Adventure Drama",
        "overview": "An astronaut stranded on Mars uses science and engineering to survive while a rescue mission is planned.",
        "keywords": "Mars astronaut survival space science engineering NASA",
        "cast": "Matt Damon Jessica Chastain",
        "director": "Ridley Scott",
        "production": "20th Century Fox",
        "release_year": 2015,
        "rating": 8.0,
    },
    {
        "movie_id": 3,
        "title": "Gravity",
        "genres": "Science Fiction Thriller Drama",
        "overview": "Astronauts struggle to survive after a disaster leaves them isolated in orbit above Earth.",
        "keywords": "space astronaut orbit survival disaster gravity",
        "cast": "Sandra Bullock George Clooney",
        "director": "Alfonso Cuaron",
        "production": "Warner Bros",
        "release_year": 2013,
        "rating": 7.7,
    },
    {
        "movie_id": 4,
        "title": "Arrival",
        "genres": "Science Fiction Drama Mystery",
        "overview": "A linguist works with scientists to communicate with mysterious visitors who arrive on Earth.",
        "keywords": "aliens language science communication mystery future",
        "cast": "Amy Adams Jeremy Renner",
        "director": "Denis Villeneuve",
        "production": "Paramount Pictures",
        "release_year": 2016,
        "rating": 7.9,
    },
    {
        "movie_id": 5,
        "title": "Inception",
        "genres": "Science Fiction Action Thriller",
        "overview": "A skilled team enters dreams to manipulate information and complete a dangerous mission.",
        "keywords": "dreams mind heist technology subconscious mission",
        "cast": "Leonardo DiCaprio Joseph Gordon-Levitt",
        "director": "Christopher Nolan",
        "production": "Warner Bros",
        "release_year": 2010,
        "rating": 8.8,
    },
    {
        "movie_id": 6,
        "title": "Blade Runner 2049",
        "genres": "Science Fiction Drama Mystery",
        "overview": "A future detective uncovers a secret that could change the relationship between humans and artificial beings.",
        "keywords": "future detective artificial intelligence robots mystery dystopia",
        "cast": "Ryan Gosling Harrison Ford",
        "director": "Denis Villeneuve",
        "production": "Warner Bros",
        "release_year": 2017,
        "rating": 8.0,
    },
    {
        "movie_id": 7,
        "title": "Avatar",
        "genres": "Science Fiction Adventure Action",
        "overview": "A marine explores an alien world and becomes involved in a conflict over its resources.",
        "keywords": "alien planet exploration nature technology adventure",
        "cast": "Sam Worthington Zoe Saldana",
        "director": "James Cameron",
        "production": "20th Century Fox",
        "release_year": 2009,
        "rating": 7.9,
    },
    {
        "movie_id": 8,
        "title": "The Lord of the Rings: The Fellowship of the Ring",
        "genres": "Fantasy Adventure Drama",
        "overview": "A young hero begins a dangerous journey to destroy a powerful ring while companions protect him.",
        "keywords": "fantasy ring quest journey magic fellowship war",
        "cast": "Elijah Wood Ian McKellen Viggo Mortensen",
        "director": "Peter Jackson",
        "production": "New Line Cinema",
        "release_year": 2001,
        "rating": 8.8,
    },
    {
        "movie_id": 9,
        "title": "The Dark Knight",
        "genres": "Action Crime Drama Thriller",
        "overview": "A masked hero faces a criminal mastermind who creates chaos across a major city.",
        "keywords": "hero crime city villain justice chaos vigilante",
        "cast": "Christian Bale Heath Ledger",
        "director": "Christopher Nolan",
        "production": "Warner Bros",
        "release_year": 2008,
        "rating": 9.0,
    },
    {
        "movie_id": 10,
        "title": "Mad Max: Fury Road",
        "genres": "Action Adventure Science Fiction",
        "overview": "A driver and a group of rebels race across a harsh wasteland while escaping a powerful tyrant.",
        "keywords": "desert survival chase rebellion vehicles future action",
        "cast": "Tom Hardy Charlize Theron",
        "director": "George Miller",
        "production": "Warner Bros",
        "release_year": 2015,
        "rating": 8.1,
    },
    {
        "movie_id": 11,
        "title": "The Prestige",
        "genres": "Drama Mystery Science Fiction Thriller",
        "overview": "Two rival magicians compete to create increasingly extraordinary illusions with dangerous consequences.",
        "keywords": "magic rivalry illusion invention mystery obsession",
        "cast": "Christian Bale Hugh Jackman",
        "director": "Christopher Nolan",
        "production": "Touchstone Pictures",
        "release_year": 2006,
        "rating": 8.5,
    },
    {
        "movie_id": 12,
        "title": "Jurassic Park",
        "genres": "Science Fiction Adventure Thriller",
        "overview": "Scientists visit a theme park where cloned dinosaurs escape and threaten the visitors.",
        "keywords": "dinosaurs science island park genetics survival adventure",
        "cast": "Sam Neill Laura Dern Jeff Goldblum",
        "director": "Steven Spielberg",
        "production": "Universal Pictures",
        "release_year": 1993,
        "rating": 8.2,
    },
    {
        "movie_id": 13,
        "title": "The Matrix",
        "genres": "Science Fiction Action Thriller",
        "overview": "A hacker discovers that reality is a simulated system and joins a rebellion against machines.",
        "keywords": "simulation hacker artificial intelligence machines reality rebellion",
        "cast": "Keanu Reeves Laurence Fishburne",
        "director": "The Wachowskis",
        "production": "Warner Bros",
        "release_year": 1999,
        "rating": 8.7,
    },
    {
        "movie_id": 14,
        "title": "The Shawshank Redemption",
        "genres": "Drama",
        "overview": "A prisoner builds friendship and hope while serving a long sentence in a harsh institution.",
        "keywords": "friendship hope prison freedom redemption",
        "cast": "Tim Robbins Morgan Freeman",
        "director": "Frank Darabont",
        "production": "Columbia Pictures",
        "release_year": 1994,
        "rating": 9.3,
    },
    {
        "movie_id": 15,
        "title": "Spider-Man: Into the Spider-Verse",
        "genres": "Animation Action Adventure Science Fiction",
        "overview": "A teenager becomes a hero and meets alternate versions of Spider-Man from different realities.",
        "keywords": "superhero multiverse animation teenager alternate reality hero",
        "cast": "Shameik Moore Jake Johnson Hailee Steinfeld",
        "director": "Bob Persichetti Peter Ramsey Rodney Rothman",
        "production": "Sony Pictures",
        "release_year": 2018,
        "rating": 8.4,
    },
    {
        "movie_id": 16,
        "title": "The Social Network",
        "genres": "Drama Biography",
        "overview": "A drama about the creation of a major social network and the relationships surrounding its founders.",
        "keywords": "technology startup programming friendship entrepreneurship social media",
        "cast": "Jesse Eisenberg Andrew Garfield",
        "director": "David Fincher",
        "production": "Columbia Pictures",
        "release_year": 2010,
        "rating": 7.8,
    },
    {
        "movie_id": 17,
        "title": "Hidden Figures",
        "genres": "Drama History Biography",
        "overview": "Mathematicians at a space agency contribute important work during the early era of human spaceflight.",
        "keywords": "NASA mathematics space science history engineering",
        "cast": "Taraji P Henson Octavia Spencer Janelle Monae",
        "director": "Theodore Melfi",
        "production": "20th Century Fox",
        "release_year": 2016,
        "rating": 7.8,
    },
    {
        "movie_id": 18,
        "title": "Oppenheimer",
        "genres": "Drama History Biography",
        "overview": "A scientist leads a historic research project while confronting the consequences of a powerful invention.",
        "keywords": "science scientist physics history invention research war",
        "cast": "Cillian Murphy Emily Blunt Robert Downey Jr",
        "director": "Christopher Nolan",
        "production": "Universal Pictures",
        "release_year": 2023,
        "rating": 8.6,
    },
    {
        "movie_id": 19,
        "title": "Top Gun: Maverick",
        "genres": "Action Drama",
        "overview": "An experienced pilot trains a new generation for a difficult aviation mission while confronting his past.",
        "keywords": "pilot aviation training military flight mission friendship",
        "cast": "Tom Cruise Miles Teller",
        "director": "Joseph Kosinski",
        "production": "Paramount Pictures",
        "release_year": 2022,
        "rating": 8.2,
    },
    {
        "movie_id": 20,
        "title": "WALL-E",
        "genres": "Animation Science Fiction Adventure",
        "overview": "A small robot cleaning an abandoned Earth discovers a new purpose and joins a journey into space.",
        "keywords": "robot space Earth environment future exploration animation",
        "cast": "Ben Burtt Elissa Knight",
        "director": "Andrew Stanton",
        "production": "Pixar",
        "release_year": 2008,
        "rating": 8.4,
    },
]


REQUIRED_COLUMNS = {
    "movie_id": "movie_id",
    "title": "title",
    "genres": "genres",
    "overview": "overview",
    "keywords": "keywords",
    "cast": "cast",
    "director": "director",
    "production": "production",
    "release_year": "release_year",
    "rating": "rating",
}

TEXT_COLUMNS = [
    "genres",
    "overview",
    "keywords",
    "cast",
    "director",
    "production",
]


# -----------------------------------------------------------------------------
# DATA STRUCTURES
# -----------------------------------------------------------------------------
@dataclass
class Recommendation:
    title: str
    genres: str
    release_year: object
    rating: object
    similarity: float
    explanation: str


# -----------------------------------------------------------------------------
# DATA LOADING AND CLEANING
# -----------------------------------------------------------------------------
def demo_dataframe() -> pd.DataFrame:
    """Return the clearly-labelled educational demonstration dataset."""
    return pd.DataFrame(DEMO_MOVIES)


def load_dataset(path: Optional[str] = None) -> tuple[pd.DataFrame, bool]:
    """Load a CSV if supplied/found, otherwise use educational demo data."""
    candidate = Path(path) if path else DEFAULT_DATA_PATH

    if candidate.exists():
        print(f"Loading dataset: {candidate}")
        df = pd.read_csv(candidate)
        return df, False

    print("No data/movies.csv was found.")
    print("Using the built-in educational demonstration dataset instead.")
    print("This demo dataset is not presented as a complete real-world movie database.\n")
    return demo_dataframe(), True


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common alternative column names to the expected project names."""
    result = df.copy()
    normalized = {
        re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_"): column
        for column in result.columns
    }

    aliases = {
        "id": "movie_id",
        "movieid": "movie_id",
        "movie_title": "title",
        "name": "title",
        "genre": "genres",
        "description": "overview",
        "summary": "overview",
        "plot": "overview",
        "tagline": "overview",
        "actors": "cast",
        "stars": "cast",
        "producer": "production",
        "production_companies": "production",
        "year": "release_year",
        "release_date_year": "release_year",
        "vote_average": "rating",
        "score": "rating",
    }

    rename_map: dict[str, str] = {}
    for normalized_name, original_name in normalized.items():
        if normalized_name in REQUIRED_COLUMNS:
            rename_map[original_name] = REQUIRED_COLUMNS[normalized_name]
        elif normalized_name in aliases:
            rename_map[original_name] = aliases[normalized_name]

    result = result.rename(columns=rename_map)

    if "title" not in result.columns:
        raise ValueError(
            "The dataset must contain a movie title column such as 'title', 'movie_title', or 'name'."
        )

    for column in TEXT_COLUMNS:
        if column not in result.columns:
            result[column] = ""

    if "movie_id" not in result.columns:
        result["movie_id"] = np.arange(1, len(result) + 1)

    if "release_year" not in result.columns:
        result["release_year"] = np.nan

    if "rating" not in result.columns:
        result["rating"] = np.nan

    return result


def clean_text(value: object) -> str:
    """Normalize text without pretending to understand its semantic meaning."""
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, invalid titles, and numeric fields."""
    result = standardize_columns(df)

    result["title"] = result["title"].fillna("").astype(str).str.strip()
    result = result[result["title"] != ""].copy()

    result = result.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

    for column in TEXT_COLUMNS:
        result[column] = result[column].fillna("").astype(str)

    result["release_year"] = pd.to_numeric(result["release_year"], errors="coerce")
    result["rating"] = pd.to_numeric(result["rating"], errors="coerce")

    for column in TEXT_COLUMNS:
        result[f"clean_{column}"] = result[column].map(clean_text)

    result["tags"] = result.apply(build_tags, axis=1)
    result["tags"] = result["tags"].replace("", "unknown")

    return result.reset_index(drop=True)


def build_tags(row: pd.Series) -> str:
    """Combine useful metadata into one content representation."""
    pieces = [row.get(column, "") for column in TEXT_COLUMNS]
    return " ".join(piece for piece in pieces if piece).strip()


# -----------------------------------------------------------------------------
# EXPLORATION
# -----------------------------------------------------------------------------
def inspect_dataset(df: pd.DataFrame) -> None:
    """Print beginner-friendly dataset inspection information."""
    print("\n" + "=" * 70)
    print("DATASET INSPECTION")
    print("=" * 70)
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nMissing values:")
    print(df.isna().sum().to_string())
    print("\nDuplicate titles:", df["title"].duplicated().sum())


def genre_series(df: pd.DataFrame) -> pd.Series:
    """Convert multi-genre strings into one genre per row for counting."""
    if "genres" not in df.columns:
        return pd.Series(dtype=str)

    values: list[str] = []
    for raw in df["genres"].fillna(""):
        parts = re.split(r"[,|;/]+", str(raw))
        values.extend(part.strip() for part in parts if part.strip())
    return pd.Series(values, dtype=str)


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print useful statistics without inventing evaluation results."""
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Number of movies: {len(df):,}")

    years = pd.to_numeric(df.get("release_year"), errors="coerce").dropna()
    if not years.empty:
        print(f"Release years: {int(years.min())} - {int(years.max())}")

    ratings = pd.to_numeric(df.get("rating"), errors="coerce").dropna()
    if not ratings.empty:
        print(f"Rating range: {ratings.min():.1f} - {ratings.max():.1f}")

    genres = genre_series(df)
    if not genres.empty:
        print("\nMost common genres:")
        print(genres.value_counts().head(10).to_string())


# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
class MovieRecommender:
    """TF-IDF + cosine-similarity content-based recommender."""

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self.movies: Optional[pd.DataFrame] = None
        self.title_to_index: dict[str, int] = {}
        self.fitted = False

    def fit(self, movies: pd.DataFrame) -> "MovieRecommender":
        """Fit TF-IDF vectors to cleaned movie tags."""
        if "tags" not in movies.columns:
            raise ValueError("Movies must be cleaned and contain a 'tags' column before fitting.")

        self.movies = movies.reset_index(drop=True).copy()
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
        )

        self.matrix = self.vectorizer.fit_transform(self.movies["tags"].fillna(""))
        self.title_to_index = {
            self._normalize_title(title): index
            for index, title in enumerate(self.movies["title"])
        }
        self.fitted = True
        return self

    @staticmethod
    def _normalize_title(title: object) -> str:
        text = str(title).lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text)

    def _check_fitted(self) -> None:
        if not self.fitted or self.movies is None or self.vectorizer is None or self.matrix is None:
            raise RuntimeError("The recommender has not been fitted yet.")

    def search(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Find titles containing all or part of a query."""
        self._check_fitted()
        query = str(query).strip()
        if not query:
            return self.movies.head(limit).copy()

        normalized_query = self._normalize_title(query)
        title_series = self.movies["title"].map(self._normalize_title)
        mask = title_series.str.contains(re.escape(normalized_query), regex=True, na=False)
        return self.movies.loc[mask].head(limit).copy()

    def close_title_matches(self, query: str, limit: int = 5) -> list[str]:
        """Return lightweight lexical matches when an exact title is missing."""
        self._check_fitted()
        query_tokens = set(self._normalize_title(query).split())
        scored: list[tuple[int, str]] = []

        for title in self.movies["title"]:
            title_tokens = set(self._normalize_title(title).split())
            overlap = len(query_tokens & title_tokens)
            if overlap:
                scored.append((overlap, title))

        scored.sort(key=lambda item: (-item[0], item[1].lower()))
        return [title for _, title in scored[:limit]]

    def _shared_terms(self, index_a: int, index_b: int, limit: int = 5) -> list[str]:
        """Explain recommendations using overlapping cleaned metadata terms."""
        assert self.movies is not None
        a = set(str(self.movies.iloc[index_a]["tags"]).split())
        b = set(str(self.movies.iloc[index_b]["tags"]).split())
        ignored = {
            "the", "and", "with", "from", "that", "this", "into", "movie",
            "film", "one", "new", "man", "woman", "story", "people",
        }
        shared = sorted((a & b) - ignored, key=lambda item: (-len(item), item))
        return shared[:limit]

    def recommend(
        self,
        movie_title: str,
        number_of_recommendations: int = 10,
        genre: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        min_rating: Optional[float] = None,
    ) -> list[Recommendation]:
        """Return dynamically-ranked recommendations for a selected movie."""
        self._check_fitted()

        if number_of_recommendations < 1:
            raise ValueError("number_of_recommendations must be at least 1.")

        key = self._normalize_title(movie_title)
        if key not in self.title_to_index:
            raise KeyError(movie_title)

        source_index = self.title_to_index[key]
        similarities = cosine_similarity(self.matrix[source_index], self.matrix).ravel()
        ranked_indices = np.argsort(-similarities)

        recommendations: list[Recommendation] = []
        seen_titles: set[str] = set()

        for index in ranked_indices:
            if int(index) == source_index:
                continue

            row = self.movies.iloc[int(index)]
            title = str(row["title"])
            normalized_title = self._normalize_title(title)

            if normalized_title in seen_titles:
                continue

            if genre:
                genre_text = str(row.get("genres", "")).lower()
                if genre.lower() not in genre_text:
                    continue

            year = row.get("release_year")
            year_value = None if pd.isna(year) else int(year)

            if min_year is not None and (year_value is None or year_value < min_year):
                continue
            if max_year is not None and (year_value is None or year_value > max_year):
                continue

            rating = row.get("rating")
            rating_value = None if pd.isna(rating) else float(rating)
            if min_rating is not None and (rating_value is None or rating_value < min_rating):
                continue

            shared = self._shared_terms(source_index, int(index))
            explanation = (
                "Shares metadata terms: " + ", ".join(shared)
                if shared
                else "Ranked by similarity across the available movie metadata."
            )

            recommendations.append(
                Recommendation(
                    title=title,
                    genres=str(row.get("genres", "Unknown")),
                    release_year=year_value if year_value is not None else "Unknown",
                    rating=rating_value if rating_value is not None else "Unknown",
                    similarity=float(similarities[index]),
                    explanation=explanation,
                )
            )
            seen_titles.add(normalized_title)

            if len(recommendations) >= number_of_recommendations:
                break

        return recommendations

    def recommendation_dataframe(self, movie_title: str, number_of_recommendations: int = 10, **filters) -> pd.DataFrame:
        """Return recommendations as a DataFrame for analysis or export."""
        records = [recommendation.__dict__ for recommendation in self.recommend(
            movie_title, number_of_recommendations, **filters
        )]
        return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------
def plot_genres(df: pd.DataFrame, top_n: int = 10) -> None:
    """Plot the most common genres."""
    genres = genre_series(df).value_counts().head(top_n).sort_values()
    if genres.empty:
        print("No genre information is available for this plot.")
        return

    plt.figure(figsize=(10, 6))
    genres.plot(kind="barh")
    plt.title("Most Common Movie Genres")
    plt.xlabel("Number of Movies")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.show()


def plot_release_years(df: pd.DataFrame) -> None:
    """Plot movie counts by release year."""
    years = pd.to_numeric(df["release_year"], errors="coerce").dropna()
    if years.empty:
        print("No release-year information is available for this plot.")
        return

    counts = years.astype(int).value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    counts.plot(kind="line", marker="o")
    plt.title("Movie Release-Year Distribution")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Movies")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_ratings(df: pd.DataFrame) -> None:
    """Plot the distribution of available ratings."""
    ratings = pd.to_numeric(df["rating"], errors="coerce").dropna()
    if ratings.empty:
        print("No rating information is available for this plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(ratings, bins=10, kde=True)
    plt.title("Movie Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of Movies")
    plt.tight_layout()
    plt.show()


def plot_recommendation_scores(recommendations: list[Recommendation]) -> None:
    """Plot similarity scores for the returned recommendations."""
    if not recommendations:
        print("There are no recommendations to plot.")
        return

    frame = pd.DataFrame({
        "title": [r.title for r in recommendations],
        "similarity": [r.similarity for r in recommendations],
    }).sort_values("similarity")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=frame, x="similarity", y="title")
    plt.title("Recommendation Similarity Scores")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Movie")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# MODEL PERSISTENCE
# -----------------------------------------------------------------------------
def save_recommender(recommender: MovieRecommender, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Save the vectorizer, metadata, and model configuration with joblib."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(recommender, target)
    return target


def load_recommender(path: str | Path = DEFAULT_MODEL_PATH) -> MovieRecommender:
    """Load a previously saved recommender."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Model file does not exist: {target}")
    model = joblib.load(target)
    if not isinstance(model, MovieRecommender):
        raise TypeError("The saved file does not contain a MovieRecommender object.")
    return model


# -----------------------------------------------------------------------------
# OUTPUT HELPERS
# -----------------------------------------------------------------------------
def print_recommendations(recommendations: list[Recommendation]) -> None:
    """Print recommendations in a clean beginner-friendly format."""
    if not recommendations:
        print("\nNo movies matched the selected filters.")
        return

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    for number, recommendation in enumerate(recommendations, start=1):
        rating = (
            f"{recommendation.rating:.1f}"
            if isinstance(recommendation.rating, float)
            else str(recommendation.rating)
        )
        print(f"\n{number}. {recommendation.title}")
        print(f"   Genre: {recommendation.genres}")
        print(f"   Year: {recommendation.release_year}")
        print(f"   Rating: {rating}")
        print(f"   Similarity: {recommendation.similarity:.3f}")
        print(f"   Explanation: {recommendation.explanation}")

    print("\nImportant: similarity is a mathematical metadata-similarity measure, not a movie-quality score.")


def print_search_results(results: pd.DataFrame) -> None:
    """Print movie-search results."""
    if results.empty:
        print("No matching movies found.")
        return

    print("\nMatching movies:")
    for _, row in results.iterrows():
        year = row.get("release_year", "Unknown")
        if pd.isna(year):
            year = "Unknown"
        print(f"- {row['title']} ({year})")


# -----------------------------------------------------------------------------
# COMMAND-LINE INTERFACE
# -----------------------------------------------------------------------------
def ask_integer(prompt: str, default: int, minimum: int = 1, maximum: int = 20) -> int:
    """Read a validated integer from the user."""
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"Invalid number. Using {default}.")
        return default
    if not minimum <= value <= maximum:
        print(f"Please choose between {minimum} and {maximum}. Using {default}.")
        return default
    return value


def interactive_cli(recommender: MovieRecommender) -> None:
    """Run the main beginner-friendly command-line application."""
    print("\n" + "=" * 70)
    print("             MOVIE RECOMMENDATION SYSTEM")
    print("=" * 70)
    print("Content-Based Filtering | TF-IDF | Cosine Similarity")
    print("Type 'help' to see commands or 'quit' to exit.\n")

    while True:
        command = input("movie-recommender> ").strip()

        if command.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        if command.lower() == "help":
            print(
                "\nCommands:\n"
                "  recommend  - recommend movies from a liked movie\n"
                "  search     - search available movie titles\n"
                "  list       - show available titles\n"
                "  summary    - show dataset information\n"
                "  quit       - exit the program\n"
            )
            continue

        if command.lower() == "list":
            print("\nAvailable movies:")
            for title in recommender.movies["title"].tolist():
                print(f"- {title}")
            continue

        if command.lower() == "summary":
            print_dataset_summary(recommender.movies)
            continue

        if command.lower() == "search":
            query = input("Search movie title: ").strip()
            print_search_results(recommender.search(query, limit=15))
            continue

        if command.lower() != "recommend":
            print("Unknown command. Type 'help' for available commands.")
            continue

        title = input("Enter a movie you like: ").strip()
        if not title:
            print("Please enter a movie title.")
            continue

        try:
            number = ask_integer("How many recommendations", 5, 1, 20)
            genre = input("Optional genre filter (press Enter to skip): ").strip() or None

            min_rating_raw = input("Optional minimum rating (press Enter to skip): ").strip()
            min_rating = float(min_rating_raw) if min_rating_raw else None

            min_year_raw = input("Optional minimum release year (press Enter to skip): ").strip()
            min_year = int(min_year_raw) if min_year_raw else None

            max_year_raw = input("Optional maximum release year (press Enter to skip): ").strip()
            max_year = int(max_year_raw) if max_year_raw else None

            print("\nGenerating recommendations...")
            recommendations = recommender.recommend(
                title,
                number_of_recommendations=number,
                genre=genre,
                min_rating=min_rating,
                min_year=min_year,
                max_year=max_year,
            )
            print_recommendations(recommendations)

        except KeyError:
            print("\nMovie not found. Please choose a movie from the available list.")
            matches = recommender.close_title_matches(title)
            if matches:
                print("Possible matches:")
                for match in matches:
                    print(f"- {match}")
        except ValueError as error:
            print(f"\nInput error: {error}")


# -----------------------------------------------------------------------------
# ARGUMENT-PARSER INTERFACE
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Content-based movie recommender using TF-IDF and cosine similarity."
    )
    parser.add_argument("--data", default=None, help="Path to a movies CSV file.")
    parser.add_argument("--movie", default=None, help="Movie title for one-shot recommendation mode.")
    parser.add_argument("--top", type=int, default=5, help="Number of recommendations.")
    parser.add_argument("--genre", default=None, help="Optional genre filter.")
    parser.add_argument("--min-rating", type=float, default=None, help="Optional minimum rating.")
    parser.add_argument("--min-year", type=int, default=None, help="Optional minimum release year.")
    parser.add_argument("--max-year", type=int, default=None, help="Optional maximum release year.")
    parser.add_argument("--save-model", action="store_true", help="Save the fitted recommender with joblib.")
    parser.add_argument("--load-model", action="store_true", help="Load the saved model instead of fitting again.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Model path for save/load.")
    parser.add_argument("--inspect", action="store_true", help="Print detailed dataset inspection.")
    parser.add_argument("--summary", action="store_true", help="Print dataset summary.")
    parser.add_argument("--plot-genres", action="store_true", help="Show genre distribution plot.")
    parser.add_argument("--plot-years", action="store_true", help="Show release-year plot.")
    parser.add_argument("--plot-ratings", action="store_true", help="Show rating distribution plot.")
    parser.add_argument("--plot-recommendations", action="store_true", help="Plot recommendation similarity scores.")
    return parser


def create_recommender(data_path: Optional[str], load_model_flag: bool, model_path: str) -> MovieRecommender:
    """Create a recommender by loading a saved model or fitting from data."""
    if load_model_flag:
        print(f"Loading saved model: {model_path}")
        return load_recommender(model_path)

    raw_df, used_demo = load_dataset(data_path)
    cleaned_df = clean_dataset(raw_df)

    if used_demo:
        print(f"Demo movies available: {len(cleaned_df)}")

    recommender = MovieRecommender()
    recommender.fit(cleaned_df)
    print(f"TF-IDF matrix shape: {recommender.matrix.shape}")
    return recommender


def run_once(args: argparse.Namespace) -> None:
    """Run one-shot CLI mode and optional analysis commands."""
    recommender = create_recommender(args.data, args.load_model, args.model_path)

    if args.inspect:
        inspect_dataset(recommender.movies)

    if args.summary:
        print_dataset_summary(recommender.movies)

    if args.plot_genres:
        plot_genres(recommender.movies)

    if args.plot_years:
        plot_release_years(recommender.movies)

    if args.plot_ratings:
        plot_ratings(recommender.movies)

    if args.save_model:
        path = save_recommender(recommender, args.model_path)
        print(f"Saved model: {path}")

    if args.movie:
        try:
            recommendations = recommender.recommend(
                args.movie,
                number_of_recommendations=args.top,
                genre=args.genre,
                min_rating=args.min_rating,
                min_year=args.min_year,
                max_year=args.max_year,
            )
            print_recommendations(recommendations)
            if args.plot_recommendations:
                plot_recommendation_scores(recommendations)
        except KeyError:
            print(f"Movie not found: {args.movie}")
            print("Possible matches:")
            for match in recommender.close_title_matches(args.movie):
                print(f"- {match}")

    has_noninteractive_action = any(
        [
            args.inspect,
            args.summary,
            args.plot_genres,
            args.plot_years,
            args.plot_ratings,
            args.movie,
            args.save_model,
            args.load_model,
        ]
    )

    if not has_noninteractive_action:
        interactive_cli(recommender)


# -----------------------------------------------------------------------------
# BEGINNER NOTEBOOK-STYLE EXAMPLE FUNCTIONS
# -----------------------------------------------------------------------------
def notebook_step_by_step(data_path: Optional[str] = None) -> MovieRecommender:
    """A readable sequence that can be copied into a notebook cell by cell."""
    raw_df, used_demo = load_dataset(data_path)

    # Step 1: inspect the raw data.
    inspect_dataset(raw_df)

    # Step 2: clean missing values, duplicates, text, and numeric fields.
    movies = clean_dataset(raw_df)

    # Step 3: build the TF-IDF representation.
    recommender = MovieRecommender()
    recommender.fit(movies)

    # Step 4: request recommendations dynamically.
    example_title = movies.iloc[0]["title"]
    recommendations = recommender.recommend(example_title, 5)
    print_recommendations(recommendations)

    # Step 5: save a reusable artifact.
    save_recommender(recommender)

    return recommender


# -----------------------------------------------------------------------------
# EDUCATIONAL EXPLANATION
# -----------------------------------------------------------------------------
def print_learning_summary() -> None:
    """Print the project concepts in the same order as the ML pipeline."""
    print(
        """
LEARNING PIPELINE
-----------------
1. Movie dataset
2. Data cleaning
3. Feature engineering
4. Text preprocessing
5. TF-IDF vectorization
6. Cosine similarity
7. Similarity ranking
8. Top-N recommendations

CONTENT-BASED FILTERING
-----------------------
The recommender compares the content/metadata of movies. It does not learn
personal preferences from a population of users in this beginner version.

TF-IDF
------
TF-IDF converts text into weighted numerical features. Terms that help
separate documents receive useful weights while very common terms receive
less influence.

COSINE SIMILARITY
-----------------
Cosine similarity compares the direction of two TF-IDF vectors. A larger
value means the metadata vectors are more similar.

LIMITATIONS
-----------
- It depends heavily on metadata quality.
- It can struggle with movies that have little metadata.
- It does not truly understand stories like a human.
- It does not model other users' preferences.
- It can create a filter bubble by recommending more of the same type.
- Similarity is not the same thing as movie quality.

FUTURE EXTENSIONS
-----------------
- Multiple favorite movies
- Genre/year/rating personalization
- User profiles and ratings
- Collaborative filtering
- Hybrid recommendation
- Embeddings and semantic similarity
- Diversity and novelty optimization
- Production web application
"""
    )


# -----------------------------------------------------------------------------
# SELF-CHECKS
# -----------------------------------------------------------------------------
def run_self_check() -> bool:
    """Run small internal checks to catch common beginner-project errors."""
    print("Running self-check...")
    df = demo_dataframe()
    cleaned = clean_dataset(df)
    model = MovieRecommender().fit(cleaned)

    assert len(cleaned) == len(df)
    assert model.matrix.shape[0] == len(cleaned)
    assert model.matrix.shape[1] > 0

    recommendations = model.recommend("Interstellar", 3)
    assert len(recommendations) == 3
    assert all(item.title != "Interstellar" for item in recommendations)
    assert all(0 <= item.similarity <= 1 for item in recommendations)

    search_results = model.search("inter", limit=10)
    assert not search_results.empty

    print("Self-check passed.")
    return True


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top must be at least 1.")

    if args.movie is None and not any(
        [
            args.inspect,
            args.summary,
            args.plot_genres,
            args.plot_years,
            args.plot_ratings,
            args.save_model,
            args.load_model,
        ]
    ):
        # The interactive CLI is the default beginner experience.
        pass

    try:
        run_once(args)
    except FileNotFoundError as error:
        print(f"File error: {error}", file=sys.stderr)
        raise SystemExit(1)
    except ValueError as error:
        print(f"Data/input error: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
