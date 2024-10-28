import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('netflix_titles.csv')  # Replace with your actual file path

# Display the first few rows of the dataset
df.head()
# Fill missing values in 'director' and 'cast' with an empty string
df['director'] = df['director'].fillna('')
df['cast'] = df['cast'].fillna('')
df['description'] = df['description'].fillna('')

# Combine features into a single text column
df['combined_features'] = (
    df['type'] + ' ' +
    df['director'] + ' ' +
    df['cast'] + ' ' +
    df['listed_in'] + ' ' +
    df['description']
)
# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the combined features
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
def get_recommendations(title, cosine_sim=cosine_sim, df=df):
    # Convert the input title to lowercase
    title_lower = title.lower()

    # Check if the title is in the DataFrame
    if title_lower not in df['title'].str.lower().values:
        print(f"The title '{title}' is not found in the dataset.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Find the index of the show that matches the title
    idx = df.index[df['title'].str.lower() == title_lower].tolist()[0]

    # Get the pairwise similarity scores for all shows
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort shows based on similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 similar shows, ignoring the first one (itself)
    sim_scores = sim_scores[1:11]

    # Get the indices of the recommended shows
    show_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar shows
    return df.iloc[show_indices][['title', 'director', 'listed_in']]

title = 'Stranger Things'  # Replace with an actual title from your dataset
recommendations = get_recommendations(title)

if not recommendations.empty:
    print("Recommendations based on the show:", title)
    print(recommendations)
else:
    print("No recommendations found.")

