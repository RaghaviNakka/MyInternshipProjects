import nltk
nltk.download('punkt_tab')
import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
import random

# Download necessary datasets
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('punkt')

# Load movie reviews data
def load_data():
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    random.shuffle(documents)
    return documents

# Text preprocessing: remove stopwords and lowercase words
def preprocess_words(words):
    stop_words = set(stopwords.words('english'))
    return [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

# Extract features: create a bag of words for the document
def document_features(document, word_features):
    document_words = set(document)
    features = {f'contains({word})': (word in document_words) for word in word_features}
    return features
def train_model(documents):
    # Preprocess all documents
    preprocessed_documents = [(preprocess_words(d), c) for (d, c) in documents]

    # Create a frequency distribution of words
    all_words = nltk.FreqDist(word for doc, _ in preprocessed_documents for word in doc)

    # Select the top 2000 words as features
    word_features = list(all_words)[:2000]

    # Create featuresets
    featuresets = [(document_features(d, word_features), c) for (d, c) in preprocessed_documents]

    # Split the data into training and test sets (80% train, 20% test)
    train_size = int(0.8 * len(featuresets))
    train_set, test_set = featuresets[:train_size], featuresets[train_size:]

    # Train a Naive Bayes classifier
    classifier = NaiveBayesClassifier.train(train_set)

    return classifier, test_set, word_features
def evaluate_model(classifier, test_set):
    accuracy = nltk.classify.accuracy(classifier, test_set)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    classifier.show_most_informative_features(10)
def analyze_sentiment(classifier, word_features, text):
    words = nltk.word_tokenize(text)
    preprocessed = preprocess_words(words)
    features = document_features(preprocessed, word_features)
    sentiment = classifier.classify(features)
    return sentiment

# Main script to run the above functions
if __name__ == '__main__':
    # Load and preprocess data
    documents = load_data()

    # Train the sentiment analysis model
    classifier, test_set, word_features = train_model(documents)

    # Evaluate the model
    evaluate_model(classifier, test_set)

    # Test with new examples
    samples = [
        "This movie was absolutely wonderful, full of thrilling scenes!",
        "I did not like the film at all. It was boring and predictable.",
        "It was an okay movie, nothing extraordinary but not terrible either."
    ]

    for sample in samples:
        print(f'Text: "{sample}"')
        print(f'Sentiment: {analyze_sentiment(classifier, word_features, sample)}')
