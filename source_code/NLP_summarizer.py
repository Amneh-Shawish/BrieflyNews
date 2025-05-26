# Import necessary libraries
import nltk
import networkx as nx
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK sentence tokenizer model
nltk.download('punkt')

def preprocess_text(text):
    """
    Preprocess the input text:
    - Split the text into individual sentences using NLTK.
    """
    return sent_tokenize(text)

def build_similarity_matrix(sentences):
    """
    Build a similarity matrix for the given sentences:
    - Convert sentences into TF-IDF vectors.
    - Compute cosine similarity between each pair of sentences.
    - Return the similarity matrix as a 2D array.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    return similarity_matrix

def rank_sentences(similarity_matrix):
    """
    Rank sentences using the TextRank algorithm:
    - Treat each sentence as a node in a graph.
    - Use cosine similarity as edge weights.
    - Apply PageRank to score each sentence.
    - Return a dictionary of sentence indices and their scores.
    """
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)
    return scores

def generate_summary(text, num_sentences=3):
    """
    Generate an extractive summary:
    - Preprocess and tokenize the input text into sentences.
    - Compute similarity matrix and rank sentences.
    - Select top N highest-ranked sentences.
    - Return them as the summary.
    """
    sentences = preprocess_text(text)
    
    # If text is too short, return as-is
    if len(sentences) <= num_sentences:
        return text

    # Build similarity matrix and rank sentences
    similarity_matrix = build_similarity_matrix(sentences)
    scores = rank_sentences(similarity_matrix)
    
    # Sort sentences by their TextRank scores in descending order
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Select the top N sentences
    top_sentences = [s for _, s in ranked_sentences[:num_sentences]]

    # Combine top sentences into the final summary
    return ' '.join(top_sentences)

# Sample usage (for testing the summarizer)
if __name__ == "__main__":
    # Get body value
    text = body->value;

    # Generate summary from body input
    summary = generate_summary(text, num_sentences=2)
    print("\nGenerated Summary:\n", summary)
