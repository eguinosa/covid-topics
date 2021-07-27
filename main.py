# Gelin Eguinosa Rosique

from pprint import pprint
from docs_stream import docs_stream
from docs_tokenization import docs_tokenization
from topic_processing import topic_processing


if __name__ == '__main__':
    # Load all the documents about Covid-19 from the Wikipedia in the
    # docs/ folder.
    print("\nLoading Documents.")
    documents = docs_stream()

    # Tokenize all the Documents loaded using Spacy
    print("Tokenizing all the documents.")
    docs_tokens = docs_tokenization(documents)

    # Create the dictionary, transform the documents in Bag-of-Words
    # and create LDA Model
    print("Training the LDA Model.")
    topic_process = topic_processing(docs_tokens)
    corpus_bow = topic_process['corpus_bow']
    dictionary = topic_process['dictionary']
    lda_model = topic_process['lda_model']

    # Print Corpus Information
    print(f"\nNumber of documents: {len(documents)}")
    print(f"Number of unique tokens: {len(dictionary)}")

    # Printing Topics
    top_topics = lda_model.top_topics(corpus_bow)

    # Average topic coherence
    average_coherence = sum([topic[1] for topic in top_topics]) / len(top_topics)
    print("\nAverage topic coherence: %.4f." % average_coherence)

    # Top Topics
    print("\nThe Top Topics are:")
    pprint(top_topics)
