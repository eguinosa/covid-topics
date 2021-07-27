# Gelin Eguinosa Rosique

from gensim.corpora import Dictionary
from gensim.models import LdaModel

from docs_stream import docs_stream
from docs_tokenization import docs_tokenization
from pprint import pprint


def lda_model(corpus_text):
    """
    Receives a corpus in the form of tokenized documents and creates a LDA Model
    from them.
    :param corpus_text: Sequence of tokenized documents.
    :return: An LDA Model
    """
    # Create a dictionary representation of the documents
    dictionary = Dictionary(corpus_text)
    # Bag-of-words representation of the documents
    corpus_bow = [dictionary.doc2bow(doc) for doc in corpus_text]

    # Train the LDA Model
    # Set training parameters.
    num_topics = 10
    chunksize = 20
    passes = 20
    iterations = 200
    eval_every = None

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = LdaModel(
        corpus=corpus_bow,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    # Printing Topics
    top_topics = model.top_topics(corpus_bow)  # , num_words=20)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('\nAverage topic coherence: %.4f.' % avg_topic_coherence)

    print("\nThe top topics are:")
    pprint(top_topics)


print("Loading Documents...")
docs = docs_stream()
print("Document Tokenization...")
docs_tokens = docs_tokenization(docs)
print("LDA Model Creation...")
lda_model(docs_tokens)
