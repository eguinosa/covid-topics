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

    return model
