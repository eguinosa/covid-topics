# Gelin Eguinosa Rosique

import time
from pprint import pprint

from gensim.corpora import Dictionary
from gensim.models import LdaModel

from docs_stream import DocumentsManager
from corpus_tokenizer import CorpusTokenizer
from docs_tokenization import corpus_tokenization


def run_time(time_diff):
    """
    Transforms the elapsed time from the start of the program to a new format
    in hours, minutes and seconds.
    :param time_diff: The time difference from the start to the end of the
    program.
    :return: A string containing the elapsed time in <hours:minutes:seconds>
    """
    hours = int(time_diff / 3600)
    minutes = int((time_diff - hours * 3600) / 60)
    seconds = int(time_diff - hours * 3600 - minutes * 60)
    milliseconds = int((time_diff - int(time_diff)) * 1000)

    return f'{hours} h : {minutes} min : {seconds} sec : {milliseconds} mill'


if __name__ == '__main__':
    # To record the runtime of the program
    start_time = time.time()

    # Load all the documents about Covid-19 from the Wikipedia in the
    # docs/ folder.
    print("\nLoading Documents.")
    doc_files = DocumentsManager()

    # Tokenize all the Documents loaded using Spacy
    print("Tokenizing all the documents.")
    # tokenizer = CorpusTokenizer(doc_files.documents_texts())
    corpus_tokens = corpus_tokenization(doc_files.documents_texts())

    # Create the dictionary
    dictionary = Dictionary(corpus_tokens)

    # Bag-of-words representation of the documents
    corpus_bow = [dictionary.doc2bow(doc) for doc in corpus_tokens]

    # Train the LDA Model
    print("Training the LDA Model.")

    # Set training parameters.
    num_topics = 4
    chunksize = 20
    passes = 10
    iterations = 400
    eval_every = None

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    lda_model = LdaModel(
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

    # Print Corpus Information
    print(f"\nNumber of documents: {len(doc_files.documents)}")
    print(f"Number of unique tokens: {len(dictionary)}")

    # Printing Topics
    top_topics = lda_model.top_topics(corpus_bow)

    # Average topic coherence
    average_coherence = sum([topic[1] for topic in top_topics]) / len(top_topics)
    print("\nAverage topic coherence: %.4f." % average_coherence)

    # Top Topics
    print("\nThe Top Topics are:")
    pprint(top_topics)

    # Print the total runtime of the program
    elapsed_time = time.time() - start_time
    elapsed_time = run_time(elapsed_time)
    print(f"\nTotal time of the program: {elapsed_time}")
