# Gelin Eguinosa Rosique

import time
from pprint import pprint

from docs_stream import docs_stream
from docs_tokenization import docs_tokenization
from topic_processing import topic_processing


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

    return f'{hours} h : {minutes} min : {seconds} seg : {milliseconds} mill'


if __name__ == '__main__':
    # To record the runtime of the program
    start_time = time.time()

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

    # Print the total runtime of the program
    elapsed_time = time.time() - start_time
    elapsed_time = run_time(elapsed_time)
    print(f"\nTotal time of the program: {elapsed_time}")
