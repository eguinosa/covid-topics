# Gelin Eguinosa Rosique

import time
from pprint import pprint

from gensim.corpora import Dictionary
from gensim.models import LdaModel

from docs_stream import DocumentsManager
from corpus_tokenizer import CorpusTokenizer
from topic_processing import TopicManager


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
    if CorpusTokenizer.are_tokens_saved():
        print("Loading saved tokenizer.")
        tokenizer = CorpusTokenizer.saved_tokenizer()
    else:
        tokenizer = CorpusTokenizer(doc_files.documents_texts())

    # Creating the Dictionary and the Corpus Bag-of-Words
    print("Creating the Dictionary and the Corpus Bag-of-Words.")
    if TopicManager.is_topic_manager_saved():
        topic_manager = TopicManager.saved_topic_manager()
    else:
        topic_manager = TopicManager(tokenizer)

    # Train the LDA Model
    print("Training the LDA Model.")

    # Set training parameters.
    num_topics = 4
    chunksize = 20
    passes = 10
    iterations = 400
    eval_every = None

    # Create and train the LDA Model
    lda_model = topic_manager.lda_model(num_topics,
                                        chunksize,
                                        passes,
                                        iterations,
                                        eval_every)

    # Print Corpus Information
    print(f"\nNumber of documents: {len(doc_files.documents)}")
    print(f"Number of unique tokens: {len(topic_manager.dictionary)}")

    # Printing Topics
    top_topics = lda_model.top_topics(topic_manager.corpus_bow)

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
