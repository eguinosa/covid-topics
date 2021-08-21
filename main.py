# Gelin Eguinosa Rosique

from sys import exit
from pprint import pprint

from docs_stream import DocumentsManager
from corpus_tokenizer import CorpusTokenizer
from topic_processing import TopicManager
from time_keeper import TimeKeeper


if __name__ == '__main__':
    # To record the runtime of the program
    stopwatch = TimeKeeper()

    # First, check if the documents are available.
    if not DocumentsManager.documents_available():
        print("\nNo documents where found in the 'docs' folder.")
        exit()
    # Load the documents inside the 'docs' folder.
    print("\nLoading Documents.")
    doc_files = DocumentsManager()

    # Tokenize all the Documents loaded using Spacy
    print("\nTokenizing the documents:")
    # Check if the user wants to use the saved tokenized documents in case they
    # are available.
    use_saved_info = False
    if CorpusTokenizer.are_tokens_saved():
        print("There are saved tokenized documents available.")
        # Check what the user wants to do, and stop recording the time while
        # we wait for the answer
        stopwatch.pause()
        answer = input("Would you like to use them? (yes/no) ")
        stopwatch.restart()
        if answer.lower().strip() in ['yes', 'y']:
            use_saved_info = True
    # Load or Create the CorpusTokenizer
    if use_saved_info:
        print("Loading the saved tokenized documents.")
        tokenizer = CorpusTokenizer.saved_tokenizer()
    else:
        print("Tokenizing the documents from scratch.")
        tokenizer = CorpusTokenizer(doc_files.documents_texts())

    # Creating the Dictionary and the Corpus Bag-of-Words
    print("\nCreating the Dictionary and the Corpus Bag-of-Words.")
    # Check if the user wants to use the saved dictionary and corpus
    # bag-of-words if they are available.
    use_saved_info = False
    if TopicManager.is_topic_manager_saved():
        print("The dictionary and corpus bag-of-words are saved and available.")
        # Check what the user wants to do, and stop recording the time while
        # we wait for the answer
        stopwatch.pause()
        answer = input("Would you like to use them? (yes/no) ")
        stopwatch.restart()
        if answer.lower().strip() in ['yes', 'y']:
            use_saved_info = True
    # Load or Create the TopicManager
    if use_saved_info:
        print("Loading the saved dictionary and corpus bag-of-words.")
        topic_manager = TopicManager.saved_topic_manager()
    else:
        print("Creating the dictionary and corpus bag-of-words from scratch.")
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
    print(f"\nRuntime -> {stopwatch.run_time()}")
