# Gelin Eguinosa Rosique

from sys import exit, argv
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
    print("Done. ")
    print(f"[{stopwatch.run_time()}]")

    # Tokenize all the Documents loaded using Spacy
    print("\nTokenizing the documents.")
    # Load the CorpusTokenizer, if it was saved.
    if CorpusTokenizer.are_tokens_saved():
        print("Loading the saved tokenized documents.")
        tokenizer = CorpusTokenizer.saved_tokenizer()
    # Create the corpus tokenizer, if it can't be loaded.
    else:
        print("Tokenizing the documents from scratch.")
        tokenizer = CorpusTokenizer(doc_files.documents_texts())
    print("Done. ")
    print(f"[{stopwatch.run_time()}]")

    # Creating the Dictionary and the Corpus Bag-of-Words
    print("\nCreating the Dictionary and the Corpus Bag-of-Words.")
    # Load the corpus tokenizer if available, create it otherwise.
    if TopicManager.is_topic_manager_saved():
        print("Loading the saved dictionary and corpus bag-of-words.")
        topic_manager = TopicManager.saved_topic_manager()
    else:
        print("Creating the dictionary and corpus bag-of-words from scratch.")
        topic_manager = TopicManager(tokenizer)
    print("Done. ")
    print(f"[{stopwatch.run_time()}]")

    # Train the LDA Model
    print("\nTraining the LDA Model.")

    # Set training parameters.
    num_topics = 4
    chunksize = 20
    passes = 10
    iterations = 400
    eval_every = None

    # Check if the desired number of topics was passed as an argument in the
    # Command Line
    if len(argv) > 1:
        # We received a parameter from the command line.
        # Assume it is the desired number of topics.
        num_topics = int(argv[1])

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

    # Top Topics
    print("\nThe Top Topics are:")
    pprint(top_topics)

    # Average topic coherence
    average_coherence = sum([topic[1] for topic in top_topics]) / len(top_topics)
    print(f"\nThe {num_topics} topics average topic coherence is: {average_coherence:.4f}")

    # Print the total runtime of the program
    print("\nProgram Finished.")
    print(f"[{stopwatch.run_time()}]")
