# Gelin Eguinosa Rosique

import pickle
from os import mkdir
from os.path import isdir, isfile, join

from docs_tokenization import lazy_corpus_tokenization


class CorpusTokenizer:
    """
    Class to tokenize the documents of a corpus and save the results in case
    they are needed later.
    """
    # Save the location of the tokens for static methods.
    tokens_folder = 'data/corpus_tokens'

    # Save index name for static methods.
    index_name = 'index.pickle'

    def __init__(self, documents, _use_saved=False):
        """
        Receives the texts from the documents in the corpus and creates, and
        transforms each document into an array of tokens.
        Removes all the stop words, punctuation symbols and numbers in the
        documents, lowercases the text and lemmatizes each token.
        :param documents: An iterable sequence containing the texts of the
        documents in the corpus.
        :param _use_saved: Bool to determine if we used a previously
        calculated tokenization of the corpus, or if we start from scratch, even
        if we have the result of the tokenization saved.
        """
        # The Location of the data of the program
        self._data_folder = 'data'
        # Create data folder if it doesn't exist
        if not isdir(self._data_folder):
            mkdir(self._data_folder)

        # The Location of the Tokens
        self._tokens_folder = join(self._data_folder, 'corpus_tokens')
        # Create the Tokens folder if it doesn't exist
        if not isdir(self._tokens_folder):
            mkdir(self._tokens_folder)

        # The location of the index of the tokens
        self._index_name = 'index.pickle'

        # Check if the user wants to use the saved tokens
        if _use_saved:
            index_path = join(self._tokens_folder, self._index_name)
            # Check if we saved the tokens for this corpus
            if not isfile(index_path):
                raise Exception("No tokens previously saved for this corpus.")
            # Load the tokens information from the index file:
            with open(index_path, 'rb') as file:
                self.tokens_info = pickle.load(file)

        # Do the tokenization of the documents
        else:
            # Initialize values
            self.tokens_info = {}
            doc_id = 0
            base_name = 'doc_tokens_'

            # Do a lazy tokenization and save the results
            for doc_tokens in lazy_corpus_tokenization(documents):
                # Create the name of the file where the tokenization will be
                # saved
                doc_id += 1
                doc_name = base_name + str(doc_id) + '.pickle'
                # Save the name in a dictionary for later use
                self.tokens_info[doc_id] = doc_name
                # Save the tokenization in a file.
                file_path = join(self._tokens_folder, doc_name)
                with open(file_path, 'wb') as file:
                    pickle.dump(doc_tokens, file)

            # Save the index of the tokens
            index_path = join(self._tokens_folder, self._index_name)
            with open(index_path, 'wb') as file:
                pickle.dump(self.tokens_info, file)

    def corpus_tokens(self):
        """
        Load, one at a time, the saved tokens of the documents.
        :return: a sequence of the tokens of the documents in the corpus.
        """
        # Iterate through the names of the files where the tokens are stored
        for file_name in self.tokens_info.values():
            tokens_path = join(self._tokens_folder, file_name)
            # Load the tokens belonging to one of the documents
            with open(tokens_path, 'rb') as file:
                doc_tokens = pickle.load(file)
            yield doc_tokens

    @staticmethod
    def saved_tokens():
        """
        Check if the tokens from a previos tokenizer are saved and ready to
        be used.
        :return: bool representing if we have the tokens or the corpus saved or
        not.
        """
        tokens_folder = CorpusTokenizer.tokens_folder
        index_name = CorpusTokenizer.index_name
        index_path = join(tokens_folder, index_name)

        # Check that the index file exists.
        if not isfile(index_path):
            return False

        # Open the index file and check there is information available
        with open(index_path, 'rb') as file:
            tokens_info = pickle.load(file)

        # Check if token_info has null values or is empty
        if not tokens_info or not len(tokens_info):
            return False

        return True

    @classmethod
    def saved_tokenizer(cls):
        """
        Creates a CorpusTokenizer from the information saved by a previous
        tokenizer.
        :return: A CorpusTokenizer
        """
        return cls(None, _use_saved=True)
