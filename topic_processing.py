# Gelin Eguinosa Rosique

from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models import LdaModel

import pickle
from os import mkdir, listdir, remove
from os.path import isdir, isfile, abspath, join


class TopicManager:
    """
    Class to manage all the need processing using the gensim package.
    """
    # Location of the class data
    _data_folder = 'data'
    _dict_file = 'dictionary.pickle'
    _corpus_file = 'corpus_bow.mm'
    _lda_folder = 'lda_models'
    _lda_index_file = 'index_lda_model.pickle'
    _lda_prefix = 'lda_model_'

    def __init__(self, tokenizer, _use_saved=False):
        """
        Builds the dictionary, the corpus bag-of-words and the lda-model using
        the preprocessed tokens of the documents in the corpus.
        :param tokenizer: A CorpusTokenizer instance to get the tokens of
        the documents in a lazy way, document per document.
        """

        # Loading the saved TopicManager
        if _use_saved:
            # Check if the dictionary file exists.
            dict_path = join(self._data_folder, self._dict_file)
            if not isfile(dict_path):
                raise Exception("The dictionary of the TopicManager was not"
                                " saved.")
            # Load the dictionary
            with open(dict_path, 'rb') as file:
                self.dictionary = pickle.load(file)

            # Check if the corpus bag-of-word file exists.
            corpus_path = join(self._data_folder, self._corpus_file)
            if not isfile(corpus_path):
                raise Exception("The Corpus Bag-of-Words of the TopicManager"
                                " was not saved.")
            # Load the corpus bag-of-words
            corpus_full_path = abspath(corpus_path)
            self.corpus_bow = corpora.MmCorpus(corpus_full_path)

            # Check if the LDA Model index file exists.
            lda_index_path = join(self._data_folder, self._lda_folder,
                                  self._lda_index_file)
            if not isfile(lda_index_path):
                raise Exception("The LDA Model Index of the TopicManager was not"
                                " saved.")
            # Load the lda index:
            with open(lda_index_path, 'rb') as file:
                self.lda_index = pickle.load(file)

        # Create the TopicManager from scratch
        else:
            # Create data folder if it doesn't exist
            if not isdir(self._data_folder):
                mkdir(self._data_folder)

            # Create LDA Model folder if it doesn't exist
            lda_folder_path = join(self._data_folder, self._lda_folder)
            if not isdir(lda_folder_path):
                mkdir(lda_folder_path)

            # Create & Save the dictionary
            self.dictionary = Dictionary(tokenizer.corpus_tokens())
            # Filter out words that occur less than 2 documents, or more than
            # 75% of the documents.
            self.dictionary.filter_extremes(no_below=2, no_above=0.75)
            # Save the dictionary
            dict_path = join(self._data_folder, self._dict_file)
            with open(dict_path, 'wb') as file:
                pickle.dump(self.dictionary, file)

            # Create & Save the corpus bag-of-words representation, using a
            # lazy representation of the corpus bag-of-words, so we only have
            # one document bag-of-words at a time in memory.
            corpus_path = join(self._data_folder, self._corpus_file)
            corpus_full_path = abspath(corpus_path)
            corpora.MmCorpus.serialize(corpus_full_path,
                                       self._lazy_corpus_bow(tokenizer))
            self.corpus_bow = corpora.MmCorpus(corpus_full_path)

            # Create an index to keep track of the lda models that will be
            # created and save it.
            self.lda_index = {}
            lda_index_path = join(self._data_folder, self._lda_folder,
                                  self._lda_index_file)
            with open(lda_index_path, 'wb') as file:
                pickle.dump(self.lda_index, file)

            # Delete all the LDA Models created for a previous TopicManager
            lda_folder_path = join(self._data_folder, self._lda_folder)
            for file_name in listdir(lda_folder_path):
                lda_file_path = join(lda_folder_path, file_name)
                # Check is the name of the file is formatted as a lda model.
                if (isfile(lda_file_path) and file_name.startswith(self._lda_prefix)
                        and file_name.endswith('.pickle')):
                    # Once we confirm it is an LDA Model, delete the file
                    remove(lda_file_path)

    def _lazy_corpus_bow(self, tokenizer):
        """
        Create a sequence of the document in bag-of-words form that streams one
        document at a time, to be memory friendly.
        """
        # Iterating through the documents in tokens form and transform them in
        # bag-of-words
        for doc in tokenizer.corpus_tokens():
            yield self.dictionary.doc2bow(doc)

    def lda_model(self, num_topics, chunksize, passes=20, iterations=400,
                  eval_every=None):
        # Save the parameters in a tuple, so they are easier to use.
        lda_params = (num_topics, chunksize, passes, iterations, eval_every)

        # Check if a LDA Model with these parameters was already calculated.
        if lda_params in self.lda_index:
            # Get the name of the file where the LDA Model is saved.
            lda_model_file = self.lda_index[lda_params]
            lda_model_path = join(self._data_folder, self._lda_folder,
                                  lda_model_file)

            # Load the LDA Model
            with open(lda_model_path, 'rb') as file:
                lda_model = pickle.load(file)

            # Return the loaded LDA Model
            return lda_model

        # This LDA Model is not in the index, we need to calculate it.
        else:
            # Make the id to word dictionary
            temp = self.dictionary[0]  # This is only to "load" the dictionary
            id2word = self.dictionary.id2token

            # Create and Train the LDA Model
            lda_model = LdaModel(
                corpus=self.corpus_bow,
                id2word=id2word,
                chunksize=chunksize,
                alpha='auto',
                eta='auto',
                iterations=iterations,
                num_topics=num_topics,
                passes=passes,
                eval_every=eval_every
            )

            # Save the LDA Model
            lda_id = len(self.lda_index) + 1
            lda_model_name = self._lda_prefix + str(lda_id) + '.pickle'
            # Save the name in the lda_index
            self.lda_index[lda_params] = lda_model_name
            # Save in the LDA Model in a file
            lda_model_path = join(self._data_folder, self._lda_folder,
                                  lda_model_name)
            with open(lda_model_path, 'wb') as file:
                pickle.dump(lda_model, file)

            # Update the value of the Index of the LDA Model
            lda_index_path = join(self._data_folder, self._lda_folder,
                                  self._lda_index_file)
            with open(lda_index_path, 'wb') as file:
                pickle.dump(self.lda_index, file)

            # Return the calculated LDA Model
            return lda_model

    @classmethod
    def is_topic_manager_saved(cls):
        """
        Checks is the data from the TopicManager is saved and ready to be used.
        :return: Bool representing if we can load the saved TopicManager or we
        need to create it from scratch.
        """

        # Creating the paths
        dict_path = join(cls._data_folder, cls._dict_file)
        corpus_path = join(cls._data_folder, cls._corpus_file)
        lda_index_path = join(cls._data_folder, cls._lda_folder, cls._lda_index_file)

        # Check the Dictionary file
        if not isfile(dict_path):
            return False

        # Check the Corpus Bag-of-Words file
        if not isfile(corpus_path):
            return False

        # Check the Index of the LDA Model
        if not isfile(lda_index_path):
            return False

        # If all the files are ready, then:
        return True

    @classmethod
    def saved_topic_manager(cls):
        """
        Create a TopicManager from the information saved from a previous
        TopicManager
        :return: A TopicManager
        """
        # Create the TopicManager from the saved files and return it.
        return cls(None, _use_saved=True)
