# Gelin Eguinosa Rosique

from os import listdir
from os.path import isfile, join


class DocumentsManager:
    """Class to manage the files of the documents in the corpus."""

    def __init__(self, dir_path='docs', file_ext=''):
        """
        Receives the path where all the documents of the corpus are, and creates
        a dictionary with the names of the files and their locations.
        :param dir_path: The directory where the documents of the corpus are
        located.
        :param file_ext: The file extension of the documents belonging to the
        corpus.
        """
        # Creating dictionary where all the documents and their locations will
        # be saved.
        self.documents = {}

        # Iterate through all the documents in the given directory and save the
        # locations of the documents.
        for doc_name in listdir(dir_path):
            if isfile(join(dir_path, doc_name)) and doc_name.endswith(file_ext):
                doc_path = join(dir_path, doc_name)
                self.documents[doc_name] = doc_path

    def documents_texts(self):
        """
        Iterates through all the documents' files to send their texts.
        :return: A sequence containing the texts all the documents in the
        corpus.
        """
        for file_path in self.documents.values():
            document = open(file_path)
            text_document = document.read()
            document.close()
            yield text_document
