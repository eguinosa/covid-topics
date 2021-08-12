# Gelin Eguinosa Rosique

from os import listdir
from os.path import isfile, join


class DocumentsFiles:
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


def docs_stream(dir_path='docs', file_ext=''):
    """
    This method receives the path where all the documents of the corpus are and
    the returns a list of the text of these documents.
    :param dir_path: The directory where all the documents of the corpus are
    located.
    :param file_ext: The file extension of the documents belonging to the corpus.
    :return: The texts of the documents in the corpus.
    """

    # Loading the location of all the files in the given directory
    docs_path = [file for file in listdir(dir_path)
                 if isfile(join(dir_path, file)) and file.endswith(file_ext)]

    # Create a list for the text of the documents
    docs_text = []

    # Iterate through all the files and add their text to the list.
    for doc_path in docs_path:
        document = open(join(dir_path, doc_path))
        text_document = document.read()
        document.close()
        docs_text.append(text_document)

    return docs_text
