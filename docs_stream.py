# Gelin Eguinosa Rosique

from pprint import pprint
from os import listdir
from os.path import isfile, join


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
