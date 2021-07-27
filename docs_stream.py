# Gelin Eguinosa Rosique

from pprint import pprint
from os import listdir
from os.path import isfile, join


def documents_stream(dir_path='docs', file_ext=''):
    """
    This method receives the path where all the documents of the corpus are and
    the returns a sequence of the text of these documents so you can iterate
    through them without loading all the documents in memory.
    :param dir_path: The directory where all the documents of the corpus are
    located.
    :param file_ext: The file extension of the documents belonging to the corpus.
    :return: The sequence of documents in the corpus.
    """

    # Loading the location of all the files in the given directory
    docs_path = [file for file in listdir(dir_path)
                 if isfile(join(dir_path, file)) and file.endswith(file_ext)]

    # Iterate through all the files and return their text.
    for doc_path in docs_path:
        document = open(join(dir_path, doc_path))
        text_document = document.read()
        document.close()
        yield text_document
