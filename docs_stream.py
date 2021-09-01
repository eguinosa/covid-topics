# Gelin Eguinosa Rosique

from os import listdir
from os.path import isdir, isfile, join


class DocumentsManager:
    """
    Manage the documents files of the corpus. Upon initialization creates an
    index with all the documents in the corpus, and later when requested,
    returns the documents one by one, to save memory resources.
    """
    # The location of the documents
    docs_folder = 'docs'
    # The file extension of the documents
    docs_suffix = '.txt'

    def __init__(self):
        """
        Receives the path where all the documents of the corpus are, and creates
        a dictionary with the names of the files and their locations.
        """
        # Check if the docs directory exists:
        if not isdir(self.docs_folder):
            raise Exception(f"The folder '{self.docs_folder}', where the"
                            f" documents are supposed to be located, does not"
                            f" exist.")

        # Creating dictionary where all the documents and their locations will
        # be saved.
        self.documents = {}

        # Iterate through all the documents in the given directory and save the
        # locations of the documents.
        for doc_name in listdir(self.docs_folder):
            doc_path = join(self.docs_folder, doc_name)
            if isfile(doc_path) and doc_name.endswith(self.docs_suffix):
                self.documents[doc_name] = doc_path

    def documents_texts(self):
        """
        Iterates through all the documents' files to send their texts.
        :return: A sequence containing the texts all the documents in the
        corpus.
        """
        # Iterate through the locations of the documents saved in the index.
        for file_path in self.documents.values():
            # Load the content of the document and return them one by one.
            with open(file_path, 'r', encoding='utf-8') as file:
                text_document = file.read()
            yield text_document

    @classmethod
    def documents_available(cls):
        """
        Checks if the class has any documents to work with in the designated
        folder.
        :return: A bool representing if there are documents available to work
        with or not.
        """
        # Check if the docs directory exists.
        if not isdir(cls.docs_folder):
            return False

        # Check if there is any document inside docs folder.
        for doc_name in listdir(cls.docs_folder):
            doc_path = join(cls.docs_folder, doc_name)
            if isfile(doc_path) and doc_name.endswith(cls.docs_suffix):
                # There is at least one document available
                return True

        # Return false, if we couldn't find any document in the folder
        return False
