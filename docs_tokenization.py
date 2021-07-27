# Gelin Eguinosa Rosique

import spacy


def docs_tokenization(documents):
    """
    Receive the sequence of the texts of all the documents in the corpus, and
    transform each text into an array of tokens.
    Removes all the stop words, punctuation symbols and numbers in the
    documents, lowercases the text and lemmatizes each token.
    :param documents: A sequence containing all the texts of the documents.
    :return: A list of tokens for each document.
    """

    # Loading the English Package
    nlp = spacy.load('en_core_web_sm')

    # Iterate through the text of the documents and return their tokens
    for text in documents:
        text_doc = nlp(text)
        text_tokens = [token.lemma_.lower().strip()
                       for token in text_doc
                       if token.is_alpha and not token.is_stop]
        yield text_tokens
