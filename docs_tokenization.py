# Gelin Eguinosa Rosique

import spacy

from pprint import pprint
from docs_stream import docs_stream


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


docs = docs_stream()
tokens_docs = docs_tokenization(docs)

doc_id = 1
for tokens in tokens_docs:
    print(f"\nPrinting the tokens of the Document #{doc_id}:")
    pprint(tokens[:50], width=120, compact=True)
    doc_id += 1
