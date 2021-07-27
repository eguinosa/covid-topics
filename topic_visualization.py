# Gelin Eguinosa Rosique

import pyLDAvis
from pyLDAvis import gensim_models


def topic_visualization(lda_model, corpus, dictionary):
    """
    Uses the pyLDAvis package to visualize the topics in a LDA Model on a
    Jupyter Notebook.
    :param lda_model: A LDA Topic Model
    :param corpus: The Corpus in the Bag-of-Words form
    :param dictionary: The dictionary of the Corpus
    """
    visual_data = gensim_models.prepare(topic_model=lda_model, corpus=corpus, dictionary=dictionary)
    pyLDAvis.enable_notebook()
    pyLDAvis.display(visual_data)
