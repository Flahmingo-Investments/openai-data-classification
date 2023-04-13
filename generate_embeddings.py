import pandas as pd
import logging
from openai.embeddings_utils import get_embedding
from training_data import data

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8191  # the maximum for text-embedding-ada-002 is 8191

def generate_embeddings():
    '''
    This function feeds the training data PII types to open ai's embeddings endpoint
    and returns embeddings of length 8191 using the ada model
    '''

    # convert training data into a dataframe
    training_dataframe = pd.DataFrame(data,columns=["pii","Label"])

    logging.info("Generating embeddings for training data ...")
    # apply embeddings transformation to the PII
    training_dataframe["embeddings"]= training_dataframe.pii.apply(lambda x : get_embedding(x,model))

    logging.info("Saving file in current directory ...")
    # save embeddings in a csv so that they can be re-used
    training_dataframe.to_csv("pii_embeddings.csv")

    return True