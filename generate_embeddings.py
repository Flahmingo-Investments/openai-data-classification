import pandas as pd
from openai.embeddings_utils import get_embedding
from training_data import data

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8191  # the maximum for text-embedding-ada-002 is 8191

# convert training data into a dataframe
training_dataframe = pd.DataFrame(data,columns=["PII","Label"])


training_dataframe["embeddings"]= training_dataframe.PII.apply(lambda x : get_embedding(x,model))