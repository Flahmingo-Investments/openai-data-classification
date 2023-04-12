import pandas as pd
from openai.embeddings_utils import get_embedding
from training_data import data

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8191  # the maximum for text-embedding-ada-002 is 8191

# convert training data into a dataframe
training_dataframe = pd.DataFrame(data,columns=["pii","Label"])

# apply embeddings transformation to the PII
training_dataframe["embeddings"]= training_dataframe.pii.apply(lambda x : get_embedding(x,model))

# save embeddings in a csv so that they can be re-used
training_dataframe.to_csv("pii_embeddings.csv")