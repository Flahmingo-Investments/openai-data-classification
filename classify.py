import openai
import os
import pandas as pd
import logging
from training_data import data
from openai.error import OpenAIError

# Fetch API key from env variable
openai.api_key = os.environ["OPENAI_API_KEY"]

# Load file path environment variable
# Note: path should be in the format directory_path + filename.csv
# Sample path : "../embeddings.csv"
embeddings_file_path = os.environ(PATH_TO_EMBEDDINGS)

# embedding model parameters
embedding_model = "text-embedding-ada-002"

try:
    # load embeddings
    embeddings = pd.read_csv(embeddings_file_path)
except FileNotFoundError:
    logging.warning("Embeddings file not found. Attempting to generate embeddings ...")

def conver_to_dataframe():
    '''
    Returns training data as dataframe empty dataframe with columns pii and label
    '''
    columns = ["pii","label"]
    dataframe = pd.DataFrame(data,columns=columns)

    return dataframe

def generate_embeddings():
    '''
    This function feeds the training data PII types to open ai's embeddings endpoint
    and returns embeddings of length 8191 using the ada model
    '''

    # convert training data into a dataframe
    training_dataframe = conver_to_dataframe(data = data)

    logging.info("Generating embeddings for training data ...")
    training_dataframe["embeddings"]= training_dataframe.pii.apply(lambda x : get_embedding(x,embedding_model))

    # save embeddings in a csv so that they can be re-used
    logging.info("Saving file in current directory ...")
    training_dataframe.to_csv(embeddings_file_path+"/"+"pii_embeddings.csv")

    return True


def best_match(query_embedding):
    # define empty list for scores
    scores =[]

    # reshape query embedding
    query_embedding = np.array(query_embedding).reshape(1 ,-1)

    try:
        for i in enumerate(embeddings["embedding"]):

            # loop through embeddings to calculate similarity scores
            # store scores as tuples in the list in the form -> (pii, label,score)
            scores = [(embeddings["pii"].values[i], embeddings["label"].values[i],
                       cosine_similarity(np.array(embeddings["embedding"].values[i]).reshape(1, -1), query_embedding))
                      for
                      i, _ in enumerate(embeddings["embedding"])]
        logger.info("Converted pii to embeddings successfully ")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return []


    # sort list
    sorted_scores = sorted(scores, key=lambda x: x[2], reverse=True)

    # return best match
    return sorted_scores[:1]


# take in a query via command line
query = input("Enter a query: ")

predict = openai.Classification.create(
    search_model="davinci",
    model="davinci",
    examples = training,
    query = query,
    labels = ["High", "Medium", "Secret", "None"],
).label.lower()

print("The classification is: " + predict)