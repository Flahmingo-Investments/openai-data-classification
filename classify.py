import openai
import os
import pandas as pd

# Fetch API key from env variable
openai.api_key = os.environ["OPENAI_API_KEY"]

# Load file path environment variable
embeddings_file_path = os.environ(PATH_TO_EMBEDDINGS)

# embedding model parameters
embedding_model = "text-embedding-ada-002"

try:
    # load embeddings
    embeddings = pd.read_csv(embeddings_file_path)
except FileNotFoundError:
    logger.warning("Embeddings file not found. Attempting to generate embeddings ...")

def empty_dataframe():
    '''
    Return an empty dataframe with the given columns
    '''
    columns = ["pii","Label"]
    dataframe = pd.DataFrame(data,columns=columns)

    return dataframe

def generate_embeddings():
    '''
    This function feeds the training data PII types to open ai's embeddings endpoint
    and returns embeddings of length 8191 using the ada model
    '''

    # convert training data into a dataframe
    training_dataframe = empty_dataframe()

    logging.info("Generating embeddings for training data ...")



    # apply embeddings transformation to the PII
    training_dataframe["embeddings"]= training_dataframe.pii.apply(lambda x : get_embedding(x,embedding_model))

    logging.info("Saving file in current directory ...")
    # save embeddings in a csv so that they can be re-used
    training_dataframe.to_csv("pii_embeddings.csv")

    return True


def best_match(query_embedding):
    # define empty list for scores
    scores =[]

    # reshape query embedding
    query_embedding = np.array(query_embedding).reshape(1 ,-1)

    for i in enumerate(embeddings["embedding"]):

        # loop through embeddings to calculate similarity scores
        scores = [(embeddings["pii"].values[i], embeddings["label"].values[i],
                   cosine_similarity(np.array(embeddings["embedding"].values[i]).reshape(1, -1), query_embedding))
                  for
                  i, _ in enumerate(embeddings["embedding"])]

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