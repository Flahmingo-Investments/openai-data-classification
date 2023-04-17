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
    training_dataframe = pd.DataFrame(data,columns=["pii","Label"])

    logging.info("Generating embeddings for training data ...")

    # loop through embeddings and calculate similarity scores
    scores = [(embeddings["text"].values[i], embeddings["embedding"].values[i],
               cosine_similarity(np.array(embeddings["embedding"].values[i]).reshape(1, -1), query_embedding)[0][0]) for
              i, _ in enumerate(embeddings["embedding"])]

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

    for i in range(len(embeddings["embedding"])):
        # get pre-computed pii embedding
        pii_embedding = embeddings.iloc[i ,2]

        # reshape pii embedding
        pii_embedding = np.array(pii_embedding).reshape(1 ,-1)

        # calculate similarity score
        similarity_score_array =cosine_similarity(pii_embedding,query_embedding)
        # extract similarity score from the array
        similarity_score = similarity_score_array[0][0]
        # append tuple containing the original text, label and similarity score to the list
        scores.append((embeddings.iloc[i ,1] ,embeddings.iloc[i ,2] ,similarity_score))

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