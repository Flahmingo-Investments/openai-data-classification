import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embeddings_file_path = os.environ(PATH_TO_EMBEDDINGS)
# load embeddings
embeddings = pd.read_csv("")

def best_match(query_embedding):
    scores =[]
    for i in range(len(training_dataframe["embedding"])):

        similarity_score =cosine_similarity(np.array(training_dataframe.iloc[i ,4]).reshape(1 ,-1)
                                             ,np.array(query_embedding).reshape(1 ,-1))
        scores.append((training_dataframe.iloc[i ,0] ,training_dataframe.iloc[i ,1] ,similarity_score))

    # sort list
    sorted_scores = sorted(scores, key=lambda x: x[2], reverse=True)

    return sorted_scores[:3]
