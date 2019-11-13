from data import *


def precision(pairs, mapped_vectors, topn=1):
    """
    :args:
        pairs = list of right word pairs [(uk_word_0, ru_word_0), ...]
        mapped_vectors = list of embeddings after mapping from source embedding space to destination embedding space
        topn = the number of nearest neighbours in destination embedding space to choose from
    :returns:
        precision_val, float number, total number of words for those we can find right translation at top K.
    """
    assert len(pairs) == len(mapped_vectors)
    num_matches = 0
    for i, (_, ru) in enumerate(pairs):
        # YOUR CODE HERE
        ru_similar_words = ru_emb.most_similar([mapped_vectors[i]], topn=topn)
        for j in range(topn):
            if ru not in ru_similar_words[j]:
                continue
            num_matches = num_matches + 1
            break
    precision_val = num_matches / len(pairs)
    return precision_val


# Making it better (orthogonal Procrustean problem)
def learn_transform(X_train, Y_train):
    """
    :returns: W* : float matrix[emb_dim x emb_dim] as defined in formulae above
    """
    # YOU CODE HERE
    a = np.matmul(X_train.transpose(), Y_train)
    U, sigma, VT = np.linalg.svd(a, full_matrices=False)
    W = np.matmul(U, VT)
    return W


W = learn_transform(X_train, Y_train)

ru_emb.most_similar([np.matmul(uk_emb["серпень"], W)])
assert precision(uk_ru_test, np.matmul(X_test, W)) >= 0.653
assert precision(uk_ru_test, np.matmul(X_test, W), 5) >= 0.824
