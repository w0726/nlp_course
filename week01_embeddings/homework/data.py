# -*- coding:utf-8 -*-
import gensim
import numpy as np
from gensim.models import KeyedVectors
from sklearn.linear_model import LinearRegression

# load embeddings for ukrainian and russian
uk_emb = KeyedVectors.load_word2vec_format("cc.uk.300.vec")
ru_emb = KeyedVectors.load_word2vec_format("cc.ru.300.vec")
# result1 = ru_emb.most_similar([ru_emb["август"]], topn=5)
# print("result1:", result1)
# uk_emb.most_similar([uk_emb["серпень"]])
# result2 = ru_emb.most_similar([uk_emb["серпень"]])
# print("result2:\n", result2)
# load small dictionaries for corresponding words pairs as trainset and testset


# open test/train.txt with encoding='utf-8' !!!
def load_word_pairs(filename):
    uk_ru_pairs = []
    uk_vectors = []
    ru_vectors = []
    with open(filename, "r", encoding='utf-8') as inpf:
        for line in inpf:
            uk, ru = line.rstrip().split("\t")
            if uk not in uk_emb or ru not in ru_emb:
                continue
            # print((uk, ru))
            uk_ru_pairs.append((uk, ru))
            uk_vectors.append(uk_emb[uk])
            ru_vectors.append(ru_emb[ru])
    return uk_ru_pairs, np.array(uk_vectors), np.array(ru_vectors)


uk_ru_train, X_train, Y_train = load_word_pairs("ukr_rus.train.txt")
uk_ru_test, X_test, Y_test = load_word_pairs("ukr_rus.test.txt")

print("X_train.shape={}\n Y_train.shape={}\n X_test.shape={}\n Y_test.shape={}".format(X_train.shape,
                                                                                       Y_train.shape,
                                                                                       X_test.shape,
                                                                                       Y_test.shape))

# # embedding_space_mapping.py
# # from sklearn.linear_model import LinearRegression
# # MY CODE HERE
# mapping = LinearRegression(fit_intercept=False).fit(X_train, Y_train)
# print(mapping)
# print('\n{}'.format(mapping.coef_))


