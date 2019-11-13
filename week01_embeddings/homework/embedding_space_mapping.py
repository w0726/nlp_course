# -*- coding:utf-8 -*-
from data import *
from sklearn.linear_model import LinearRegression

# MY CODE HERE
mapping = LinearRegression(fit_intercept=False).fit(X_train, Y_train)
print(mapping)
print('\n{}'.format(mapping.coef_))

#####################################################################
# Let's take a look at neigbours of the vector of word "серпень" ("август" in Russian) after linear transform.
august = mapping.predict(uk_emb["серпень"].reshape(1, -1))
ru_emb.most_similar(august)
'''
[('апрель', 0.8541285395622253), ('июнь', 0.841120183467865), ('март', 0.8396993279457092), ('сентябрь', 0.8359869122505188), ('февраль', 0.8329297304153442), ('октябрь', 0.8311846256256104), ('ноябрь', 0.8278923034667969), ('июль', 0.8234528303146362), ('август', 0.8120501041412354), ('декабрь', 0.8039003610610962)]
（“ 4月'，0.8541285395622253），
（“ 6月”，0.841120183467865），
 (“ 3月”，0.8396993279457092），
（“ 9月”，0.8359869122505188），
（“ 2月”，0.8329297304153442），
（“10月”，0.8311846256256104），
（“11月”，0.8278923034667969），
（“ 7月”，0.8234528303146362），
（“ 8月”，0.8120501041412354），
（“ 12月”，0.8039003610610962）
'''


# As quality measure we will use precision top-1, top-5 and top-10
# 对于每个经过转换的乌克兰embedding，我们计算在俄罗斯embeddings的前N个最近邻居中发现了多少对正确的目标对
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


assert precision([("серпень", "август")], august, topn=5) == 0.0
assert precision([("серпень", "август")], august, topn=9) == 1.0
assert precision([("серпень", "август")], august, topn=10) == 1.0

assert precision(uk_ru_test, X_test) == 0.0
assert precision(uk_ru_test, Y_test) == 1.0

precision_top1 = precision(uk_ru_test, mapping.predict(X_test), 1)
precision_top5 = precision(uk_ru_test, mapping.predict(X_test), 5)

assert precision_top1 >= 0.635
assert precision_top5 >= 0.813

