from better import *

def translate(sentence):
    """
        :args:
            sentence - sentence in Ukrainian (str)
        :returns:
            translation - sentence in Russian (str)

        * find ukrainian embedding for each word in sentence
        * transform ukrainian embedding vector
        * find nearest russian word and replace
        """
    # YOUR CODE HERE
    translation = ""
    for uk in sentence.split():
        if uk in "`~!-_:;',.?\"":
            ru = uk
        # else:
        #     ru, pred = ru_emb.most_similar([np.matmul(uk_emb[uk], W)], topn=1)[0]
        elif uk not in uk_emb:
            ru = "####"
        else:
            ru, pred = ru_emb.most_similar([np.matmul(uk_emb[uk], W)], topn=1)[0]
        translation = translation + ru + ' '

    return translation


f1 = open("trans_logging.txt", "w", encoding="utf-8")
f2 = open('translated.txt', 'w', encoding='utf-8')

with open("fairy_tale.txt", "r", encoding='utf-8') as inpf:
    uk_sentences = [line.rstrip().lower() for line in inpf]
    for sentence in uk_sentences:
        print("src: {}\ndst: {}\n".format(sentence, translate(sentence)))
        print("src: {}\ndst: {}\n".format(sentence, translate(sentence)), file=f1)
        print("{}".format(translate(sentence)), file=f2)

f1.close()
f2.close()
