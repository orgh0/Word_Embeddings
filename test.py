from __future__ import print_function
from gensim.models import KeyedVectors


find_similar_to = 'dog'
positive_instance = ['king', 'queen']
negative_instance = ['male']

print("Model loading started")
en_model = KeyedVectors.load_word2vec_format('../wiki.en/wiki.en.vec')
print("Model loading ended")

words = []
for word in en_model.vocab:
    words.append(word)


for similar_word in en_model.similar_by_word(find_similar_to):
    print("Word: {0}, Similarity: {1:.2f}".format(similar_word[0], similar_word[1]))


for resultant_word in en_model.most_similar(positive=positive_instance, negative=negative_instance, topn=1):
    print("Word : {0} , Similarity: {1:.2f}".format(resultant_word[0], resultant_word[1]))

