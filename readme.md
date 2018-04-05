# Word Embeddings

## What are word embeddings ?

Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with much lower dimension.

## About

This repository contains scripts for playing around with the common pretrained work embedding vectors like `Fasttext` , `Glove`


## Gensim Library

A library called `Gensim` is used for the following analysis.
Gensim is a robust open-source vector space modeling and topic modeling toolkit implemented in Python. It uses NumPy, SciPy and optionally Cython for performance. Gensim is specifically designed to handle large text collections, using data streaming and efficient incremental algorithms, which differentiates it from most other scientific software packages that only target batch and in-memory processing.

## Code Explanation

Loads the Model:
```python

en_model = KeyedVectors.load_word2vec_format('../wiki.en/wiki.en.vec')

```

Finds similar words:
```python
for similar_word in en_model.similar_by_word(find_similar_to):
    print("Word: {0}, Similarity: {1:.2f}".format(similar_word[0], similar_word[1]))
```

Finds analogous words:
```python
for resultant_word in en_model.most_similar(positive=positive_instance, negative=negative_instance, topn=1):
    print("Word : {0} , Similarity: {1:.2f}".format(resultant_word[0], resultant_word[1]))
```