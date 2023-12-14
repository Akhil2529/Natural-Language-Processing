
import time
import gensim
import os
from gensim.models import Word2Vec
import numpy as np
import spacy
import en_core_web_sm
from nltk import sent_tokenize,word_tokenize
import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download('punkt')

sent = []
path_ ="comments1k"
for files in os.listdir(path_):
  var1 = open(os.path.join(path_,files),'r')
  sent.append(var1.read())

nlp = spacy.load("en_core_web_sm")
def prepossessing(sent):


    # Split comments into sentences
    sentences = []
    for comment in sent:
        doc = comment
        for sente in sent_tokenize(doc):
            sentences.append(sente)



    # Do tokenization for the dataset
    tokens = []
    for comment in sent:
        doc = sent_tokenize(comment)
        for token in doc:
            tokens.append(word_tokenize(token))

    # Report the average number of tokens per comment
    num_sentences = [len(sent_tokenize(comment)) for comment in sent]
    avg_sentences_per_comment = sum(num_sentences) / len(sent)
    print("Average number of sentences per comment:", avg_sentences_per_comment)


    # Without considering punctuation, how many words are in each comment on average?
    num_words = [len([token for token in nlp(comment) if not token.is_punct]) for comment in sent]
    avg_words_per_comment = sum(num_words) / len(sent)
    print("Average number of words per comment (without punctuation):", avg_words_per_comment)

    # Text preprocessing techniques for the dataset to generate a corpus for word embeddings
    # Lowercase the comments
    lowercase_comments = [comment.lower() for comment in sent]

    # Remove stop words
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    filtered_comments = []
    for comment in lowercase_comments:
        doc = nlp(comment)
        filtered_tokens = [token.text for token in doc if not token.is_stop]
        filtered_comments.append(" ".join(filtered_tokens))

    # Lemmatize the comments
    lemmatized_comments = []
    for comment in filtered_comments:
        doc = nlp(comment)
        lemmatized_tokens = [token.lemma_ for token in doc]
        lemmatized_comments.append(" ".join(lemmatized_tokens))

    # Generate the corpus for word embeddings
    corpus = " ".join(lemmatized_comments)
    
    # Explanation:
    # Lowercasing the comments is important to reduce the dimensionality of the word embeddings.
    # Removing stop words helps to focus on the important words in the comments.
    # Lemmatizing the comments helps to reduce the dimensionality further by grouping together words that have a

    return corpus

def train_word2vec(sentences): 
    text_data = [sentence.split() for sentence in sentences]

    start = time.time()
    
    # Train the model
    model = Word2Vec(text_data, size=100, window=5, min_count=1, workers=4, sg=0)

    end = time.time()
    
    
    print("Training Time:", end-start, "seconds")

    return model

a = prepossessing(sent)

file_open = open("corpus.txt",'w')
for word_sent in sent:
  file_open.write(word_sent+"\n")
file_open.close()

train_word2vec(sent)

train = train_word2vec(sent)

train.wv.index2word

def evaluate_word2vec(train):
    words_to_evaluate = ['Christmas', 'movie', 'music', 'woman']
    for word in words_to_evaluate:
        similar_words = train.wv.most_similar(word)
        print(f"Most similar words to '{word}': {similar_words}")

evaluate_word2vec(train)

import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts



# set the vector sizes
vector_sizes = [1, 10, 100]

# train the models with different vector sizes
models = []
sentences = sent
text_data = [sentence.split() for sentence in sentences]
for vector_size in vector_sizes:
    model = Word2Vec(text_data, size=vector_size, window=5, min_count=1, workers=4, sg=0)
    models.append(model)

# evaluate the models
for i, model in enumerate(models):
    print(f"Model with vector size {vector_sizes[i]}")
    print(model.wv.most_similar("movie"))

vector_sizes

def train_Glove():
  #trained glove vectors
  file_vectors = open("vectors.txt",'r')

  vectors_for_gloves = []

  for y in file_vectors.readlines():
    vectors_for_gloves.append(y)
  glove_vectors = {}
  i=0
  for line in vectors_for_gloves:
    tokens = line.strip().split()
    word = tokens[0]
    try:
      vector = np.array(tokens[1:], dtype=np.float)
    except:
      vector = np.array(tokens[2:], dtype=np.float)
    glove_vectors[word] = vector

  # Extract the vocabulary and vectors from the glove_vectors dictionary
  vocab_g = list(glove_vectors.keys())
  vecs = list(glove_vectors.values())

  return vocab_g,vecs

import numpy as np
def evaluate_word2vec(vocab_g,vecs):
  test_cases = ['music','movie','Christmas','woman']
  # Assume that test_cases is a list of strings containing the test cases
  index_test = []
  test_array = []
  for tc in test_cases:
      # Convert the test case to lowercase
      tc = tc.lower()

      # Find the index of the test case in the vocabulary
      index = np.where(np.array(vocab_g)==tc)[0][0]

      # Add the index to the ix_tc list
      index_test.append(index)

      # Get the corresponding vector from the vecs list and add it to the tc_ar list
      vector = vecs[index]
      test_array.append(vector)
      test_array_cases = np.array(test_array).astype(float).reshape(4,100)
      vectors_array = np.array(vecs).astype(float).reshape(len(vocab_g),100)
      matrix_multiplication= np.dot(test_array_cases,vectors_array.T)

      return matrix_multiplication

def result(matrix_multiplication,vocab_g):
  matrix_words = {}
  top = 11  # number of top similar words to consider
  test_cases = ['music','movie','Christmas','woman']

  for i, word in enumerate(test_cases):
    matrix_words[word] = []
    m_i = np.argsort(matrix_multiplication[i, :])[-top:]
    for j in m_i:
        matrix_words[word].append(vocab_g[j])
  return matrix_words

matrix_words

