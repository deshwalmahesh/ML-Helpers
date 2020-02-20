import pandas as pd
import numpy as np
import operator 
import re


# apart from these things, use the correction if spellings to improve your model


contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", 
"could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", 
"hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", 
"how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", 
"I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", 
"i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", 
"it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", 
"let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not",
"mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
"needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", 
"oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", 
"she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
 "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
 "so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", 
 "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",
 "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", 
 "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", 
 "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", 
 "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  
 "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", 
 "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", 
 "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", 
 "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", 
 "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are",
 "y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
 "you'll've": "you will have", "you're": "you are", "you've": "you have" }


special = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'


 special_dict = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", 
 "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 
 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

# replace the contractions as well as the punctations and special characters in text first. While using these, make
# you have a 'space' between the words and replaced chars like in given function below


def replace_special(text,special,special_dict)
'''
A function to replace the specal characters in a text. You can use it as >>
df['col'].apply(lambda x:replace_special(x,special,special_dict)) to use in a df or series
param:
    text: list of tokens or a string
    special: a list or string of special characters
    special_dict: a dict mapping the special characters to the one which we want to replace with
out:
    text: modified text
'''
    for s in special:
        text = text.replace(s, f' {special_dict[s]} ')
    return text


# Below are the path to pre-built to encoding files built using huge data. Download these from their respective website 
glove = '../Data/embeddings/glove.840B.300d/glove.840B.300d.txt'
paragram =  '../Data/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
wiki_news = '../Data/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'


def get_vocab(texts_series):
    '''
    simple custom function to build a vocablary from given texts in series. You can use NLTK,keras' Tokenizer and 
    scikit-learn's CountVecterizor or any other library also to get vocablaries
    param:
        texts_series: a series of texts
    out:
        vocab: a dictonary with wordsas keys and respective frequencies as values
        '''

    all_sentences = texts_series.apply(lambda x: x.split()).values
    vocab = {} # dict that will be returned
    for sentence in all_sentences:
        for word in sentence: # get every word from every sentence
            # could use if vocab[word] in vocab then +=1 else 1 
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab
    

def get_coefs(word,*arr): 
    '''
    Function to get the embeddings from an embedding file. Same as the above defined function
    param:
        word: word whose embedding we have to find
        *arr: an array of unknown length
    output:
        an array of size and values equal to *arr
        '''
    return word, np.asarray(arr, dtype='float32')


def extract_embeddings(file):
    '''
    Function to extract the embeddings from the files. There are different embeddings available trained on different data. 
    Function coded for 300-D embedding files
    param:
        file: path of the file which contains the embeddings
    out:
        embeddings_index: a dict of word,embbeding pair with words as the keys
    '''
    
    if file[-3::] == 'vec': # for 300-D .vec file
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
        
    return embeddings_index


def get_vocab_embed_diff(vocab, embeddings_index):
    '''
    A function to check and return the difference between the vocabulary and embeddings. In simple words, it just checks 
    what % of vocab and total text is present in embeddings
    param:
        vocab: a vocab in form of dict where keys epresents the words and values frequency
        embedding_index: an embedding dict just like vocab
    out:
        unknown_words: dict of all the unknown words
    '''
    known_words = {} # that are in embeddings
    unknown_words = {} # which are not
    known_words_counts = 0
    unknown_words_counts = 0
    # total count of known and unknown makes our whole text

    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            known_words_counts += vocab[word]
            # for every word that is present in both vocab and known to embedding, put it in known_words and in the 
            # known_word_counts add the number of times that word has occured in text, i.e frequency of that word
        except:
            unknown_words[word] = vocab[word]
            unknown_words_counts += vocab[word]
            pass

    print('{:.2%} of total vocab is present in embeddings'.format(len(known_words) / len(vocab)))
    print('Embeddings covers {:.2%} of all text'.format(known_words_counts/(known_words_counts+unknown_words_counts)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words


def lower_embedding_augmentation(embedding, vocab):
    '''
    As some embeddings have lower case handled and some don't, then in case they don't it'll treat same words differently
    because of the capitalization or upper case presentation. So we are adding some extra to our existing embeddings,
    param:
        embedding: embedding of a specific model extracted using extract_embedding
        vocab: vocab of words
    out:
        None. It just changes the existing embedding
    '''
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:   # you can also check for capitalize
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")



