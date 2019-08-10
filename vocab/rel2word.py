# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 16:40:46 2018

@author: LN
"""

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import progressbar
import re
list_stopWords=list(set(stopwords.words('english')))
english_punctuations = ["'s",'-','',',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','``',"''",'/'] 
stop_words=list_stopWords+english_punctuations
stop_words.remove('won')

def getWordVocab():
    wordDict=[]
    with open('word.glove100k.txt','r',encoding='utf8') as file:
        files = file.readlines()
        for _,line in enumerate(files):
            wordDict.append(line[:-1])
    return wordDict

wordDict=getWordVocab()

#     get all synonyms of a word from the wordnet database
def getSynonyms(words):##同义词
#         include the original word
    synonyms = []
    names = set(words)
#    hyponyms = []
    deriv = []
    for word in words:
        for syn in wordnet.synsets(word):#同义词
            for lemma in syn.lemmas():
                if lemma.name() != word:
    #                     since wordnet lemma.name will include _ for spaces, we'll replace these with spaces
                    lemma_count = lemma.count()                
                    w, n = re.subn("_", " ", lemma.name())
                    flag = 1
                    for _,sub in enumerate(w.split(' ')):
                        if sub not in wordDict:
                            flag=0
                            break
                    if flag == 1:
                        if w not in names: #and lemma_count != 0:
                            synonyms.append((w,lemma_count))#同义词
                            names.add(w)             
                related_list=lemma.derivationally_related_forms()###变形
                for _,related in enumerate(related_list):
                    flag = 1
                    related_name, n = re.subn("_", " ", related.name())
                    related_count = related.count()
                    for _,sub in enumerate(related_name.split(' ')):
                        if sub not in wordDict:
                            flag=0
                            break
                    if flag == 1:
                        if related_name not in names and related_count != 0 :
                            deriv.append((related_name,related_count))
                            names.add(related_name)

 
                        
#        hyps=syn.hyponyms()###下位词
#        #hypers=syn.hypernyms()###上位词
#        for hyp in hyps:
#            hyp_count = 0
#            for hyp_lemma in hyp.lemmas():
#                hyp_count = hyp_count + hyp_lemma.count()
#            hyp_name = hyp.name().split('.')[0]
#            hyp_name, n = re.subn("_", " ", hyp_name)
#            flag = 1
#            for _,sub in enumerate(hyp_name.split(' ')):
#                if sub not in wordDict:
#                    flag=0
#                    break
#            if flag == 1:
#                if hyp_name not in names and hyp_name != word and hyp_count != 0 :
#                    hyponyms.append((hyp_name,hyp_count))
#                    names.append(hyp_name)
                

#    synonyms.sort(key=lambda data:data[-1],reverse=True)
#    hyponyms.sort(key=lambda data:data[-1],reverse=True)
#    deriv.sort(key=lambda data:data[-1],reverse=True)
    res=synonyms+deriv
    res.sort(key=lambda data:data[1],reverse=True)
    res=[x[0] for x in res]
#    print(synonyms+hyponyms+deriv)
#    return synonyms+hyponyms+deriv
#    print(res)
    return res
#getSynonyms(['release_type'])
def rel2word():
    relTable=[]
    relString=[]
    relWordVocab = set()
    with open('FB5M.rel.txt','r',encoding='utf8') as rels :
        lines = rels.readlines()
        for i,rel in enumerate(lines):
            rel_token = rel.strip().split('.')[-1].split('_')
#            for j,_ in enumerate(rel_token):
#                if rel_token[j] in stop_words:
#                    del rel_token[j]
#            if len(rel_token)==1 :
#                append_token = rel.split('.')[-2].split('_')
#                for j,_ in enumerate(append_token):
#                    if append_token[j] in stop_words or append_token[j] in rel_token:
#                        del append_token[j]
#                rel_token = append_token + rel_token
            relTable.append(rel_token)
            relString.append(' '.join(rel_token))
            relWordVocab=relWordVocab.union(set(rel_token))
    return relTable,relString,len(relTable),list(relWordVocab)

def create_vocab(corpus):
    _corpus = corpus.split('\n')
    #print(corpus)
    text = ' . '.join(_corpus)
    text = text + ' .'
    #print(text)
    tokens = word_tokenize(text)
    #print(tokens)
    tokens = set(tokens)
    tokens = list(tokens)
    #print(tokens)
    return tokens
def callable_analyzer():
    return lambda doc:create_vocab(doc)

def tf_idf(relTable,relString,wordnum):
    vitalWordList = []
    corpus = relString
#    vectorizer = TfidfVectorizer(analyzer=callable_analyzer(),strip_accents=None,stop_words=english_punctuations)
    vectorizer = TfidfVectorizer(strip_accents=None,stop_words=english_punctuations)
    mat_tfidf = vectorizer.fit_transform(corpus).toarray()
    vocab = vectorizer.get_feature_names()
    vocab=np.array(vocab)
    for idx,rel in enumerate(relTable):              
        mat_scores = mat_tfidf[idx]
        mat_idx = (-mat_scores).argsort()
        ind = 0
        if mat_scores[mat_idx[ind]]!=0:
            while(mat_scores[mat_idx[ind+1]]==mat_scores[mat_idx[ind]]):
                ind+=1
            extra_ind = list(map(int,mat_idx[0:ind+1].tolist()))
            vitalWordList.append((vocab[extra_ind]).tolist())
        else:
            vitalWordList.append(rel[0])
    return vitalWordList,len(vitalWordList),mat_tfidf,vocab
    
relTable,relString,ls,relWordVocab=rel2word()
print(len(relWordVocab))
relSynTable = []
vitalWordList,ls2,mat_tfidf,vocab=tf_idf(relTable,relString,9)
max_len_lmt = 9
with progressbar.ProgressBar(max_value=ls) as progress:
    for idx,vitalWords in enumerate(vitalWordList):
        relSynTable.append([])
        synonyms = getSynonyms(vitalWords)
        if synonyms:
            currentLen = len(relSynTable[idx])
            start = 0
            while currentLen < max_len_lmt and start<len(synonyms):
                appending_syn = synonyms[start].split(' ')
                relSynTable[idx].extend(appending_syn)
                start+=1
                if len(relSynTable[idx])>max_len_lmt:
                    del relSynTable[idx][max_len_lmt:]
                currentLen = len(relSynTable[idx])
    
        progress.update(idx)

for _,append in enumerate(relSynTable):
    relWordVocab.extend(append)
relWordVocab=list(set(relWordVocab))
with open('word.relation.txt','w',encoding = 'utf8') as file :
    for _,content in enumerate(relWordVocab):
        file.write(content+'\n')
    
with open('rel_syn.txt','w',encoding = 'utf8') as file:
    for _,content in enumerate(relSynTable):
        file.write('\t'.join(content)+'\n')
#    
#    
    
    
    
    
    
    
    
    
    
    
    
    