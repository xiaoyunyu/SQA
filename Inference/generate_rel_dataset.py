# -*- coding: utf-8 -*-
# python2
############额外添加了tf-idf信息，用于生成relRNN训练数据,删除了QAdata.cand_rel列表中的正关系,只保留前32个cand_rel。
#正则表达式必知必会
import sys
import pickle
import io
import progressbar
from nltk.corpus import stopwords
from nltk import word_tokenize
import re

sys.path.append( '../src/py_module' )
from qa_data import *
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

special_char = ["*", ".", "?", "+", "$", "^", "[", "]", "(", ")", "{", "}", "|", "/"]
stopWords = ['…','°',"'","'s",'-','',',', '.',"+", ':',"^", ';', '?', '(', ')',"{", "}", "|",'[', ']', '&', '!', '*', '@', '#', '$', '%','``',"''",'/'] 
method = 'word_tokenize'

def getNumofCommonSubstr(str1, str2):  
    lstr1 = len(str1)  
    lstr2 = len(str2)  
    record = [[0 for i in range(lstr2+1)] for j in range(lstr1+1)]  # 多一位  
    maxNum = 0          # 最长匹配长度  
    p = 0               # 匹配的起始位  
  
    for i in range(lstr1):  
        for j in range(lstr2):  
            if str1[i] == str2[j]:  
                # 相同则累加  
                record[i+1][j+1] = record[i][j] + 1  
                if record[i+1][j+1] > maxNum:  
                    # 获取最大匹配长度  
                    maxNum = record[i+1][j+1]  
                    # 记录最大匹配长度的终止位置  
                    p = i + 1  
    return str1[p-maxNum:p], maxNum

def create_vocab(corpus):
    _corpus = corpus.split('\n')
    #print(corpus)
    text = ' . '.join(_corpus)
    text = text + ' .'
    #print(text)
    tokens = text.split(' ')
    #print(tokens)
    tokens = set(tokens)
    tokens = list(tokens)
    #print(tokens)
    return tokens
def callable_analyzer():
    return lambda doc:create_vocab(doc)

def gen_fb_dgree_info():
    with io.open('/srv/sfw/CFO/KnowledgeBase/VirtuosoKG/data/FB5M.core.txt','r',encoding='utf-8') as file:
        lines = file.readlines()
        fb = {}#[out-degree,in-degree,out-in degree]
        print(lines[0])
        for idx,line in enumerate(lines):
            fields = line.strip().split('\t')
            node_s = fields[0][1:-1]
            node_o = fields[2][1:-1]
            if idx ==0 :
                print(node_s)
                print(node_o)
            if node_s not in fb:
                fb[node_s] = [0,0,0]
            if node_o not in fb:
                fb[node_o] = [0,0,0]
            fb[node_s][0] = fb[node_s][0]+1
            fb[node_o][1] = fb[node_o][1]+1
        for k,v in fb.items():
            v[2]=v[0]+v[1]
    print('Finish creating dict')
    pickle.dump(fb,open('FB5M.ent.degree','wb'))

def degree(datas,negsize,datasize,fb,criterion,suffix):
    if criterion=='out':
        crt = 0
    elif criterion=='in':
        crt = 1
    elif criterion=='out-in':
        crt = 2    

    top=np.array([0 for i in range(7)])
    top_ind=[1,5,10,20,50,100,400]
    count = 0
    emp_cand = 0
    total = len(datas)
    global method
    new_datalist = []
    max_neglen = 0
    ms_cnt = 0
    with progressbar.ProgressBar(max_value=datasize) as progress:
        for time_cnt,data in enumerate(datas):
            question = data.question
            relation = data.relation
            try:
                data.cand_rel = list(set(data.cand_rel))
            except:
                data.cand_rel = ['unknown']
            max_neglen = max(max_neglen,len(data.cand_rel))
            # subject_mentions = data.subject_mentions[:]
            # subject_mentions = list(set(subject_mentions))
            try:
                candi_sub = data.cand_sub
            except:
                candi_sub = ['unknown']
            try:
                sub_rels = data.sub_rels
            except:
                sub_rels = [['unknown']]

            if 'fb:type.object.ngram' in data.cand_rel:
                data.cand_rel.remove('fb:type.object.ngram')
            if 'fb:type.object.name' in data.cand_rel:
                data.cand_rel.remove('fb:type.object.name')
            if 'fb:common.topic.alias' in data.cand_rel:
                data.cand_rel.remove('fb:common.topic.alias')
            scores = []
            temp_scores = []
            for idx,sub in enumerate(candi_sub):
                if sub == data.subject:
                    if data.relation not in sub_rels[idx]:
                        print('missing relation')
                        ms_cnt+=1
                if sub not in fb:
                    candi_sub[idx] = ''
                    sub_rels[idx] = []
                    scores.append(0)
                    temp_scores.append((sub,0))
                else:
                    scores.append(fb[sub][crt])
                    temp_scores.append((sub,fb[sub][crt]))
            data.cand_sub = [sub for sub in candi_sub]
            data.sub_rels = [sub for sub in sub_rels]
            try:
                assert(len(scores)==len(data.cand_sub))
            except:
                print(temp_scores)
                print(data.cand_sub)
                raise
            try:
                assert(len(scores)==len(data.sub_rels))
            except:
                print(temp_scores)
                print(data.cand_sub)
                print(data.sub_rels)    
                raise

            data.cand_sub_score=scores

            temp_scores.sort(key=lambda tup:tup[1],reverse=True)
            sorted_sub=[sub[0] for sub in temp_scores]
            for i,key in enumerate(top_ind):
                if data.subject in sorted_sub[:key]:
                    top[i:]+=1
                    break

            if relation not in data.cand_rel:
                count += 1
            new_datalist.append((data,len(data.anonymous_question.split(' '))))
			# del(appending_cand[32:])
			# data.cand_rel = appending_cand[:]##实现list复制
            # fo.write(  u'%s\t%s\t%s\n' % (question, relation, '\t'.join(data.cand_rel)))
            progress.update(time_cnt)

    # fo.close()
    print('not found %d'%count)
    print('missing relation %d'%ms_cnt)
    top = top/total
    print('total len:%d'%total)
    print("Under %s criterion: top k recall is "%(criterion))
    print('&'.join(list(map(str,top.tolist()))))
    print('max neg size = %d'%max_neglen)
    new_datalist.sort(key=lambda data:data[1],reverse=True)
    new_datalist=[data[0] for data in new_datalist]
    fo = io.open('relAN.final.%s.txt'%(suffix), 'w',encoding='utf8')
    for i,data in enumerate(new_datalist):
        question = data.anonymous_question
        if question:
            fo.write('%s\t%s\t%s\n' % (question, data.relation, '\t'.join(data.cand_rel)))
        else:
            fo.write('%s\t%s\t%s\n' % (data.question, data.relation, '\t'.join(data.cand_rel)))
    fo.close()
    return new_datalist
########################################tf-idf-begin#########################################
            # tokens = question.split()
            # cand_corpus = data.sub_texts[:]
            # for i,sub_text in enumerate(cand_corpus):
            #     cand_corpus[i] = list(set(sub_text))
            #     cand_corpus[i].sort(key=lambda str:len(str),reverse=False)

            # matched_tokens_corpus = []
            # mat_scores = np.zeros(len(candi_sub))
            # method = 'word_tokenize'
            # for idx in range(len(candi_sub)):
            #     # sub_str = cand_corpus[idx]+' .'
            #     for _,mention in enumerate(subject_mentions):
            #         pattern = r'(^|\s)(%s)($|\s)' % (re.escape(mention))
            #         for _,sentence in enumerate(cand_corpus[idx]):
            #             if cand_corpus[idx] ==['out of the unknown']:
            #                 sentence = ''
            #                 matched_senten=['.']
            #                 break
            #             matched_senten = re.findall(pattern,sentence)
            #             if matched_senten:
            #                 break
            #         if matched_senten:
            #             break 
            #     if len(matched_senten)==0:
            #         print('') 
            #         print(time_cnt)
            #         print(cand_corpus[idx])
            #         print(subject_mentions)
            #         raise NameError('there is not a sentence matching the mentions')
            #     corpus = [sentence+' .']
            #     matched_tokens = ''.join(matched_senten[0]).split()
            #     # matched_tokens_corpus.append(matched_tokens)

            #     vectorizer = TfidfVectorizer(analyzer=callable_analyzer(),strip_accents=None,stop_words=stopWords)
            #     mat_tfidf = vectorizer.fit_transform(corpus).toarray()
            #     vocab = vectorizer.get_feature_names()

            #     # for idx,sub in enumerate(matched_tokens):
            #     matched_idx=[]
            #     try:
            #         for _,token in enumerate(matched_tokens):
            #             matched_idx.append(vocab.index(token)) 
            #     except:               
            #         print('')
            #         print(matched_tokens)
            #         print('')
            #         print(cand_corpus[idx])
            #         print('')
            #         print(subject_mentions)
            #         raise 
            #     mat_scores[idx] = mat_tfidf[0][matched_idx].sum()

            # data.add_tfidf_score(mat_scores)####类实例是引用传递；np阵列是引用传递；所以必须用a=np.array(b)而不是a=b传递属性
########################################tf-idf-end#########################################


def main():
    QADataName = 'reduce.relQAData.label.fb2m.test.cpickle'
    fb = pickle.load(open('FB5M.ent.degree','rb'))
    print('finish loading fb-info')
    file = open(QADataName, 'rb')
    data_list = pickle.load(file)
    print ('Finish loading QAData') 
    negsize = 0
    datasize = len(data_list)
    new_datalist = degree(data_list,negsize,datasize,fb,'out','test')
    pickle.dump(new_datalist,open(QADataName, 'wb'))
    # save_file = open('relQAData.label.strict.test.cpickle', 'wb')
    # pickle.dump(data_list, save_file)
    # save_file.close()

main()


