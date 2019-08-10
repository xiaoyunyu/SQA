# -*- coding: utf-8 -*-
import os, sys, re,time
import pickle
import numpy as np
import matplotlib  
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk import word_tokenize

sys.path.append( '../src/py_module' )
import qa_data
from qa_data import *
import virtuoso 

fb='fb2m:'

type_dict = None
save_all = False
list_stopWords=list(set(stopwords.words('english')))
english_punctuations = ["'s",'-','',',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','``',"''",'/'] 
stop_words=list_stopWords+english_punctuations
# def generate_ngrams(tokens, min_len, max_len):
#     ngrams = []
#     num_token = len(tokens)
#     assert(num_token >= max_len)
#     for num in range(min_len, max_len+1):
#         for i in range(num_token-num+1):
#             ngram = ' '.join(tokens[i:i+num])
#             if not ngram in stop_words:
#                 ngrams.append(ngram)
#     return list(set(ngrams))
#     
def generate_ngrams(tokens,num):
    n_grams=[]
    num_token = len(tokens)
    if num_token < num :
        return []
    for i in range(num_token-num+1):
        ngram = ' '.join(tokens[i:i+num])
        if num == 1:
            if not ngram in stop_words:
                n_grams.append(ngram)
        else :
            n_grams.append(ngram)
    return n_grams


def beg_end_indices(scores, threshold):
    seq_len = len(scores)
    beg_idx = 0
    end_idx = 0
    flag_new = True
    bi=0
    ei=0
    for i in range(seq_len):
        if scores[i]>threshold :
            if flag_new:
                beg_idx=i
                flag_new=False
            if i==seq_len-1:
                end_idx=i 
                if end_idx-beg_idx>=ei-bi:
                    ei=end_idx
                    bi=beg_idx
        elif scores[i]<threshold:
            if i>0 and i<seq_len-1 and scores[i-1]>threshold and scores[i+1]>threshold:
                continue
            else:
                if not flag_new:
                    end_idx=i-1
                    if end_idx-beg_idx>=ei-bi:
                        ei=end_idx
                        bi=beg_idx
                    flag_new=True

           
    return bi,ei

def form_anonymous_quesion(question, beg_idx, end_idx):
    anonymous_tokens = []
    tokens = question.split()
    anonymous_tokens.extend(tokens[:beg_idx])
    anonymous_tokens.append('X')
    anonymous_tokens.extend(tokens[end_idx+1:])
    anonymous_question = ' '.join(anonymous_tokens)

    return anonymous_question

def query_candidate(data_list, pred_list,logName):
    log_file = open(logName, 'w')
    new_data_list = []

    NoneMatch = 0
    succ_match = 0
    data_index = 0
    for pred, data in zip(pred_list, data_list):
        # extract scores
        scores = np.array([int(float(score)) for score in pred.decode().strip().split()])

        # extract fields needed
        relation = data.relation
        subject  = data.subject
        question = data.question
        text_attention_indices = data.text_attention_indices
        if not text_attention_indices:
            continue
        # incremnt data_index
        data_index += 1            
        # print([question])
        tokens   = np.array(question.split())

        # query name / alias by subject (id)
        candi_sub_list = []
        # for threshold in np.arange(0.5, 0.0, -0.095):
        #     beg_idx, end_idx = beg_end_indices(scores, threshold)
        #     sub_text = ' '.join(tokens[beg_idx:end_idx+1])
        #     candi_sub_list.extend(virtuoso.str_query_id(sub_text))
        #     if len(candi_sub_list) > 0:
        #         break

        beg_idx, end_idx = beg_end_indices(scores, 0.2)
        tokens_crop = tokens[beg_idx:end_idx+1]
        sub_text = ' '.join(tokens_crop)
        text_list=[]
        # for i in [1,-1,2,-2]:
        #     if beg_idx-i>=0 and beg_idx-i<=end_idx:
        #         tokens_crop2=tokens[beg_idx-i:end_idx+1]
        #         text_list.append(' '.join(tokens_crop2))
        #     if end_idx+i<seq_len and end_idx+i>=beg_idx:
        #         tokens_crop2=tokens[beg_idx:end_idx+i+1]
        #         text_list.append(' '.join(tokens_crop2))
        #         if beg_idx-i>=0 and beg_idx-i<=end_idx+i:
        #             tokens_crop2=tokens[beg_idx-i:end_idx+i+1]
        #             text_list.append(' '.join(tokens_crop2))
        candi_sub_list.extend(virtuoso.str_query_id(sub_text))
        if '' in candi_sub_list:
            candi_sub_list.remove('')

            # if candi_sub_list==[]:
            #     for text in text_list:
            #         candi_sub_list.extend(virtuoso.str_query_id(text))
            #         if '' in candi_sub_list:
            #             candi_sub_list.remove('')
            #         if candi_sub_list!=[]:
            #             break
        if candi_sub_list:
            data.set_strict_flag(True)
            pass
        else:
            data.set_strict_flag(False)
            candi_sub_list.extend(virtuoso.partial_str_query_id(sub_text))
            if '' in candi_sub_list:
                candi_sub_list.remove('')
        if not candi_sub_list:
            for i in range(len(tokens_crop)-1,1,-1):
                tempList = generate_ngrams(tokens_crop,i)
                for x,y in enumerate(tempList):
                    idList = virtuoso.str_query_id(y)
                    if '' in idList:
                        idList.remove('')
                    candi_sub_list.extend(idList)
                if candi_sub_list:
                    break

        candi_sub_list=list(set(candi_sub_list))##[string,string...]
        # if '' in candi_sub_list:
        #     candi_sub_list.remove('')
       # using freebase suggest
        # if len(candi_sub_list) == 0:
        #     beg_idx, end_idx = beg_end_indices(scores, 0.2)
        #     sub_text = ' '.join(tokens[beg_idx:end_idx+1])
        #     sub_text = re.sub(r'\s(\w+)\s(n?\'[tsd])\s', r' \1\2 ', sub_text)
        #     suggest_subs = []
        #     for trial in range(3):
        #         try:
        #             suggest_subs = freebase.suggest_id(sub_text)
        #             print >> log_file, str(suggest_subs)
        #             break
        #         except:
        #             print >> sys.stderr, 'freebase suggest_id error: trial = %d, sub_text = %s' % (trial, sub_text)
        #     candi_sub_list.extend(suggest_subs)
            # if subject not in candi_sub_list:
            #     print >> log_file, '%s' % (str(question))

        # if potential subject founded
        if len(candi_sub_list) > 0:
            # add candidates to data
            countarry = np.zeros(len(candi_sub_list))
            for idx,candi_sub in enumerate(candi_sub_list):
                candi_rel_list = virtuoso.id_query_out_rel(fb,candi_sub)
                candi_rel_list = list(set(candi_rel_list))
                if '' in candi_rel_list:
                    candi_rel_list.remove('')
                if len(candi_rel_list) > 0:
                    if type_dict:
                        candi_type_list = [type_dict[t] for t in virtuoso.id_query_type(candi_sub) if type_dict.has_key(t)]
                        if len(candi_type_list) == 0:
                            candi_type_list.append(len(type_dict))
                        data.add_candidate(candi_sub, candi_rel_list, candi_type_list)
                    else:
                    	data.add_candidate(candi_sub, candi_rel_list)

                        # countarry[idx] = virtuoso.id_query_count(candi_sub)
                        # if '' in text:
                        #     text.remove('')
                        # if len(text) > 0:
                        #     data.add_candidate(candi_sub, candi_rel_list)
                        #     data.add_sub_text(text)
        # data.add_node_score(countarry)
            # make score mat
        if hasattr(data, 'cand_sub') and hasattr(data, 'cand_rel'):##有召回的存储
            # remove duplicate relations
            data.remove_duplicate()
        else :
            NoneMatch += 1
       
        data.anonymous_question = form_anonymous_quesion(question, beg_idx, end_idx) 
        new_data_list.append(data)
            # append to new_data_list
        # elif save_all:
        #     new_data_list.append(data)
                
        # loging information
        if subject in candi_sub_list:
            succ_match += 1

        if data_index % 100 == 0:
            print( '{0} / {1}: {2} / {3} = {4}'.format( data_index, len(data_list), succ_match,data_index,succ_match/float(data_index)))
           
    log_file.write('{0} {1} {2} '.format(succ_match, data_index, NoneMatch))
    log_file.write( '{0} / {1} = {2} '.format(succ_match, data_index, succ_match / float(data_index)))
    log_file.write( 'not matched number is {0}'.format(NoneMatch))

    log_file.close()
    return new_data_list

def run():
    data_list   = pickle.load(open('../SimpleQuestions/PreprocessData/QAData.test.pkl', 'rb'))
    pred_list   = open('../FocusedLabeling/result.focused_labeling.txt', 'rb').readlines()
    suffix = 'QAData.test.pkl'.split('.')[-2]
    # #删除多余项
    # del(pred_list[len(data_list):-1])
    # del(pred_list[-1])
    # Create log directory
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    time_str = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    logName = 'logs/log.%s.txt'%time_str
    # Allocate dataload
    # Spawn processes
    new_data_list=query_candidate(data_list,pred_list,logName)
    pickle.dump(new_data_list, open('reduce.relQAData.label.%s.%s.cpickle'%(fb[:-1],suffix), 'wb'))
    
    strictMatch = 0
    nonMatch = 0
    
    temp = open(logName, 'rb').readlines()[0]
    tempNum = temp.split()
    strictMatch += float(tempNum[0])
    length = float(tempNum[1])
    nonMatch += float(tempNum[2])
    print('strictMatch is %d'%strictMatch)
    print('the recall is %f' % (strictMatch/length))####1066/21686=5%没有focused label
    print('not match is %f' %(nonMatch/length))
    valid_length = length - nonMatch
    return length,valid_length


if __name__ == '__main__':
    start_time = time.time()
    totalNum,validNum = run()
    # totalNum = 75776
    # validNum = 75776-1113
    # statisFun(totalNum,validNum)
    end_time = time.time()
    print('time is %f'%(end_time-start_time))
