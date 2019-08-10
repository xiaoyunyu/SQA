# -*- coding: utf-8 -*-
import os, sys, re,time
import multiprocessing as mp
import cPickle as pickle
import numpy as np
import matplotlib  
matplotlib.use('Agg')  
  
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk import word_tokenize

sys.path.append( '../src/py_module' )
from QAData import *
import virtuoso
import freebase

type_dict = None
save_all = True
list_stopWords=list(set(stopwords.words('english')))
english_punctuations = ["'s",'-','',',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','``',"''",'/'] 
stop_words=list_stopWords+english_punctuations

def query_candidate(data_list, pid = 0):
    log_file = open('logs/log.round13.%d.txt'%(pid), 'wb')
    new_data_list = []
    data_index = 0
    NoneMatch = 0
    maxRelLen = 0
    for data in data_list:
        # incremnt data_index
        data_index += 1
        # extract fields needed
        relation = data.relation
        subject  = data.subject
        question = data.question
        ANquestion = data.anonymous_question
        if len(question.split()) > 1 and ANquestion:
            # query name / alias by subject (id)
            candi_rel_list = []
            candi_rel_list.extend(virtuoso.id_query_out_rel(subject))
           
            candi_rel_list=list(set(candi_rel_list))##[string,string...]
            if '' in candi_rel_list:
                candi_sub_list.remove('')
            data.add_candidate(subject, candi_rel_list)
            if relation in candi_rel_list:
                new_data_list.append(data)
                if len(candi_rel_list)>maxRelLen:
                    maxRelLen=len(candi_rel_list)
            else :
                NoneMatch += 1
                # print >> log_file,'%s' % (question)
            
    print ('not matched number is %d' % (NoneMatch)) 
    print('maximum candidate relation number is %d' % (maxRelLen))
    log_file.close()
    pickle.dump(new_data_list, file('temp.%d.cpickle'%(pid),'wb'))

def run():
    # Check number of argv
    # if len(sys.argv) == 4:
    #     # Parse input argument
    #     num_process = int(sys.argv[1])
    #     data_list   = pickle.load(file(sys.argv[2], 'rb'))
    #     pred_list   = file(sys.argv[3], 'rb').readlines()
    # elif len(sys.argv) == 5:
    #     # Parse input argument
    #     num_process = int(sys.argv[1])
    #     data_list   = pickle.load(file(sys.argv[2], 'rb'))
    #     pred_list   = file(sys.argv[3], 'rb').readlines()
    #     type_dict   = pickle.load(file(sys.argv[4], 'rb'))
    # else:
    #     print 'usage: python query_candidate_relation.py num_processes QAData_cpickle_file attention_score_file [[type_dict]]'
    #     sys.exit(-1)


    num_process = 6
    data_list   = pickle.load(file('QAData.test.pkl', 'rb'))
    suffix = 'QAData.test.pkl'.split('.')[-2]

    # Allocate dataload
    length = len(data_list)
    data_per_p = (length + num_process - 1) / num_process
    print(data_per_p)
    print(length)
    # Spawn processes
    processes = [ 
        mp.Process(
            target = query_candidate,
            args = (data_list[pid*data_per_p:(pid+1)*data_per_p], 
                    pid)   
        )   
        for pid in range(num_process)
    ]  


    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    # Merge all data [this will preserve the order]
    new_data_list = []
    for p in range(num_process):
        temp_fn = 'temp.%d.cpickle'%(p)
        new_data_list.extend(pickle.load(file(temp_fn, 'rb')))
    pickle.dump(new_data_list, file('relQADataAN.label.%s.cpickle'%(suffix), 'wb'))
    
    # Remove temp data
    for p in range(num_process):
        temp_fn = 'temp.%d.cpickle'%(p)
        os.remove(temp_fn)



def statisFun(totalNum,validNum):
    dataList = pickle.load((file('relQAData.label.ic.train.cpickle','rb')))
    canNum = 0#有候选的
    recallNum = 0#候选中有正确答案的
    singleMatch = 0
    MultiMatch = 0
    canLen = np.zeros(validNum)
    relLen = np.zeros(validNum)
    index = -1
    for _,data in enumerate(dataList):
        if hasattr(data,'cand_sub') :
            index +=1
            candi_sub_list = data.cand_sub
            candi_rel_list = data.cand_rel
            subject = data.subject
            canLen[index] = len(candi_sub_list)
            relLen[index] = len(candi_rel_list)
            if subject in candi_sub_list:#有召回的
                recallNum += 1
                
                if len(candi_sub_list) == 1:
                    singleMatch += 1
                else:
                    MultiMatch += 1
    canNum = float(index+1)
    recallNum = float(recallNum)
    avg_relLen = relLen.mean()
    print('candidate return is %f,recall is %f,single matched is %f,multiple matched is %f'
        % (canNum/totalNum, recallNum/totalNum, singleMatch/recallNum, MultiMatch/recallNum))
    print('the average relation candicate length is %f'%avg_relLen)
    plt.hist(canLen,100,range=[100,4000],facecolor='green')
    plt.savefig('candiLen.jpg')

    plt.hist(relLen,1000,facecolor='green')
    plt.savefig('relLen.jpg')


if __name__ == '__main__':
    start_time = time.time()
    run()
    # totalNum = 75776
    # validNum = 75776-1113
    #statisFun(totalNum,validNum)
    end_time = time.time()
    print('time is %f'%(end_time-start_time))
