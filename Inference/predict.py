# -*- coding: utf-8 -*-
#python3
import numpy as np
import pickle
import sys
sys.path.append( '../src/py_module' )
from qa_data import  QAData
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import progressbar
from tqdm import trange
import virtuoso


def readRelRank(path):
	file = open(path,'r')
	fields = file.readlines()
	relRanks = fields[0].strip().split('\t')

	return relRanks

def relfindsubs(data,relRank):
	candi_rel = data.cand_rel
	candi_sub = data.cand_sub
	sub_rels = data.sub_rels
	pred_rel = candi_rel[relRank]
	found_subs = []
	assert(len(candi_sub)==len(sub_rels),'error')
	for i in range(len(candi_sub)):
		if pred_rel in sub_rels[i]:
			found_subs.append(i)

	return pred_rel, found_subs

def predSub(data,found_subs):
	candi_sub = data.cand_sub
	candi_sub_scores = data.cand_sub_score
	sub_scores = []
	try:
		assert(len(candi_sub)==len(candi_sub_scores))
	except:
		print(len(candi_sub),len(candi_sub_scores))
		raise
	for i in range(len(found_subs)):
		idx = found_subs[i]
		sub_scores.append((candi_sub[idx],candi_sub_scores[idx]))

	sub_scores.sort(key = lambda data:data[-1])
	sub_scores.reverse()
	pred_sub = sub_scores[0][0]
	return pred_sub

def predSub_wo_rel(data,p):
	sub_scores=[]
	candi_sub = data.cand_sub
	candi_sub_scores = data.cand_sub_scores
	sub_len = len(candi_sub)
	for i in range(sub_len):
		sub_scores.append((candi_sub[i],candi_sub_scores[i]))

	sub_scores.sort(key= lambda data:data[-1])
	sub_scores.reverse()
	if sub_len < p:
		pred_sub = sub_scores
	else:
		pred_sub = sub_scores[0:p]

	return pred_sub

def update_candi_sub_scores(data):
	candi_subs = data.cand_sub
	scores = np.zeros(len(candi_subs))
	for idx,sub in enumerate(candi_subs):
		scores[idx] = virtuoso.id_query_count(sub)
		# print(scores[idx])
	data.add_tfidf_score(scores)
	# print(data.cand_sub_scores)

if __name__ == '__main__':
	dataList = pickle.load(open('reduce.relQAData.label.fb2m.test.cpickle','rb'))
	relRanks = readRelRank('../RelationRNN/predicted_relation.txt')
	# preCount_wo_rel = np.zeros(totRange-1)
	preCount = np.zeros(1)
	cnt_strict = 0
	cnt_non_strict = 0
	tqdm_range = trange(0, len(dataList))
	for i in tqdm_range:
		data = dataList[i]
		if hasattr(data,'cand_rel'):
			candi_rel = data.cand_rel
		else:
			continue
		trueSub = data.subject
		trueRel = data.relation
		candi_sub = data.cand_sub
		relRank = int(relRanks[i])-1
		if relRank>=len(candi_rel):
			continue
		pred_rel, found_subs = relfindsubs(data, relRank)

		# for p in range(1,totRange):
		# 	pred_sub = predSub_wo_rel(data,p)
		# 	pred_sub = [sub[0] for sub in pred_sub]
		# 	if trueSub in pred_sub:
		# 		preCount_wo_rel[p-1] += 1
		if data.strict_flag:
			if pred_rel == trueRel:
				pred_sub = predSub(data,found_subs)
				if trueSub == pred_sub:
					cnt_non_strict+=1
					cnt_strict+=1
		else:
			if pred_rel == trueRel:
				pred_sub = predSub(data,found_subs)
				if trueSub == pred_sub:
					cnt_non_strict+=1



	print('In strict matching case, precision is %f'%(cnt_strict/len(dataList)))
	print('In non-strict matching case, precision is %f'%(cnt_non_strict/len(dataList)))

