import io
import numpy as np

cat = "test"

relLen = 0
count = 0
negSize = 0
with io.open('%s.replace_ne.withpool'%cat,'r',encoding = 'utf-8') as fo:
	file = fo.readlines()
	for k,v in enumerate(file):
		fields = v.split('\t')
		posRel = fields[0]
		negRel = fields[1].split(' ')
		if posRel not in negRel:
			count += 1
		if int(posRel) > relLen:
			relLen = int(posRel)
		if len(negRel)>negSize:
			negSize=len(negRel)
print(relLen)
print(count)
print(negSize)



with io.open('new_relation.2M.list','w',encoding = 'utf-8') as out:
	with io.open('relation.2M.list','r',encoding = 'utf-8') as fo:
		file = fo.readlines()
		for k,v in enumerate(file):
			rel=v.replace('/','.')[1:]
			out.write('fb:'+rel)

relList_2M = {}
with io.open('new_relation.2M.list','r',encoding ='utf-8') as fo:
	file = fo.readlines()
	for k,v in enumerate(file):
		v=v[0:-1]
		relList_2M[k+1]=v

relList_5M={}
with io.open('../KnowledgeBase/FB5M.rel.txt','r',encoding ='utf-8') as fo:
	file = fo.readlines()
	for k,v in enumerate(file):
		v=v[0:-1]
		relList_5M[v]=k+1

for relID in range(0,len(relList_2M)):
	key = relList_2M[relID+1]
	newID = relList_5M[key]
	relList_2M[relID+1] = newID

dataList = []
with io.open('new_%s.txt'%cat,'w',encoding = 'utf-8') as out:
	with io.open('%s.replace_ne.withpool'%cat,'r',encoding = 'utf-8') as fo:
		file = fo.readlines()
		for k,v in enumerate(file):
			fields = v.split('\t')
			posRel = int(fields[0])
			negRel = fields[1].split(' ')
			question = fields[2].replace('#head_entity#','X')[0:-1]

			posRel = str(relList_2M[posRel])
			for i in range(0,len(negRel)):
				if negRel[i] != 'noNegativeAnswer':
					negRel[i]=str(relList_2M[int(negRel[i])])
				else :
					negRel=''
			negRels = '\t'.join(negRel)
			dataList.append([question,posRel,negRels,len(question.split(' '))])
		dataList.sort(key = lambda data:data[-1],reverse=True)
	data_num = len(dataList)
	chosen_num = data_num - (data_num % 128)
	chosen_indices = np.sort(np.random.permutation(data_num)[:chosen_num])
	chosen_indices_idx = 0
    # for each data triple in data_turple list
	for idx in range(data_num):
		data = dataList[idx]
		data[0]=data[0].replace("-lrb-","(")
		data[0]=data[0].replace("-rrb-",")")
		if idx == chosen_indices[chosen_indices_idx]:
			if data[2] != '':
				out.write(data[0]+'\t'+data[1]+'\t'+data[2]+'\n')
			else:
	 			out.write(data[0]+'\t'+data[1]+'\n')
			chosen_indices_idx += 1

relList_5M={}
with io.open('../KnowledgeBase/FB5M.rel.txt','r',encoding ='utf-8') as fo:
	file = fo.readlines()
	for k,v in enumerate(file):
		v=v.strip()
		relList_5M[k+1]=v

with open('%s.txt'%cat,'w',encoding='utf8') as fo:
	with open('new_%s.txt'%cat,'r',encoding='utf8') as fi:
		lines = fi.readlines()
		for line in lines:
			fields=line.strip().split('\t')
			out = ""
			out+=fields[0]
			for num in fields[1:]:
				out+='\t'
				out+=relList_5M[int(num)]
			fo.write(out+'\n')