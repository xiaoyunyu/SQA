import io
with io.open('FB5M.rel.txt','r',encoding='utf-8') as inp:
	rr=[]
	rt=[]
	rtt=[]
	maxlen = 0
	rels = inp.readlines()
	for rel in rels:
		rel_comp = rel.strip().split('.')
		rr.append(rel_comp[-1])
		rtt.append(rel_comp[0])
		# if len(rel_comp[:-1]) > maxlen:
		# 	maxlen=len(rel_comp[:-1])
		rt.append('.'.join(rel_comp[0:-1]))

	rr= list(set(rr))
	rt = list(set(rt))
	rtt = list(set(rtt))
	print(len(rr),len(rt),len(rtt))
	with io.open('rr.txt','w',encoding='utf-8') as out:
		out.write('\n'.join(rr))
	with io.open('rt.txt','w',encoding='utf-8') as out:
		out.write('\n'.join(rt))
	with io.open('rtt.txt','w',encoding='utf-8') as out:
		out.write('\n'.join(rtt))

	# print(maxlen)
