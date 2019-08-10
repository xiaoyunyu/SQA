import io,sys,time,os
import multiprocessing as mp
import cPickle as pickle
# relation=set([])
# with io.open('freebase-FB5M.txt','r',encoding='utf8') as f:
# 	for i,line in enumerate(f.readlines()):
# 		relation = relation | set([line.split('\t')[1]])
# 		if i > 50 :
# 			break
def relExtra(dataList,pid):
	relation=set([])
	for i,line in enumerate(dataList):
		relation = relation | set([line.split('\t')[1].replace('www.freebase.com/','')])
	pickle.dump(relation, file('temp.%s.pkl'%(pid), 'wb'))

def process(num_process,dataList):
	    # Split workload
    length = 22441880
    data_per_p = (length + num_process - 1) / num_process

    # Spawn processes
    processes = [
        mp.Process(
            target = relExtra,
            args = ( 
                dataList[pid*data_per_p:(pid+1)*data_per_p],
                pid
                )
            )
        for pid in range(num_process-1)
    ]
    processes.append(
        mp.Process(
            target = relExtra,
            args = ( 
                dataList[(num_process-1)*data_per_p:-1],
                num_process-1
                )
            )
    )

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

if __name__ == '__main__':
	# time_start=time.time()

	# relation = set([])
	
	# with io.open('freebase-FB5M.txt','r') as f:
	# 	dataset = f.readlines()
	# 	process(num_process, dataset)
	# time_end=time.time()
	# print((time_end-time_start)/60)
    
    # num_process = 12
    # new_data_list = set([])
    # for p in range(num_process):
    #     temp_fn = 'temp.%d.pkl'%(p)
    #     new_data_list=new_data_list | pickle.load(file(temp_fn, 'rb'))
    #     os.remove(temp_fn)

    # relation=list(new_data_list)
    # pickle.dump(relation, file('relation.pkl', 'wb'))
    # typeSet=set([])
    # for i,j in enumerate(relation):
    #     temp=j.split('/')[0]
    #     typeSet = typeSet | set([temp])
    # print(len(list(typeSet)))#92

    # with io.open('../annotated_fb_data_train.txt','r') as f:
    #     dataset = f.readlines()
    #     relExtra(dataset,'train')
    # relationList = list(pickle.load(file('temp.train.pkl','rb')))
    # typeSet=set([])
    # for i,j in enumerate(relationList):
    #     temp=j.split('/')[0]
    #     typeSet = typeSet | set([temp])
    # print(len(list(typeSet)))
    # 
    with io.open('../annotated_fb_data_valid.txt','r') as f:
        dataset = f.readlines()
        relExtra(dataset,'valid')
    relationList = list(pickle.load(file('temp.valid.pkl','rb')))
    typeSet=set([])
    for i,j in enumerate(relationList):
        temp=j.split('/')[0]
        typeSet = typeSet | set([temp])
    trainList = pickle.load(file('temp.train.pkl','rb'))
    for i,j in enumerate(trainList):
        temp=j.split('/')[0]
        typeSet = typeSet | set([temp])    
    testList = pickle.load(file('temp.test.pkl','rb'))
    for i,j in enumerate(testList):
        temp=j.split('/')[0]
        typeSet = typeSet | set([temp])  
    print((list(typeSet)))