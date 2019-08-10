import sys, os
import cPickle as pickle

def www2fb(in_str):
    out_str = 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    return out_str

def main():

    dbData = ['www.freebase.com/m/09c7w0    www.freebase.com/location/location/contains www.freebase.com/m/04947f8',
        'www.freebase.com/m/064t9   www.freebase.com/music/genre/artists    www.freebase.com/m/0pdnwwz',
        'www.freebase.com/m/09c7w0  www.freebase.com/location/location/people_born_here www.freebase.com/m/05x3p27',
        'www.freebase.com/m/01hmnh  www.freebase.com/tv/tv_genre/programs   www.freebase.com/m/0yq217r',
        'www.freebase.com/m/03npn   www.freebase.com/media_common/netflix_genre/titles  www.freebase.com/m/0sy44',
        'www.freebase.com/m/09c7w0  www.freebase.com/location/location/contains www.freebase.com/m/0dmjdr',
        'www.freebase.com/m/09c7w0  www.freebase.com/location/location/contains www.freebase.com/m/0clvycv',
        'www.freebase.com/m/09c7w0  www.freebase.com/location/location/contains www.freebase.com/m/03m7brv',
        'www.freebase.com/m/0hcr    www.freebase.com/tv/tv_genre/programs   www.freebase.com/m/07s05f',
        'www.freebase.com/m/09c7w0  www.freebase.com/location/location/contains www.freebase.com/m/0t1g4',
        'www.freebase.com/m/09c7w0  www.freebase.com/location/location/people_born_here www.freebase.com/m/026bwsm',
        'www.freebase.com/m/09c7w0  www.freebase.com/location/location/contains www.freebase.com/m/0rn65',
        'www.freebase.com/m/03rk0   www.freebase.com/media_common/netflix_genre/titles  www.freebase.com/m/0crx41v',
        'www.freebase.com/m/016clz  www.freebase.com/music/genre/artists    www.freebase.com/m/01w4b3m',
        'www.freebase.com/m/0jtdp   www.freebase.com/media_common/netflix_genre/titles  www.freebase.com/m/0crth9x',
        'www.freebase.com/m/09c7w0  www.freebase.com/location/location/contains www.freebase.com/m/0xhc6',
        'www.freebase.com/m/09c7w0  www.freebase.com/location/location/contains www.freebase.com/m/049_nd8',
        'www.freebase.com/m/09c7w0  www.freebase.com/location/location/contains www.freebase.com/m/0493c1v',
        'www.freebase.com/m/0hcr    www.freebase.com/tv/tv_genre/programs   www.freebase.com/m/04c5yg',
        'www.freebase.com/m/09c7w0  www.freebase.com/location/location/contains www.freebase.com/m/05c6r0']
    for k,v in enumerate(dbData):
        fields = v.split(' ')
        fields = [f for f in fields if f!='']
        sub = www2fb(fields[0])
        rel = www2fb(fields[1])
        obj = www2fb(fields[2])
        dbData[k]=(sub,rel,obj)
    triple_dict = pickle.load(file('triple.pickle','rb'))
    out_fn = 'FB5M.core.txt'
    count = 0
    with file(out_fn, 'wb') as fo:
        for (sub, rel, obj) in triple_dict.keys():
            count += 1
            if (sub, rel, obj) in dbData:
                print(count)
            print >> fo, '<%s>\t<%s>\t<%s>\t.' % (sub, rel, obj)
    print len(triple_dict)
    print(count)
    # for k,v in enumerate(dbData):
    #     if v in triple_dict:
    #         print('the data is in the dictionary')
    #     else:
    #         print('the %d th data is lost'%k)

    # in_fn = sys.argv[1]
    # db = in_fn.split('-')[-1].split('.')[0]

    # out_fn = '%s.core.txt' % (db)
    # ent_fn = '%s.ent.pkl' % (db)
    # rel_fn = '%s.rel.pkl' % (db)

    # ent_dict = {}
    # rel_dict = {}
    # triple_dict = {}

    # with file(in_fn, 'rb') as fi:
    #     for line in fi:
    #         fields = line.strip().split('\t')
    #         sub = www2fb(fields[0])
    #         rel = www2fb(fields[1])
    #         objs = fields[2].split()
    #         if ent_dict.has_key(sub):
    #             ent_dict[sub] += 1
    #         else:
    #             ent_dict[sub] = 1
    #         if rel_dict.has_key(rel):
    #             rel_dict[rel] += 1
    #         else:
    #             rel_dict[rel] = 1
    #         for obj in objs:
    #             obj = www2fb(obj)
    #             triple_dict[(sub, rel, obj)] = 1
    #             if ent_dict.has_key(obj):
    #                 ent_dict[obj] += 1
    #             else:
    #                 ent_dict[obj] = 1

    # pickle.dump(triple_dict,file('triple.pickle','wb'))
    # pickle.dump(ent_dict, file(ent_fn, 'wb'))
    # with file('%s.ent.txt' % (db), 'wb') as fo:
    #     for k, v in sorted(ent_dict.items(), key = lambda kv: kv[1], reverse = True):
    #         print >> fo, k

    # pickle.dump(rel_dict, file(rel_fn, 'wb'))
    # with file('%s.rel.txt' % (db), 'wb') as fo:
    #     for k, v in sorted(rel_dict.items(), key = lambda kv: kv[1], reverse = True):
    #         print >> fo, k

    # with file(out_fn, 'wb') as fo:
    #     for (sub, rel, obj) in triple_dict.keys():
    #         print >> fo, '<%s>\t<%s>\t<%s>\t.' % (sub, rel, obj)
    # print len(triple_dict)

if __name__ == '__main__':
    main()
