#python3
import pickle
import io

# if __name__ == '__main__':##########relation word+近义词+下位词+变形
# 	word_corpus = {}
# 	relWord_corpus = {}
# 	rareWord_corpus = {}
# 	with open('word.glove100k.txt','r',encoding='utf8') as corpus_file:
# 		lines = corpus_file.readlines()
# 		for idx,word in enumerate(lines):
# 			word=word[:-1]
# 			word_corpus[word]=idx
# 	totalLen=len(word_corpus)
# 	file = open('word.relation.txt','r',encoding= 'utf8')
# 	relWords = file.readlines()
# 	file.close()
# 	NumRareWord = 0
# 	for _,relWord in enumerate(relWords):
# 		relWord = relWord[:-1]
# 		if relWord in word_corpus:
# 			relWord_corpus[relWord]=word_corpus[relWord]#3178
# 		else:
# 			NumRareWord+=1
# 			rareWord_corpus[relWord]=totalLen+NumRareWord
# 	with open('relWord.txt','w',encoding = 'utf8') as output_file:
# 		for _,idx in relWord_corpus.items():
# 			output_file.write(str(idx)+'\n')
# 	print(NumRareWord)#271

# 	with open('word.relation.txt','w',encoding='utf8') as output_file:
# 		for word in relWord_corpus.keys():
# 			output_file.write(word+'\n')
# 		for word in rareWord_corpus.keys():
# 			output_file.write(word+'\n')
# 			
# 			
if __name__ == '__main__':###relation word
	word_corpus = {}
	relWord_corpus = {}
	rareWord_corpus = {}
	with open('word.glove100k.txt','r',encoding='utf8') as corpus_file:
		lines = corpus_file.readlines()
		for idx,word in enumerate(lines):
			word = word.strip()
			word_corpus[word]=idx
	totalLen=len(word_corpus)
	# relWords = pickle.load(open('relWord.pickle','rb'))
	with open('word.relation.txt','r',encoding='utf8') as relword_file:
		relWords = []
		lines = relword_file.readlines()
		for _,word in enumerate(lines):
			word=word.strip()
			relWords.append(word)

	NumRareWord = 0
	for _,relWord in enumerate(relWords):
		if relWord in word_corpus:
			relWord_corpus[relWord]=word_corpus[relWord]#3178
		else:
			NumRareWord+=1
			rareWord_corpus[relWord]=totalLen+NumRareWord
	with open('relWord_id.txt','w',encoding = 'utf8') as output_file:
		for _,idx in relWord_corpus.items():
			output_file.write(str(idx)+'\n')
	print(NumRareWord)#271

	with open('word.relation.sorted.txt','w',encoding='utf8') as output_file:
		for word in relWord_corpus.keys():
			output_file.write(word+'\n')
		for word in rareWord_corpus.keys():
			output_file.write(word+'\n')