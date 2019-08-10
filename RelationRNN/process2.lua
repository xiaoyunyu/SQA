require '../init2.lua'

function trainData()
    wordVocab = torch.load('../vocab/vocab.word.t7')
    relationVocab = torch.load('../vocab/vocab.rel.t7')
    relWordVocab = torch.load('../vocab/vocab.relword.back.t7')

    createSeqRankingData('../dataset/new_train.txt', '../data/train.relword.torch', wordVocab, relationVocab, 256, relWordVocab, 31, 9)
end

function testData(suffix,negsize,use_random,target_path)
    wordVocab = torch.load('../vocab/vocab.word.t7')
    relationVocab = torch.load('../vocab/vocab.rel.t7')
    relWordVocab = torch.load('../vocab/vocab.relword.back.t7')
    print(wordVocab.size)
    print(relationVocab.size)
    print(relWordVocab.size)
    target_path = ((target_path==nil) and '../data/'..suffix..'.rel.FB5M.torch') or target_path
    createRankingData('../dataset/'..suffix..'.txt', target_path, wordVocab, relationVocab, 64, negsize,use_random)--,32)
end

trainData()
testData('test',150,false)
-- for i = 5,250,10 do
--     testData(i,true,'/home/yxy/base/CFO/data/test.rel.FB5M.'..i..'.light.torch')
-- end