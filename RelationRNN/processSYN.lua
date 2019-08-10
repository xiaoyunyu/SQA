require '../init3.lua'

function trainData(negsize)
    wordVocab = torch.load('../vocab/vocab.word.t7')
    relationVocab = torch.load('../vocab/vocab.rel.t7')
    relWordVocab = torch.load('../vocab/vocab.relword.t7')
    print(wordVocab.size)
    print(relationVocab.size)
    print(relWordVocab.size)

    createSeqRankingData('../dataset/train.txt', '../data/train.relword.torch', wordVocab, relationVocab, 128,relWordVocab,negsize,9,40)
end

function testData(suffix,negsize,use_random,target_path)
    wordVocab = torch.load('../vocab/vocab.word.t7')
    relationVocab = torch.load('../vocab/vocab.rel.t7')
    relWordVocab = torch.load('../vocab/vocab.relword.t7')
    print(wordVocab.size)
    print(relationVocab.size)
    print(relWordVocab.size)
    use_random = ((use_random==nil) and false) or use_random
    target_path = ((target_path==nil) and '../data/'..suffix..'.rel.FB5M.torch') or target_path
    createRankingData('../dataset/'..suffix..'.txt', target_path, wordVocab, relationVocab, 64, relWordVocab,negsize,9,40,use_random)--4)
end


trainData(150)
testData('test',150)
-- for i = 5,250,10 do
--     testData('test', i,true,'/home/yxy/base/CFO/data/test.rel.FB5M.'..i..'.torch')
-- end
