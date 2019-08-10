require 'init.lua'

function trainData()
    local wordVocab = torch.load('vocab/non_vocab.word.t7')
    local entVocab = torch.load('vocab/vocab.ent.t7')
    local relVocab = torch.load('vocab/vocab.rel.t7')

    trainDir = '../SimpleQuestions/trainingData'

    -- -- focused labeling
    createSeqLabelingData(trainDir..'/data.train.focused_labeling', 'data/train.focused_labeling.t7', wordVocab, 128)
    createSeqLabelingData(trainDir..'/data.test.focused_labeling', 'data/data.test.focused_labeling.t7', wordVocab, 1)

end

trainData()
