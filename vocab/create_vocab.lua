require '..'

function createWordVocab(src_path,tg_path)
    local wordVocab = cudacheck(Vocab(src_path))
    -- wordVocab:add_X_token()
    wordVocab:add_unk_token()
    wordVocab:add_pad_token()

    torch.save(tg_path, wordVocab)
end
-- createWordVocab('word.glove100k.txt','non_vocab.word.t7')
-- createWordVocab('relWordVocab/word.relation.sorted.txt','vocab.relword.t7')--7735


function createFBVocab()
    local vocabPath = '../KnowledgeBase'

    -- local relVocab = Vocab(vocabPath..'/FB5M.rel.txt')
    local relVocab = Vocab('../KnowledgeBase/FB5M.rel.txt')
    relVocab:add_unk_token()
    local word_vocab = torch.load('vocab.relword.t7')
    -- local rel2wordPath = 'relWordVocab/rel.txt'
    local rel2synPath = 'relWordVocab/rel_syn.txt'
    relVocab:add_word_token(word_vocab,nil,7523)
    relVocab:add_syn_token(word_vocab,rel2synPath,7523)
    -- local entVocab = Vocab(vocabPath..'/FB5M.ent.txt')
    -- entVocab:add_unk_token()

    torch.save('vocab.rel.t7', relVocab)
    -- torch.save('vocab.ent.t7', entVocab)
end

function createCompVocab()
    local RTTVocab = cudacheck(Vocab('../KnowledgeBase/rtt.txt'))
    -- local RTVocab = cudacheck(Vocab('../KnowledgeBase/rt.txt'))
    -- wordVocab:add_X_token()
    RTTVocab:add_unk_token()
    -- RTVocab:add_unk_token()

    torch.save('vocab.rtt.t7', RTTVocab)
    -- torch.save('vocab.rt.t7', RTVocab)
end

-- createWordVocab()
createFBVocab()
-- createCompVocab()