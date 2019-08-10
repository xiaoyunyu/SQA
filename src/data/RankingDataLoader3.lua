local RankingDataLoader = torch.class('RankingDataLoader')

function RankingDataLoader:__init(datafile, negRange, logger)
    -- sequence & pos match
    local data = torch.load(datafile)
    self.sequences  = data.seq
    self.totMatches_w = data.relWord
    self.totMatches_r = data.rel
    self.totMatches_s = data.relSyn
    -- self.pos = data.pos


    -- additional variables
    self.batchSize = self.sequences[1]:size(2)
    self.relWordLen = self.totMatches_w[1]:size(1)
    self.numBatch  = #self.sequences
    self.currIdx   = 1
    self.indices   = randperm(self.numBatch)
    self.negRange = negRange
    self.negSize = self.totMatches_r[1]:size(1)-1
    self.lastBatchlen = data.lastBatchlen

    if torch.Tensor():type() == 'torch.CudaTensor' then
        for i = 1, self.numBatch do
            self.sequences[i]  = self.sequences[i]:cuda()            
            self.totMatches_w[i] = self.totMatches_w[i]:cuda()
            self.totMatches_r[i] = self.totMatches_r[i]:cuda()
            self.totMatches_s[i] = self.totMatches_s[i]:cuda()
            -- self.pos[i] = self.pos[i]:cuda()

        end    
    end

    if logger then
        self.logger = logger
        self.logger.info(string.rep('-', 50))
        self.logger.info(string.format('SeqRankingLoader Configurations:'))
        self.logger.info(string.format('    number of batch : %d', self.numBatch))
        self.logger.info(string.format('    data batch size : %d', self.batchSize))
        self.logger.info(string.format('    neg sample range: %d', self.negRange))
    end
end

function RankingDataLoader:nextBatch(circular)
    if self.currIdx > self.numBatch then
        self.currIdx = 1
        self.indices = randperm(self.numBatch)
    end
    local dataIdx
    if circular then
        dataIdx = self.currIdx
    else
        dataIdx = self.indices[self.currIdx]
    end 
    self.currIdx = self.currIdx + 1


    return self.sequences[dataIdx], self.totMatches_w[dataIdx], self.totMatches_r[dataIdx],self.totMatches_s[dataIdx]


end

function createRankingData(dataPath, savePath, wordVocab, fbVocab, batchSize,relWordVocab,negSize,relWordLen,max_pos,use_random)
    -- class variables
    local totMatches_w = {}
    local totMatches_r = {}
    local totMatches_s = {}
    local pos_r = {}
    local sequences  = {}
    local pos = {}

    -- read data fileh
    local file = io.open(dataPath, 'r')
    local batchIdx = 0    -- the index of sequence batches
    local seqIdx   = 0    -- sequence index within each batch
    local line
    local lens = 0
    local flag_freeNegSize = 0
    if negSize == 0 then
        flag_freeNegSize = 1
    end
    local ind = 0
    while true do
        line = file:read()
        if line == nil then break end
        lens = lens + 1
        line = stringx.strip(line)
        local fields = stringx.split(line, '\t')
        
        -- fields[1]: language sequence
        local tokens = stringx.split(fields[1])
        ind = ind + 1

        if flag_freeNegSize == 1 then
            negSize = #fields-2
        end

        -- allocate tensor memory
        if seqIdx % batchSize == 0 then
        -- print('batch: '..batchIdx)
        seqIdx = 1
        batchIdx = batchIdx + 1            

        pos_r[batchIdx] = torch.LongTensor(batchSize, negSize+1, relWordLen):fill(max_pos)
        sequences [batchIdx] = torch.LongTensor(batchSize, #tokens):fill(wordVocab.pad_index)
        totMatches_w[batchIdx] = torch.LongTensor(batchSize, negSize+1, relWordLen):fill(relWordVocab.pad_index)
        totMatches_r[batchIdx] = torch.LongTensor(batchSize, negSize+1):fill(fbVocab.unk_index)
        totMatches_s[batchIdx] = torch.LongTensor(batchSize, negSize+1, relWordLen):fill(relWordVocab.pad_index)
        else
            seqIdx = seqIdx + 1
        end

        -- parse each token in sequence
        for i = 1, #tokens do
            local token = tokens[i]            
            sequences[batchIdx][{seqIdx, i}] = wordVocab:index(token)
        end
        
        -- fields[2]: positive match
        -- local posMatch = fbVocab:index(fields[2])

        local posMatch = tonumber(fields[2])
        if not posMatch then
            posMatch=fbVocab:index(fields[2])
        end
        local posWord = fbVocab:index2word(posMatch)
        local rel_word_len = torch.ne(posWord,relWordVocab.pad_index):sum()
        local posSyn=torch.LongTensor(relWordLen):fill(relWordVocab.pad_index) --= fbVocab:index2syn(posMatch)
        pcall(function(posMatch) posSyn = fbVocab:index2syn(posMatch) end, posMatch)
        if not posSyn then
            print(posMatch)
        end
        if posMatch ~= fbVocab.unk_index then
            totMatches_w[batchIdx][seqIdx][1][{{1,posWord:size(1)}}] = posWord
            totMatches_r[batchIdx][seqIdx][1] = posMatch
            pos_r[batchIdx][seqIdx][1][{{1,rel_word_len}}] = torch.range(1,rel_word_len)
            totMatches_s[batchIdx][seqIdx][1][{{1,posSyn:size(1)}}] = posSyn
        end
        -- fields[3]: negtive match
        if not use_random then
            for i = 3,#fields,1 do
                -- local negMatch = fbVocab:index(fields[i])
                local negMatch = tonumber(fields[i])
                if not negMatch then
                    negMatch=fbVocab:index(fields[i])
                end
                local negWord = fbVocab:index2word(negMatch)
                local negSyn = torch.LongTensor(relWordLen):fill(relWordVocab.pad_index) --= fbVocab:index2syn(posMatch)
                pcall(function(negMatch) negSyn = fbVocab:index2syn(negMatch) end, negMatch)
                if not negSyn then
                    print(negMatch)
                end
                local rel_word_len = torch.ne(negWord,relWordVocab.pad_index):sum()
                if negWord then
                    totMatches_w[batchIdx][seqIdx][i-1][{{1,negWord:size(1)}}] = negWord
                    totMatches_r[batchIdx][seqIdx][i-1] = negMatch
                    totMatches_s[batchIdx][seqIdx][i-1][{{1,negSyn:size(1)}}] = negSyn

                    pos_r[batchIdx][seqIdx][i-1][{{1,rel_word_len}}] = torch.range(1,rel_word_len)
                end
            end
        else
            local insert_pos = math.random(1,negSize)
            for i = 1, negSize do
                    if i==insert_pos then
                        negMatch = posMatch
                    else
                        negMatch = math.random(1,fbVocab.size-1)
                        while(negMatch==posMatch) do
                            negMatch = math.random(1,fbVocab.size-1)
                        end
                    end
                local negWord = fbVocab:index2word(negMatch)
                local negSyn = torch.LongTensor(relWordLen):fill(relWordVocab.pad_index) --= fbVocab:index2syn(posMatch)
                pcall(function(negMatch) negSyn = fbVocab:index2syn(negMatch) end, negMatch)
                local rel_word_len = torch.ne(negWord,relWordVocab.pad_index):sum()
                if negMatch ~= fbVocab.unk_index then
                    totMatches_w[batchIdx][seqIdx][i+1][{{1,negWord:size(1)}}] = negWord
                    totMatches_s[batchIdx][seqIdx][i+1][{{1,negSyn:size(1)}}] = negSyn
                    totMatches_r[batchIdx][seqIdx][i+1] = negMatch
                    pos_r[batchIdx][seqIdx][i+1][{{1,rel_word_len}}] = torch.range(1,rel_word_len)
                end
            end
        end
        --插入正关系

        -- if  (#fields-2)<negSize then
        --     for i = #fields,negSize+1 do
        --         local negWord = fbVocab:index2word(i)
        --         totMatches_w[batchIdx][seqIdx][i][{{1,posWord:size(1)}}] = posWord
        --         totMatches_r[batchIdx][seqIdx][i] = posMatch
        --     end
        -- end

    end
    print('batch: '..batchIdx)
    file:close()

    for idx = 1, batchIdx do
        sequences[idx] = sequences[idx]:transpose(1,2)
        totMatches_w[idx] = totMatches_w[idx]:transpose(1,3):contiguous():view(relWordLen,-1)
        totMatches_s[idx] = totMatches_s[idx]:transpose(1,3):contiguous():view(relWordLen,-1)
        totMatches_r[idx] = totMatches_r[idx]:transpose(1,2)
        pos_r[idx] = pos_r[idx]:transpose(1,3):contiguous():view(relWordLen,-1)
    end
    local lastBatchlen = lens%batchSize



    local data = {}
    -- data.pos = posMatches
    data.pos_r = pos_r
    --token*batchsize
    data.seq = sequences
    --relwordlen*negsize*batchsize
    data.relWord = totMatches_w
    data.relSyn = totMatches_s
    --negsize*batchsize
    data.rel = totMatches_r
    -- data.pos = pos
    data.lastBatchlen = lastBatchlen

    torch.save(savePath, data)
end
