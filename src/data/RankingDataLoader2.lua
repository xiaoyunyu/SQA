local RankingDataLoader = torch.class('RankingDataLoader')

function RankingDataLoader:__init(datafile, logger)
    -- class variables
    local data = torch.load(datafile)
    self.sequences  = data.seq
    self.seqLengths = data.len
    self.posMatches = data.pos
    self.negMatches = data.neg
    self.lastBatchlen = data.lastBatchlen

    -- additional variables
    self.batchSize = self.sequences[1]:size(1)
    self.numBatch = #self.sequences
    self.negSize = self.negMatches[1]:size(2)
    self.currIdx = 1
    self.indices = randperm(self.numBatch)

    if torch.Tensor():type() == 'torch.CudaTensor' then
        for i = 1, self.numBatch do
            self.posMatches[i] = self.posMatches[i]:cuda()
            self.negMatches[i] = self.negMatches[i]:cuda()
            self.seqLengths[i] = self.seqLengths[i]:cuda()
            self.sequences[i]  = self.sequences[i]:cuda()
        end
    end

    if logger then
        self.logger = logger
        self.logger.info(string.rep('-', 50))
        self.logger.info(string.format('RankingDataLoader Configurations:'))
        self.logger.info(string.format('    number of batch: %d', self.numBatch))
        self.logger.info(string.format('    data batch size: %d', self.batchSize))
        self.logger.info(string.format('    neg sample size: %d', self.negSize))
        self.logger.info(string.format('    last batch size: %d', self.lastBatchlen))       
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
    return self.sequences[dataIdx], self.posMatches[dataIdx], self.negMatches[dataIdx], self.seqLengths[dataIdx]
end

function createRankingData(dataPath, savePath, wordVocab, fbVocab, batchSize, negSize,use_random)
    -- class variables
    local posMatches = {}
    local negMatches = {}
    local seqLengths = {}
    local sequences  = {}
    local lastBatchlen = 0

    -- read data fileh
    local file = io.open(dataPath, 'r')
    local batchIdx = 0    -- the index of sequence batches
    local seqIdx   = 0    -- sequence index within each batch
    local line
    local flag = 0
    if negSize==0 then
        flag = 1
    end
    
    while true do
        line = file:read()
        if line == nil then break end
        lastBatchlen = lastBatchlen + 1
        local fields = stringx.split(line, '\t')
        
        -- fields[1]: language sequence
        local tokens = stringx.split(fields[1])

        if flag == 1 then
            negSize = #fields-2
        end
        -- allocate tensor memory
        if seqIdx % batchSize == 0 then
            print('batch: '..batchIdx)
            seqIdx = 1
            batchIdx = batchIdx + 1
            sequences[batchIdx]  = torch.LongTensor(batchSize, #tokens):fill(wordVocab.pad_index)
            seqLengths[batchIdx] = torch.LongTensor(batchSize):fill(0)
            posMatches[batchIdx] = torch.LongTensor(batchSize):fill(fbVocab.unk_index)
            negMatches[batchIdx] = torch.LongTensor(batchSize, negSize):fill(fbVocab.unk_index)
        else
            seqIdx = seqIdx + 1
        end

        -- parse each token in sequence
        for i = 1, #tokens do
            local token = tokens[i]
            sequences[batchIdx][{seqIdx, i}] = wordVocab:index(token)
        end
        seqLengths[batchIdx][seqIdx] = #tokens
        
        -- fields[2]: positive match
        posMatches[batchIdx][seqIdx] = fbVocab:index(fields[2])

        -- fields[3-#fields]: negative match
        if not use_random then
            for i = 3, #fields do
                negMatches[batchIdx][{seqIdx, i-2}] = fbVocab:index(fields[i])
            end
        else
            local insert_pos = math.random(1,negSize)
            local posMatch = fbVocab:index(fields[2])
            for i = 1, negSize do
                if i==insert_pos then
                    negMatch = posMatch
                else
                    negMatch = math.random(1,fbVocab.size-1)
                    while(negMatch==posMatch) do
                        negMatch = math.random(1,fbVocab.size-1)
                    end
                end
                negMatches[batchIdx][{seqIdx, i}] = negMatch
            end
        end

    end
    file:close()
    lastBatchlen = lastBatchlen%batchSize

    -- table.remove(posMatches,batchIdx)
    -- table.remove(seqLengths,batchIdx)
    -- table.remove(sequences,batchIdx)

    local data = {}
    data.pos = posMatches
    data.neg = negMatches
    data.len = seqLengths
    data.seq = sequences
    data.lastBatchlen = lastBatchlen    

    torch.save(savePath, data)
end
