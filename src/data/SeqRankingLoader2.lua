local SeqRankingLoader = torch.class('SeqRankingLoader')

function SeqRankingLoader:__init(datafile, negSize, negRange, logger)
    -- sequence & pos match
    local data = torch.load(datafile)
    self.sequences  = data.seq
    self.posMatches = data.pos
    if data.len ~= nil then
        self.seqLengths = data.len
    end

    -- for negative sampling
    self.negSize  = negSize
    self.negRange = negRange

    -- additional variables
    self.batchSize = self.sequences[1]:size(1)
    self.numBatch  = #self.sequences
    self.currIdx   = 1
    self.indices   = randperm(self.numBatch)
    -- allocate memory
    self._negMatch = torch.LongTensor(self.batchSize, self.negSize)
    self._posMatch = torch.LongTensor(self.batchSize,1):expand(self.batchSize,self.negSize)

    if torch.Tensor():type() == 'torch.CudaTensor' then
        for i = 1, self.numBatch do
            self.sequences[i]  = self.sequences[i]:cuda()            
            self.posMatches[i] = self.posMatches[i]:cuda()
            if self.seqLengths ~= nil then
                self.seqLengths[i] = self.seqLengths[i]:cuda()
            end
        end
        self.negMatch = torch.CudaTensor(self.batchSize,self.negSize)
    else
        self.negMatch = self._negMatch
    end

    if logger then
        self.logger = logger
        self.logger.info(string.rep('-', 50))
        self.logger.info(string.format('SeqRankingLoader Configurations:'))
        self.logger.info(string.format('    number of batch : %d', self.numBatch))
        self.logger.info(string.format('    data batch size : %d', self.batchSize))
        self.logger.info(string.format('    neg sample size : %d', self.negSize))
        self.logger.info(string.format('    neg sample range: %d', self.negRange))
    end
end

function SeqRankingLoader:setNegSize(negSize)
    self.negSize  = negSize
    
    -- allocate memory
    self._negMatch = torch.LongTensor(self.batchSize,self.negSize)
    self._posMatch = torch.LongTensor(self.batchSize,1):expand(self.batchSize,self.negSize)

    if torch.Tensor():type() == 'torch.CudaTensor' then
        self.negMatch = torch.CudaTensor(self.batchSize,self.negSize )
    end
end

function SeqRankingLoader:nextBatch(circular)
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
    self._posMatch:storage():copy(self.posMatches[dataIdx]:storage())
    self._negMatch:random(1, self.negRange)

    while torch.sum(torch.eq(self._negMatch, self._posMatch)) > 0 do
        self._negMatch:maskedFill(torch.eq(self._negMatch, self._posMatch), math.random(1, self.negRange))
    end

    if torch.Tensor():type() == 'torch.CudaTensor' then
        self.negMatch:copy(self._negMatch)
    end
    if self.seqLengths ~= nil then
        return self.sequences[dataIdx], self.posMatches[dataIdx], self.negMatch, self.seqLengths[dataIdx]
    else
        return self.sequences[dataIdx], self.posMatches[dataIdx], self.negMatch
    end

end

function createSeqRankingData(dataPath, savePath, wordVocab, fbVocab, batchSize,relWordVocab,negSize,negWordSize)
    -- class variables
    local posMatches = {}
    local negMatches = {}
    local seqLengths = {}
    local sequences  = {}

    -- read data fileh
    local file = io.open(dataPath, 'r')
    local batchIdx = 0    -- the index of sequence batches
    local seqIdx   = 0    -- sequence index within each batch
    local line
    
    while true do
        line = file:read()
        if line == nil then break end
        local fields = stringx.split(line, '\t')
        
        -- fields[1]: language sequence
        local tokens = stringx.split(fields[1])

        -- allocate tensor memory
        if seqIdx % batchSize == 0 then
            print('batch: '..batchIdx)
            seqIdx = 1
            batchIdx = batchIdx + 1            
            posMatches[batchIdx] = torch.LongTensor(batchSize):fill(relWordVocab.pad_index)
            seqLengths[batchIdx] = torch.LongTensor(batchSize):fill(0)
            sequences [batchIdx] = torch.LongTensor(batchSize, #tokens):fill(wordVocab.pad_index)
            negMatches[batchIdx] = torch.LongTensor(negWordSize,batchSize, negSize):fill(relWordVocab.pad_index)
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
        -- posMatches[batchIdx][seqIdx] = fbVocab:index(fields[2])
        posMatches[batchIdx][seqIdx] = tonumber(fields[2])

    end
    file:close()
    table.remove(posMatches,batchIdx)
    table.remove(seqLengths,batchIdx)
    table.remove(sequences,batchIdx)

    local data = {}
    data.pos = posMatches
    data.len = seqLengths
    data.seq = sequences

    torch.save(savePath, data)
end
