local SeqLabelingLoader = torch.class('SeqLabelingLoader')

function SeqLabelingLoader:__init(datafile, logger)
    -- class variables
    local data = torch.load(datafile)
    self.sequences = data.seq


    -- additional variables
    self.batchSize = self.sequences[1]:size(2)
    self.numBatch = #self.sequences    
    self.currIdx = 1
    self.indices = randperm(self.numBatch)

    if torch.Tensor():type() == 'torch.CudaTensor' then
        for i = 1, self.numBatch do
            self.sequences[i]    = self.sequences[i]:cuda()

        end
    end

    if logger then
        self.logger = logger
        self.logger.info(string.rep('-', 50))
        self.logger.info(string.format('SeqLabelingLoader Configurations:'))
        self.logger.info(string.format('    number of batch: %d', self.numBatch))
        self.logger.info(string.format('    data batch size: %d', self.batchSize))
    end
end

function SeqLabelingLoader:nextBatch(circular)
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
    return self.sequences[dataIdx]
end

-- create torch-format data for SeqLabelingLoader
function createSeqLabelingData(dataPath, savePath, wordVocab, batchSize, noneLabel, trueLabel)
    -- class variable
    local sequences = {}

    local noneLabel = noneLabel or 1
    local trueLabel = trueLabel or 2

    -- read data fileh
    local file = io.open(dataPath, 'r')
    local batchIdx = 0    -- the index of sequence batches
    local seqIdx   = 0    -- sequence index within each batch
    local line
    local totLen = 0
    
    while true do
        line = file:read()
        if line == nil then break end
        totLen = totLen + 1
        local fields = stringx.split(line, '\t')
        
        -- fields[1]: language sequence
        local tokens = stringx.split(fields[1])


        -- allocate tensor memory
        if seqIdx % batchSize == 0 then
            print('batch: '..batchIdx)
            seqIdx = 1
            batchIdx = batchIdx + 1
            sequences[batchIdx] = torch.LongTensor(#tokens, batchSize):fill(wordVocab.pad_index)
        else
            seqIdx = seqIdx + 1
        end

        -- parse tokens into table
        for i = 1, #tokens do
            sequences[batchIdx][{i, seqIdx}] = wordVocab:index(tokens[i])
        end

        -- parse labels into table

    end
    file:close()

    local data = {}
    data.seq   = sequences

    torch.save(savePath, data)
    print(string.format('total lenght is %d',totLen))
end