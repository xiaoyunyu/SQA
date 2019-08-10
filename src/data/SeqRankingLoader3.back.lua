local SeqRankingLoader = torch.class('SeqRankingLoader')

function SeqRankingLoader:__init(datafile, negRange, logger,random_generator)
    -- sequence & pos match
    local data = torch.load(datafile)
    self.sequences  = data.seq
    self.totMatches_w = data.relWord
    self.totMatches_r = data.rel
    self.random_generator = random_generator

    -- additional variables
    self.batchSize = self.sequences[1]:size(2)
    if random_generator==0 then
        self.negSize = self.totMatches_w[1]:size(2)/self.batchSize-1
    else
        self.negSize = random_generator
    end
    self.relWordLen = self.totMatches_w[1]:size(1)
    self.numBatch  = #self.sequences
    self.currIdx   = 1
    self.indices   = randperm(self.numBatch)
    self.negRange = negRange
    self._negMatch = torch.LongTensor(self.random_generator,self.batchSize)
    self._posMatch = torch.LongTensor(1,self.batchSize):expand(self.random_generator,self.batchSize)
    self.totMatches_r_rand=torch.CudaTensor(self.random_generator+1,self.batchSize)
    if torch.Tensor():type() == 'torch.CudaTensor' then
        for i = 1, self.numBatch do
            self.sequences[i]  = self.sequences[i]:cuda()            
            self.totMatches_w[i] = self.totMatches_w[i]:cuda()
            self.totMatches_r[i] = self.totMatches_r[i]:cuda()
        end    
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
    if self.random_generator==0 then
        return self.sequences[dataIdx], self.totMatches_w[dataIdx], self.totMatches_r[dataIdx]

    else
        
        self._posMatch[{{1},{}}]=self.totMatches_r[dataIdx][{{1},{}}]:type('torch.LongTensor')
        self._negMatch:random(1, self.negRange)

        while torch.sum(torch.eq(self._negMatch, self._posMatch)) > 0 do
            self._negMatch:maskedFill(torch.eq(self._negMatch, self._posMatch), math.random(1, self.negRange))
        end
        self.totMatches_r_rand[{{1},{}}]=self._posMatch[{{1},{}}]
        self.totMatches_r_rand[{{2,self.random_generator+1},{}}]=self._negMatch
        return self.sequences[dataIdx],self.totMatches_r_rand
    end
end

function createSeqRankingData(dataPath, savePath, wordVocab, fbVocab, batchSize,relWordVocab,negSize,relWordLen, max_pos)
    -- class variables
    local totMatches_w = {}
    local totMatches_r = {}
    local pos_r = {}
    local sequences  = {}

    -- read data fileh
    local file = io.open(dataPath, 'r')
    local batchIdx = 0    -- the index of sequence batches
    local seqIdx   = 0    -- sequence index within each batch
    local line
    local ave_qu_len = 0
    local tot_len =  0
    local longest_qu_len = 0
    while true do
        line = file:read()
        if line == nil then break end
        tot_len = tot_len+1
        local fields = stringx.split(line, '\t')
        
        -- fields[1]: language sequence
        local tokens = stringx.split(fields[1])
        ave_qu_len = ave_qu_len + #tokens
        longest_qu_len = math.max(longest_qu_len,#tokens)
        -- allocate tensor memory
        if seqIdx % batchSize == 0 then
            -- print('batch: '..batchIdx)
            seqIdx = 1
            batchIdx = batchIdx + 1            

            pos_r[batchIdx] = torch.LongTensor(batchSize, negSize+1, relWordLen):fill(max_pos)
            sequences [batchIdx] = torch.LongTensor(batchSize, #tokens):fill(wordVocab.pad_index)
            totMatches_w[batchIdx] = torch.LongTensor(batchSize, negSize+1, relWordLen):fill(relWordVocab.pad_index)
            totMatches_r[batchIdx] = torch.LongTensor(batchSize, negSize+1):fill(fbVocab.unk_index)
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
        if posMatch ~= fbVocab.unk_index then
            totMatches_w[batchIdx][seqIdx][1][{{1,posWord:size(1)}}] = posWord
            totMatches_r[batchIdx][seqIdx][1] = posMatch
            pos_r[batchIdx][seqIdx][1][{{1,rel_word_len}}] = torch.range(1,rel_word_len)
        end
        -- fields[3]: negtive match
        if #fields > 2 and (#fields-2) < negSize then--有负例  
            for i = 3,#fields,1 do
                -- local negMatch = fbVocab:index(fields[i])
                local negMatch = tonumber(fields[i])
                if not negMatch then
                    negMatch=fbVocab:index(fields[i])
                end
                local negWord = fbVocab:index2word(negMatch)
                local rel_word_len = torch.ne(negWord,relWordVocab.pad_index):sum()
                if negWord then
                    totMatches_w[batchIdx][seqIdx][i-1][{{1,negWord:size(1)}}] = negWord
                    totMatches_r[batchIdx][seqIdx][i-1] = negMatch
                    pos_r[batchIdx][seqIdx][i-1][{{1,rel_word_len}}] = torch.range(1,rel_word_len)
                end
            end
            if #fields-2 < negSize then
                for i = #fields-1, negSize do
                    local negMatch = math.random(1,fbVocab.size-1)
                    while(negMatch==posMatch) do
                        negMatch = math.random(1,fbVocab.size-1)
                    end
                    local negWord = fbVocab:index2word(negMatch)
                    local rel_word_len = torch.ne(negWord,relWordVocab.pad_index):sum()
                    if negMatch ~= fbVocab.unk_index then
                        totMatches_w[batchIdx][seqIdx][i+1][{{1,negWord:size(1)}}] = negWord
                        totMatches_r[batchIdx][seqIdx][i+1] = negMatch
                        pos_r[batchIdx][seqIdx][i+1][{{1,rel_word_len}}] = torch.range(1,rel_word_len)
                    end
                end
            end
        elseif #fields > 2 and (#fields-2) >= negSize then
            for i = 3,negSize+2,1 do
                -- local negMatch = fbVocab:index(fields[i])
                local negMatch = tonumber(fields[i])
                if not negMatch then
                    negMatch=fbVocab:index(fields[i])
                end

                local negWord = fbVocab:index2word(negMatch)
                local rel_word_len = torch.ne(negWord,relWordVocab.pad_index):sum()
                if negWord then
                    totMatches_w[batchIdx][seqIdx][i-1][{{1,negWord:size(1)}}] = negWord
                    totMatches_r[batchIdx][seqIdx][i-1] = negMatch
                    pos_r[batchIdx][seqIdx][i-1][{{1,rel_word_len}}] = torch.range(1,rel_word_len)
                end
            end

        else
            for i = 1, negSize do
                local negMatch = math.random(1,fbVocab.size-1)
                while(negMatch==posMatch) do
                    negMatch = math.random(1,fbVocab.size-1)
                end
                local negWord = fbVocab:index2word(negMatch)
                local rel_word_len = torch.ne(negWord,relWordVocab.pad_index):sum()
                if negMatch ~= fbVocab.unk_index then
                    totMatches_w[batchIdx][seqIdx][i+1][{{1,negWord:size(1)}}] = negWord
                    totMatches_r[batchIdx][seqIdx][i+1] = negMatch
                    pos_r[batchIdx][seqIdx][i+1][{{1,rel_word_len}}] = torch.range(1,rel_word_len)
                end
            end
        end
    end
    file:close()
    ave_qu_len = ave_qu_len/tot_len
    print('average question length is '..ave_qu_len)
    print('longest question length is '..longest_qu_len)
    print('batch: '..batchIdx)
    for idx = 1, batchIdx do
        sequences[idx] = sequences[idx]:transpose(1,2)
        totMatches_w[idx] = totMatches_w[idx]:transpose(1,3):contiguous():view(relWordLen,-1)
        totMatches_r[idx] = totMatches_r[idx]:transpose(1,2)
        pos_r[idx] = pos_r[idx]:transpose(1,3):contiguous():view(relWordLen,-1)
    end

    local data = {}
    -- table.remove(seqLengths,batchIdx)
    -- table.remove(sequences,batchIdx)
    -- table.remove(totMatches_w,batchIdx)
    -- table.remove(totMatches_r,batchIdx)

    -- if #seqLengths%2 ==1 then
    --         table.remove(seqLengths,#seqLengths)
    --     table.remove(sequences,#seqLengths)
    --     table.remove(totMatches_w,#seqLengths)
    --     table.remove(totMatches_r,#seqLengths)
    -- end

    -- data.pos = posMatches
    -- data.pos_r = pos_r
    data.seq = sequences
    data.relWord = totMatches_w
    data.rel = totMatches_r

    torch.save(savePath, data)
end
