local Vocab = torch.class('Vocab')

function Vocab:__init(path)
    self.size = 0
    self._index = {}
    self._tokens = {}

    local file = io.open(path)
    while true do
        local line = file:read()
        if line == nil then break end
        self.size = self.size + 1
        self._tokens[self.size] = line
        self._index[line] = self.size
    end
    file:close()

    print('vocab size: '..self.size)
end

function Vocab:contains(w)
    if not self._index[w] then return false end
    return true
end

function Vocab:add(w)
    if self._index[w] ~= nil then
        return self._index[w]
    end
    self.size = self.size + 1
    self._tokens[self.size] = w
    self._index[w] = self.size
    return self.size
end

function Vocab:index(w)
    local index = self._index[w]
    if index == nil then
        if self.unk_index == nil then
            error('Token not in vocabulary and no UNK token defined: ' .. w)
        end
        return self.unk_index
    end
    return index
end

function Vocab:token(i)
    if i < 1 or i > self.size then
        error('Index ' .. i .. ' out of bounds')
    end
    return self._tokens[i]
end

function Vocab:map(tokens)
    local len = #tokens
    local output = torch.IntTensor(len)
    for i = 1, len do
        output[i] = self:index(tokens[i])
    end
    return output
end

function Vocab:add_unk_token()
    if self.unk_token ~= nil then return end
    self.unk_index = self:add('<unk>')
    print('vocab size: '..self.size)
end

function Vocab:add_X_token()
    self.X_index = self:add('X')
    print('vocab size: '..self.size)
end

function Vocab:add_pad_token()
    if self.pad_token ~= nil then return end
    self.pad_index = self:add('<pad>')
    print('vocab size: '..self.size)
end

function Vocab:add_ent_token()
    if self.ent_token ~= nil then return end
    self.ent_index = self:add('<entity>')
    print('vocab size: '..self.size)
end

function Vocab:add_start_token()
    if self.start_token ~= nil then return end
    self.start_index = self:add('<s>')
    print('vocab size: '..self.size)
end

function Vocab:add_end_token()
    if self.end_token ~= nil then return end
    self.end_index = self:add('</s>')
    print('vocab size: '..self.size)
end

function Vocab:add_space_token()
    if self.space_token ~= nil then return end
    self.space_index = self:add('<_>')
    print('vocab size: '..self.size)
end

function Vocab:add_word_token(word_vocab,rel2wordPath,num)--7523
	self._index2word = {}
	self._token2word = {}
    if rel2wordPath then
        local rel2word = io.open(rel2wordPath,'r')
        local lookupTable = torch.Tensor(num+1,9*9)
        local i = 0
    	while true do
            local line = rel2word:read()
            if line == nil then break end
    		local words = stringx.split(line,' ')
            i = i + 1
    		local indexs = {}
    		for ind,word in ipairs(words) do
    			indexs[ind] = word_vocab:index(word)
    			assert(indexs[ind] ~= nil)
    		end
            local wordTensor = torch.Tensor(#indexs):fill(word_vocab.pad_index)
            wordTensor[{{1,#indexs}}]=torch.Tensor(indexs)
            wordTensor = wordTensor:repeatTensor(9)
            wordTensor = wordTensor[{{1,9}}]
    		self._index2word[i] = wordTensor
    		self._token2word[self._tokens[i]] = wordTensor
            lookupTable[i] = wordTensor
    	end
        lookupTable[num+1]:fill(word_vocab.pad_index)
        print('saving rel2word')
        torch.save('rel2word.t7',lookupTable)
    else 
        local rel2word = io.open('../KnowledgeBase/FB5M.rel.txt','r')
        local lookupTable = torch.Tensor(num+1,9)
        local i = 0
        while true do
            local line = rel2word:read()
            if line == nil then break end
            local words = stringx.split(line,'.')
            words = stringx.split(words[#words],'_')
            i = i + 1
            local indexs = {}
            for ind,word in ipairs(words) do
                indexs[ind] = word_vocab:index(word)
                assert(indexs[ind] ~= nil)
            end
            -- local wordTensor = torch.Tensor(#indexs):fill(word_vocab.pad_index)
            -- wordTensor[{{1,#indexs}}]=torch.Tensor(indexs)
            -- wordTensor = wordTensor:repeatTensor(9)
            -- wordTensor = wordTensor[{{1,9}}]
            local wordTensor = torch.Tensor(9):fill(word_vocab.pad_index)
            wordTensor[{{1,#indexs}}]=torch.Tensor(indexs)
            self._index2word[i] = wordTensor
            self._token2word[self._tokens[i]] = wordTensor
            lookupTable[i] = wordTensor
        end
        lookupTable[num+1]:fill(word_vocab.pad_index)
        print('saving rel2word')
        torch.save('rel2word.t7',lookupTable)
    end
end

function Vocab:index2word(ind)
    local words = self._index2word[ind]

    return words
end

function Vocab:token2word(token)
    local words = self._token2word[token]

    return words
end

function Vocab:add_syn_token(word_vocab,rel2synPath,num)--7523
    self._index2syn = {}
    self._token2syn = {}

    local rel2syn = io.open(rel2synPath,'r')
    local lookupTable = torch.Tensor(num+1,9)
    local i = 0
    while true do
        local line = rel2syn:read()
        if line == nil then break end
        local words = stringx.split(line,'\t')
        i = i + 1
        local indexs = {}
        for ind,word in ipairs(words) do
            indexs[ind] = word_vocab:index(word)
            assert(indexs[ind] ~= nil)
        end
        local wordTensor = torch.Tensor(9):fill(word_vocab.pad_index)
        wordTensor[{{1,#indexs}}]=torch.Tensor(indexs)
        self._index2syn[i] = wordTensor
        self._token2syn[self._tokens[i]] = wordTensor
        lookupTable[i] = wordTensor
    end
    lookupTable[num+1]:fill(word_vocab.pad_index)
    print('saving rel2syn')
    torch.save('rel2syn.t7',lookupTable)
end

function Vocab:index2syn(ind)
    local words = self._index2syn[ind]

    return words
end

function Vocab:token2syn(token)
    local words = self._token2syn[token]

    return words
end