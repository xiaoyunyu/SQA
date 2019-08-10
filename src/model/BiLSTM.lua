local BiGRU, parent = torch.class('BiLSTM', 'BiRNN')

-- initialize the module
function BiLSTM:__init(config)
    parent.__init(self)

    -- config the model
    self.inputSize   = config.inputSize
    self.hiddenSize  = config.hiddenSize
    self.maxSeqLen   = config.maxSeqLen or 200
    self.maxBatch    = config.maxBatch  or 128

    -- allocate weights memory
    self.weight     = torch.Tensor(self.inputSize, self.hiddenSize*8):uniform(-1.0, 1.0)
    self.gradWeight = torch.Tensor(self.inputSize, self.hiddenSize*8):zero()
    
    self.bias     = torch.Tensor(self.hiddenSize*8):uniform(-1.0, 1.0)
    self.gradBias = torch.Tensor(self.hiddenSize*8):zero()

    self.recWeight_G     = torch.Tensor(2, self.hiddenSize, self.hiddenSize*4):uniform(-1.0, 1.0)
    self.gradRecWeight_G = torch.Tensor(2, self.hiddenSize, self.hiddenSize*4):zero()

    -- self.recWeight_H     = torch.Tensor(2, self.hiddenSize, self.hiddenSize):uniform(-1.0, 1.0)
    -- self.gradRecWeight_H = torch.Tensor(2, self.hiddenSize, self.hiddenSize):zero()
    
    -- allocate working memory
    self.gates  = torch.Tensor(self.maxSeqLen, self.maxBatch, self.hiddenSize*8):zero()
    self.state = torch.Tensor(self.maxSeqLen, self.maxBatch, self.hiddenSize*2):zero()
    self.hidden = torch.Tensor(self.maxSeqLen, self.maxBatch, self.hiddenSize*2):zero()
    self.sstate = torch.Tensor(self.maxSeqLen, self.maxBatch, self.hiddenSize*2):zero()

    self.gradGates  = torch.Tensor(self.maxSeqLen, self.maxBatch, self.hiddenSize*8):zero()
    self.gradState = torch.Tensor(self.maxSeqLen, self.maxBatch, self.hiddenSize*2):zero()

    self.gradInput  = torch.Tensor(self.maxSeqLen, self.maxBatch, self.inputSize *2):zero()

    self.buffer = torch.ones(self.maxSeqLen*self.maxBatch)

    -- logging information
    if config.logger then
        config.logger.info(string.rep('-', 50))
        config.logger.info('BiLSTM Configuration:')
        config.logger.info(string.format('    inputSize   : %5d', self.inputSize))
        config.logger.info(string.format('    hiddenSize  : %5d', self.hiddenSize))
        config.logger.info(string.format('    maxSeqLen   : %5d', self.maxSeqLen))
        config.logger.info(string.format('    maxBatch    : %5d', self.maxBatch))
    end

end

function BiLSTM:updateOutput(input)
    assert(self.inputSize==input:size(3), 'Input size not match')
    local seqLen, batchSize = input:size(1), input:size(2)

    self.gates:resize (seqLen, batchSize, self.hiddenSize*8)
    self.state:resize(seqLen, batchSize, self.hiddenSize*2)
    self.hidden:resize(seqLen, batchSize, self.hiddenSize*2)
    self.sstate:resize(seqLen, batchSize, self.hiddenSize*2)

    self.buffer:resize(seqLen*batchSize)

    self.state:fill(0)
    
    local denseInput = input:view(seqLen*batchSize, self.inputSize)
    local denseGates = self.gates:view(seqLen*batchSize, self.hiddenSize*8)

    denseGates:addr(0, 1, self.buffer, self.bias)
    denseGates:addmm(1, denseInput, self.weight)

    for i = 1, self.nStream do
        -- set stream: stream 1 deals with forward-GRU & stream 2 deals with backward-GRU
        if cutorch then cutorch.setStream(i) end

        -- get traverse order (depends on the stream)
        local begIdx, endIdx, stride = self:traverseOrder(seqLen, i)

        -- compute stream memory offset
        local left, right = (i-1)*self.hiddenSize, i*self.hiddenSize

        local prevHidden
        local prevState

        for seqIdx = begIdx, endIdx, stride do
            -- get current memory
            local currGates  = self.gates [{seqIdx, {}, {4*left+1, 4*right}}]
            local currState = self.state[{seqIdx, {}, {  left+1,   right}}]
            local currHidden = self.hidden[{seqIdx, {}, {  left+1,   right}}]
            local currSstate = self.sstate[{seqIdx, {}, {  left+1,   right}}]

            -- decompose currGates
            local fGate = currGates[{{}, {                  1,   self.hiddenSize}}]
            local iGate  = currGates[{{}, {  self.hiddenSize+1, 2*self.hiddenSize}}]
            local oGate = currGates[{{}, {2*self.hiddenSize+1, 3*self.hiddenSize}}]
            local cGate = currGates[{{}, {3*self.hiddenSize+1, 4*self.hiddenSize}}]

            -- recurrent connection
            if seqIdx ~= begIdx then
                currGates:addmm(1, prevHidden, self.recWeight_G[i])
            end        
            
            -- inplace non-linearity for reset & update (both) gates
            -- bothGates.nn.Sigmoid_forward(bothGates, bothGates)
            currGates.THNN.Sigmoid_updateOutput(currGates:cdata(), currGates:cdata())

            if seqIdx ~= begIdx then
                currState:cmul( prevState, fGate)
            end

            currState:addcmul( iGate, cGate)
            currState.THNN.Sigmoid_updateOutput(currState:cdata(), currSstate:cdata())

            -- currect hidden
            currHidden:cmul(oGate, currSstate)

            -- set prev hidden
            prevHidden = currHidden
            prevState = currState
        end
    end

    if cutorch then
        -- set back the stream to default stream (0):
        cutorch.setStream(0)

        -- 0 is default stream, let 0 wait for the 2 streams to complete before doing anything further
        cutorch.streamWaitFor(0, self.streamList)
    end

    self.output = self.hidden
    return self.output
end

function BiGRU:updateGradInput(input, gradOutput)
    assert(self.hiddenSize*2==gradOutput:size(gradOutput:nDimension()), 'gradOutput size not match')
    assert(input:size(1)==gradOutput:size(1) and input:size(2)==gradOutput:size(2), 'gradOutput and input size not match')
    
    local seqLen, batchSize = input:size(1), input:size(2)

    self.gradInput:resize (seqLen, batchSize, self.inputSize)
    self.gradGates:resize (seqLen, batchSize, self.hiddenSize*8)
    self.gradState:resize(seqLen, batchSize, self.hiddenSize*2)

    self.gradGates[1]:fill(0)
    self.gradGates[seqLen]:fill(0)

    for i = 1, self.nStream do
        -- set stream: stream 1 deals with forward-GRU & stream 2 deals with backward-GRU
        if cutorch then cutorch.setStream(i) end

        -- get traverse order (depends on the stream)
        local begIdx, endIdx, stride = self:traverseOrder(seqLen, i)

        -- compute stream memory offset
        local left, right = (i-1)*self.hiddenSize, i*self.hiddenSize

        local prevHidden,prevState, prevGradOutput
        local currGradSstate =torch.Tensor (batchSize, self.hiddenSize)
        for seqIdx = endIdx, begIdx, -stride do
            -- get current memory
            local currGates  = self.gates [{seqIdx, {}, {4*left+1, 4*right}}]
            local currState = self.state[{seqIdx, {}, {  left+1,   right}}]
            local currSstate = self.sstate[{seqIdx, {}, {  left+1,   right}}]
            local currHidden = self.hidden[{seqIdx, {}, {  left+1,   right}}]

            local currGradGates  = self.gradGates [{seqIdx, {}, {4*left+1, 4*right}}]
            local currGradState = self.gradState[{seqIdx, {}, {  left+1,   right}}]
            
            local currGradOutput = gradOutput     [{seqIdx, {}, {  left+1,   right}}]

            -- decompose currGates
            local fGate = currGates[{{}, {                  1,   self.hiddenSize}}]
            local iGate  = currGates[{{}, {  self.hiddenSize+1, 2*self.hiddenSize}}]
            local oGate = currGates[{{}, {2*self.hiddenSize+1, 3*self.hiddenSize}}]
            local cGate = currGates[{{}, {3*self.hiddenSize+1, 4*self.hiddenSize}}]
            local iocGate = currGates[{{}, {self.hiddenSize+1, 4*self.hiddenSize}}]

            local gradfGate = currGradGates[{{}, {                  1,   self.hiddenSize}}]
            local gradiGate  = currGradGates[{{}, {  self.hiddenSize+1, 2*self.hiddenSize}}]
            local gradoGate = currGradGates[{{}, {2*self.hiddenSize+1, 3*self.hiddenSize}}]
            local gradcGate  = currGradGates[{{}, {  3*self.hiddenSize+1, 4*self.hiddenSize}}]
            local gradiocGate = currGradGates[{{}, {  self.hiddenSize+1, 4*self.hiddenSize}}]
            -- pre-gate input: d_h[t] / d_title{h}[t]
            gradoGate:cmul(currGradOutput, currSstate)
            currGradSstate:cmul(currGradOutput, oGate)
            currGradSstate.THNN.Sigmoid_updateGradInput(currSstate:cdata(), currGradSstate:cdata(), currGradSstate:cdata(), currSstate:cdata()) -- inplace
            currGradState:add(currGradSstate)
            gradiGate:cmul(currGradState,cGate)
            gradcGate:cmul(currGradState,iGate)
            -- related to prev hidden
            if seqIdx ~= begIdx then
                prevGradState = self.gradState[{seqIdx-stride, {}, {left+1, right}}]
                prevGradState:cmul(currGradState,fGate)
                -- set prev hidden
                prevHidden = self.hidden[{seqIdx-stride, {}, {left+1, right}}]
                prevState = self.state[{seqIdx-stride, {}, {left+1, right}}]

                gradfGate:cmul(currGradState,prevState)
                currGradGates.THNN.Sigmoid_updateGradInput(currGates:cdata(), currGradGates:cdata(), currGradGates:cdata(), currGates:cdata())
            else
                gradiocGate.THNN.Sigmoid_updateGradInput(iocGate:cdata(), gradiocGate:cdata(), gradiocGate:cdata(), iocGate:cdata())
            end

            if seqIdx ~= begIdx then
                -- set prev grad hidden/output
                prevGradOutput = gradOutput[{seqIdx-stride, {}, {left+1, right}}]
                
                -- prev hidden: d_h[t] / d_h[t-1]
                prevGradOutput:addmm(1, currGradGates, self.recWeight_G[i]:t())

                -- d_h[t] / d_recWeight_G
                self.gradRecWeight_G[i]:addmm(1, prevHidden:t(), currGradGates)
            end
        end
    end

    if cutorch then
        -- set back the stream to default stream (0):
        cutorch.setStream(0)

        -- 0 is default stream, let 0 wait for the 2 streams to complete before doing anything further
        cutorch.streamWaitFor(0, self.streamList)
    end

    local denseInput     = input:view(seqLen*batchSize, self.inputSize)
    local denseGradInput = self.gradInput:view(seqLen*batchSize, self.inputSize)
    local denseGradGates = self.gradGates:view(seqLen*batchSize, self.hiddenSize*8)

    -- d_E / d_input
    denseGradInput:mm(denseGradGates, self.weight:t())

    -- d_E / d_W
    self.gradWeight:addmm(1, denseInput:t(), denseGradGates)

    -- d_E / d_b
    self.gradBias:addmv(1, denseGradGates:t(), self.buffer)

    return self.gradInput
end

function BiGRU:parameters()
    return {self.weight, self.recWeight_G,self.bias}, {self.gradWeight, self.gradRecWeight_G, self.gradBias}
end
