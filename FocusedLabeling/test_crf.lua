require '../init.lua'
function test_model(model)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Training a Recurrent Neural Network to classify a sequence of words')
    cmd:text()
    cmd:text('Comandline Options')

    cmd:option('-wordVocab','../vocab/vocab.word.t7','training data file')
    cmd:option('-testData','../data/valid.focused_labeling.t7','training data file')
    -- cmd:option('-modelFile','model.BiGRU','filename for loading trained model')

    cmd:option('-useGPU',1,'which GPU is used for computation')

    cmd:text()

    ----------------------------- Basic Options -----------------------------

    local opt = cmd:parse(arg)
    local flog = logroll.print_logger()

    local wordVocab = torch.load(opt.wordVocab)

    if opt.useGPU > 0 then
        require 'cutorch'
        require 'cunn'
        cutorch.setDevice(opt.useGPU)
        torch.setdefaulttensortype('torch.CudaTensor')
        flog.info(string.rep('-', 50))
        flog.info('Set default tensor type to CudaTensor')
    end

    ----------------------------- Data Loader -----------------------------
    local loader = SeqLabelingLoader(opt.testData, flog)

    -------------------------- Load & Init Models -------------------------
    if type(model)=='string' then
        model = torch.load(model)
    end
    local seqModel = model.seqModel
    local linearCRF = model.linearCRF
    seqModel:evaluate()
    linearCRF:evaluate()

    ----------------------------- Prediction -----------------------------
    local maxIters = loader.numBatch
    flog.info(string.rep('-', 40))
    flog.info('Begin Prediction')

    local sumPred, sumCorr, sumTrue = 0, 0, 0
    local count = 0

    for i = 1, maxIters do
        xlua.progress(i, maxIters)

        ----------------------- load minibatch ------------------------
        local seq, labels = loader:nextBatch(1)
        local currSeqLen = seq:size(1)

        local seqVec = seqModel:forward(seq)
        local predict = linearCRF:forward(seqVec)
        -- if torch.sum(torch.ne(predict, labels)) == 0 then
        --     count = count + 1
        -- end
        count = count + torch.ne(predict,labels):sum(1):eq(0):sum()
        local maskPred = torch.eq(predict, 2)
        local maskTrue = torch.eq(labels, 2)
        sumCorr = sumCorr + torch.eq(predict:type(torch.type(labels)), labels):cmul(maskTrue):sum()
        sumTrue = sumTrue + maskTrue:sum()
        sumPred = sumPred + maskPred:sum()
        -- if i == loader.numBatch then
        --     print (predict)
        -- end
        -- for i = 1, currSeqLen do
        --     print(string.format("%15s\t%1d\t%1d", wordVocab:token(seq[{i,1}]), predict[{i,1}], labels[{i,1}]))
        -- end
    end
    seqModel:training()
    linearCRF:training()
    print(count)
    local p, r = sumCorr / sumPred, sumCorr / sumTrue
    local strict = (count)/(loader.numBatch*loader.batchSize)
    print(p, r, 2 * p * r / (p + r))
    print(strict)
    return strict
end

-- test_model('/home/yxy/base/CFO/CRF/model.BiGRU.1.480')
