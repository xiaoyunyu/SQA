--******************2RNN+kernel pool+relation word*************************

function testfunc(opt,model,loader,num)


   -- modelFile='model.rel.stackBiRNN.1.'..epoch*100
-- parse input params
    --local opt = cmd:parse(arg)
    --local flog = logroll.print_logger()

    -- if opt.useGPU > 0 then
    --     require 'cutorch'
    --     require 'cunn'
    --     cutorch.setDevice(opt.useGPU)
    --     torch.setdefaulttensortype('torch.CudaTensor')    
    -- end

    -- load all models
    -- local suffix = stringx.split(modelFile, '.')
    -- suffix = suffix[#suffix]
    --local model = torch.load(modelFile)

    -- init data loader and output files
    --local loader = RankingDataLoader(opt.testData, flog)
    -- local score_file = io.open(string.format('score.%s.%s.%s',opt.KG,stringx.split(opt.testData,'.')[-4], suffix), 'w')
    -- local rank_file  = io.open(string.format('rank.%s.%s.%s', opt.KG,stringx.split(opt.testData,'.')[-4], suffix), 'w')

    -- extract sub models
    local seqModel = model.seqModel
    local relWordBranch = model.relWordBranch
    local relBranch = model.relBranch
    local relCombLayer = model.relCombLayer
    local scoreModel = TripleScore_vec4test(loader.negSize+1)

    local flog = loader.logger
    local loss = 0
    local count = 0 
    local stastic = {}
    local lastBatchlen = loader.lastBatchlen
    if lastBatchlen == 0 then
    	lastBatchlen = loader.batchSize
    end

    seqModel:evaluate()
    relWordBranch:evaluate()
    relBranch:evaluate()
    relCombLayer:evaluate()
    scoreModel:evaluate()

    -- core testing loop
    --for i = 1, loader.numBatch do
    for i = 1, loader.numBatch do
        xlua.progress(i, loader.numBatch)
        ----------------------- load minibatch ------------------------
        local seq, rel_w, rel_r = loader:nextBatch(1)
        local negSize = rel_r:size(1)-1
        local seqVec = seqModel:forward(seq)
        --relation word level branch [33*n_batch x n_dim]
        local rel_wMat = relWordBranch:forward(rel_w)
        -- relation level branch [33*n_batch x n_dim]
        local rel_rMat = relBranch:forward(rel_r)
        --relation combination [33*n_batch x n_dim]
        local rel_Mat = relCombLayer:forward({rel_wMat,rel_rMat})
        rel_Mat = rel_Mat:view(-1,loader.batchSize,opt.hiddenSize*2)
        relscores = scoreModel:forward({seqVec,rel_Mat}):squeeze(3):transpose(1,2)

        local negscores = relscores[{{},{2,negSize+1}}]
        local sorted,argSort =negscores:sort(2,true)
        argSort = argSort[{{},{1}}]
        if i==loader.numBatch then
            for idx = 1,lastBatchlen do
                if rel_r[argSort[idx][1]+1][idx] == rel_r[1][idx] then
                    count=count+1
                end
            end
        else
            for idx = 1,loader.batchSize do
                if rel_r[argSort[idx][1]+1][idx] == rel_r[1][idx] then
                    count=count+1
                elseif loader.batchSize == 1 then
                    flog.info(string.format('the wrong problem id is %d', i))
                    table.insert(stastic, i)
                
                end
            end
        end
        collectgarbage()


    end
    -- score_file:close()
    -- rank_file:close()
    seqModel:training()
    relWordBranch:training()
    relBranch:training()
    relCombLayer:training()
    scoreModel:training()

    local totalLen = ((loader.numBatch-1)*loader.batchSize)+lastBatchlen

    flog.info(string.format('********test acc is %f', count/totalLen))
    -- flog.info(string.format('********test loss is %f', loss/loader.numBatch))
    print(string.format('********test acc is %f', count/totalLen))
    if num then
        print(string.format('the batchsize is %d',num))
    end  
    collectgarbage()
    return {count,totalLen}

end

function main()
    require '../init3.lua'
    require 'gnuplot'

    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Training a Recurrent Neural Network to embed a sentence')
    cmd:text()
    cmd:text('Options')
    cmd:option('-kernelNum',4)--k>=3
    cmd:option('-dropoutRate_2',0.1,'dropout rate')
    cmd:option('-dropoutRate_3',0.3,'dropout rate')

    cmd:option('-vocabSize',100002,'number of words in dictionary')

    cmd:option('-relSize',7524,'number of relations in dictionary')
    cmd:option('-relEmbedSize',256,'size of rel embedding')
    cmd:option('-relWordSize',3390,'number of relations in dictionary')
    cmd:option('-initRange_rwl',0.004,'the range of uniformly initialize parameters')
    cmd:option('-initRange_rl',0.005,'the range of uniformly initialize parameters')

    cmd:option('-wrdEmbedSize',300,'size of word embedding')
    cmd:option('-wrdEmbedPath','../vocab/word.glove100k.t7','pretained word embedding path')

    cmd:option('-maxSeqLen',40,'number of timesteps to unroll to')
    cmd:option('-hiddenSize',256,'size of RNN internal state')
    cmd:option('-dropoutRate',0.5,'dropout rate')

    cmd:option('-negSize',32,'number of negtive samples for each iteration')
    cmd:option('-maxEpochs',1000,'number of full passes through the training data')
    cmd:option('-initRange',0.006,'the range of uniformly initialize parameters')
    cmd:option('-costMargin',0.1,'the margin used in the ranking cost')
    cmd:option('-costMargin2',0.2,'the margin used in the ranking cost')
    cmd:option('-useGPU',1,'whether to use gpu for computation')

    cmd:option('-printEvery',100,'how many steps/minibatches between printing out the loss')
    cmd:option('-saveEvery',100,'how many epochs between auto save trained models')
    cmd:option('-saveFile','/home/yxy/base/CFO/relRNN/model.rel.stackBiRNN','filename to autosave the model (protos) to')
    cmd:option('-logFile','logs/rel.stackBiRNN.','log file to record training information')
    cmd:option('-testLogFile','logs/test.','log file to record training information')
    cmd:option('-dataFile', '../data/train.relword.torch','training data file')
    cmd:option('-testData','../data/valid.rel.FB5M.torch','run test on which data set')
    cmd:option('-round',7)
    cmd:option('-testOn',false)

    cmd:option('-seed',123,'torch manual random number generator seed')
    cmd:text()
    local opt = cmd:parse(arg)    
    
    if opt.useGPU > 0 then
        require 'cutorch'
        require 'cunn'
        cutorch.setDevice(opt.useGPU)
        torch.setdefaulttensortype('torch.CudaTensor')    
    end
    local flog = logroll.file_logger(opt.testLogFile..opt.round..'.log')
    local wordVocab = torch.load('../vocab/vocab.word.t7')
    local relationVocab = torch.load('../vocab/vocab.rel.t7')
    local relWordVocab = torch.load('../vocab/vocab.relword.t7')

    for x = 100,100,100 do
        local model = torch.load(opt.saveFile..'.'..opt.round..'.'..x)
            
        --local multitestLoader = RankingDataLoader(opt.multitestData, opt.relSize, flog)
        for bs = 2,16 do
            createRankingData('../dataset/new_test.txt', '../data/valid.rel.FB5M.torch', wordVocab, relationVocab, bs, relWordVocab,150,9)--4)
            local singletestLoader = RankingDataLoader(opt.testData, opt.relSize, flog)
            local singleCount,singleNum = unpack(testfunc(opt,model,singletestLoader,bs))
        end
    end
    --local multiCount,multiNum = unpack(testfunc(opt,model,multitestLoader,num))
    -- print('total test acc is '..(singleCount+multiCount)/(singleNum+multiNum))
    -- print('total number is '..(singleNum+multiNum))

end

-- main()
collectgarbage()

--********************************
-- total test acc is 0.88650488310151   
-- total number is 20274   
--********************************
-- total test acc is 0.87879819145316  
-- total number is 20569   
--********************************