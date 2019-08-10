--*******************************************
-- require '..'
-- local cmd = torch.CmdLine()
-- cmd:text()
-- cmd:text('Training a Recurrent Neural Network to embed a sentence')
-- cmd:text()
-- cmd:text('Options')

-- cmd:option('-vocabSize',100002,'number of words in dictionary')

-- cmd:option('-relSize',7524,'number of relations in dictionary')
-- cmd:option('-relEmbedSize',256,'size of rel embedding')

-- cmd:option('-wrdEmbedSize',300,'size of word embedding')
-- cmd:option('-wrdEmbedPath','../vocab/word.glove100k.t7','pretained word embedding path')

-- cmd:option('-numLayer',2,'number of RNN layers')
-- cmd:option('-maxSeqLen',40,'number of timesteps to unroll to')
-- cmd:option('-hiddenSize',256,'size of RNN internal state')
-- cmd:option('-dropoutRate',0.5,'dropout rate')

-- cmd:option('-negSize',1024,'number of negtive samples for each iteration')
-- cmd:option('-maxEpochs',1000,'number of full passes through the training data')
-- cmd:option('-initRange',0.08,'the range of uniformly initialize parameters')
-- cmd:option('-costMargin',0.1,'the margin used in the ranking cost')
-- cmd:option('-useGPU',1,'whether to use gpu for computation')

-- cmd:option('-printEvery',100,'how many steps/minibatches between printing out the loss')
-- cmd:option('-saveEvery',100,'how many epochs between auto save trained models')
-- cmd:option('-saveFile','model.rel.stackBiRNN','filename to autosave the model (protos) to')
-- cmd:option('-logFile','logs/test.','log file to record training information')
-- cmd:option('-dataFile', '../data/train.relation_ranking.t7','training data file')
-- cmd:option('-singletestData','../data/test.single.label.FB5M.torch','run test on which data set')
-- cmd:option('-multitestData','../data/test.multi.label.FB5M.torch','run test on which data set')
-- cmd:option('-round',0)
-- cmd:option('-testOn',true)

-- cmd:option('-seed',123,'torch manual random number generator seed')
-- cmd:text()

--**********************************************************************

function testfunc(opt,model,loader,criterion,num)


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
    local relEmbed   = model.relEmbed
    local seqModel   = model.seqModel
    local scoreModel = TripleScore(loader.negSize)
--    local negRelDrop = model.negRelDrop
--    local posRelDrop = model.posRelDrop
--    local posNorm = model.posNorm
--    local negNorm = model.negNorm
 
    -- local wordEmbed = model.wordEmbed 
    local flog = loader.logger
    local loss = 0
    local count = 0 
    local lastBatchlen = loader.lastBatchlen
    if lastBatchlen == 0 then
        lastBatchlen = loader.batchSize
    end

    seqModel:evaluate()
    relEmbed:evaluate()
--    posRelDrop:evaluate()
--    negRelDrop:evaluate()
    print(string.format('loader batchsize is %d',loader.batchSize))
    local rel = torch.Tensor(loader.negSize+1,loader.batchSize)
    fo=io.open('predicted_relation.'..opt.round..'.'..opt.iniMethod..'.txt','w')
    -- core testing loop
    for i = 1, loader.numBatch do
        xlua.progress(i, loader.numBatch)
        ----------------------- load minibatch ------------------------
        local seq, pos, negs = loader:nextBatch(1)
        -- neg = neg:view(-1)
        local currSeqLen = seq:size(1)
        seq=seq:transpose(1,2)
        negs = negs:transpose(1,2)
        rel[{{1},{}}]=pos
        rel[{{2,loader.negSize+1},{}}]=negs

        ------------------------ forward pass -------------------------    
        -- sequence vectors [n_batch x n_dim]
        --local wordVec = wordEmbed:forward(seq)
        local seqVec = seqModel:forward(seq)
        local relMat = relEmbed:forward(rel)
        -- positive vectors [n_batch x n_dim]
        local posVec = relMat[{{1},{},{}}]:squeeze(1)
     --   local posDropVec = posRelDrop:forward(posVec)
    --    local posDropVec = posNorm:forward(posDropVec_)

        -- negative matrix  [n_neg x n_batch x n_dim]
        local negMat = relMat[{{2,loader.negSize+1},{},{}}]
    --    local negDropMat = negRelDrop:forward(negMat)
    --    local negDropMat = negNorm:forward(negDropMat_:view(negDropMat_:size(1)
    --        *negDropMat_:size(2),-1))
        -- scores table {[1] = postive_scores, [2] = negative_scores}
        --local iscores = scoreModel:forward({seqVec, posVec, negMat})
        local scores = scoreModel:forward({seqVec, posVec, negMat
    --        :view(negDropMat_:size(1),negDropMat_:size(2),-1)
            })
        Posscores=scores[1]:squeeze(3)
        Negscores=scores[2]:squeeze(3)
        -- sequence matrix  [n_neg x n_batch x n_dim]
        -- local seqMat = torch.repeatTensor(seqVec, negMat:size(1), 1)
        -- print(seqMat:size())

        -- if opt.useGPU > 0 then
        --     print(negMat:size())
        --     Negscores = torch.cmul(seqMat, negMat):sum(2):view(-1)

        --     Posscores = torch.cmul(seqVec, posVec):sum(2):view(-1):repeatTensor(Negscores:size(1)):view(-1)
        -- else
        --     Negscores = torch.mm(seqMat, negMat:t()):diag()
        -- end
        -- local input = {Posscores,Negscores}
        -- local ones = torch.ones(Negscores:size(1))
        -- loss = loss + criterion:forward(input,ones)
        local s,argSort =(Posscores - Negscores) :sort(1,false)
        if i==loader.numBatch then
            for idx = 1,lastBatchlen do
                fo:write(argSort[1][idx])
                fo:write('\t')
                if pos[idx] == negs[argSort[1][idx]][idx] then
                    count = count + 1
                end
            end
        else
            for idx = 1,loader.batchSize do
                fo:write(argSort[1][idx])
                fo:write('\t')
                if pos[idx] == negs[argSort[1][idx]][idx] then
                    count = count + 1
                end
            end
        end
        -- -- write to rank file
        -- if scores:size(1) > 1 then
        --     local _, argSort = scores:sort(1, true)

        --     rank_file:write(pos[1], '\t')
        --     for i = 1, argSort:size(1) do
        --         rank_file:write(neg[argSort[i]], ' ')
        --     end
        --     rank_file:write('\n')

        --     -- write to score file
        --     local topIndices = {}
        --     for i = 1, argSort:size(1) do
        --         topIndices[argSort[i]] = 1
        --     end
        --     for i = 1, scores:size(1) do
        --         if topIndices[i] then
        --             score_file:write(scores[i], ' ')
        --         else
        --             score_file:write(0, ' ')
        --         end
        --     end
        --     score_file:write('\n')
        -- else
        --     rank_file:write(pos[1], '\t')
        --     rank_file:write(neg[1])
        --     rank_file:write('\n')
        --     score_file:write(scores[1])
        --     score_file:write('\n')
        -- end

       --collectgarbage()
    end
    fo:close()
    -- score_file:close()
    -- rank_file:close()
    seqModel:training()
    relEmbed:training()
--    posRelDrop:training()
--    negRelDrop:training()
    flog.info(string.format('********test acc is %f', count/(loader.numBatch*loader.batchSize)))
    flog.info(string.format('********test loss is %f', loss/(loader.numBatch*loader.batchSize)))
    print(string.format('********test acc is %f', count/(loader.numBatch*loader.batchSize)))
    return {count,((loader.numBatch-1)*loader.batchSize)+lastBatchlen}

end

function main()
    require '../init2.lua'
    require 'socket'
    local cmd = torch.CmdLine()
    cmd:option('-updateRel',true,'number of words in dictionary')
    cmd:option('-iniMethod','HLE','number of words in dictionary')

    cmd:option('-addLinear',false,'number of words in dictionary')
    cmd:option('-vocabSize',100003,'number of words in dictionary')

    cmd:option('-relSize',7524,'number of relations in dictionary')
    cmd:option('-relEmbedSize',256,'size of rel embedding')

    cmd:option('-wrdEmbedSize',300,'size of word embedding')
    cmd:option('-wrdEmbedPath','../vocab/word.glove100k.t7','pretained word embedding path')

    cmd:option('-numLayer',2,'number of RNN layers')
    cmd:option('-maxSeqLen',40,'number of timesteps to unroll to')
    cmd:option('-hiddenSize',256,'size of RNN internal state')
    cmd:option('-dropoutRate',0.5,'dropout rate')

    cmd:option('-negSize',1024,'number of negtive samples for each iteration')
    cmd:option('-maxEpochs',1000,'number of full passes through the training data')
    cmd:option('-initRange',0.08,'the range of uniformly initialize parameters')
    cmd:option('-costMargin',0.1,'the margin used in the ranking cost')
    cmd:option('-useGPU',1,'whether to use gpu for computation')
    cmd:option('-printEvery',100,'how many steps/minibatches between printing out the loss')
    cmd:option('-saveEvery',100,'how many epochs between auto save trained models')
    cmd:option('-saveFile','/home/yxy/base/CFO/relRNN/model.rel.stackBiRNN','filename to autosave the model (protos) to')
    cmd:option('-logFile','logs/rel.stackBiRNN.','log file to record training information')
    cmd:option('-dataFile', '../data/train.relword.torch','training data file')
    cmd:option('-testData','../data/test.rel.FB5M.torch','run test on which data set')
    cmd:option('-round','cfo')
    cmd:option('-testOn',true)
    cmd:option('-plt',false)
    cmd:option('-seed',123,'torch manual random number generator seed')
    cmd:text()

    local opt = cmd:parse(arg)    
    
    if opt.useGPU > 0 then
        require 'cutorch'
        require 'cunn'
        cutorch.setDevice(opt.useGPU)
        torch.setdefaulttensortype('torch.CudaTensor')    
    end
    local flog = logroll.file_logger(opt.logFile..opt.round..'.log')
    local model = torch.load('/home/yxy/base/CFO/relRNN/model.rel.stackBiRNN.'..opt.iniMethod)
    local criterion = nn.MarginRankingCriterion(opt.costMargin)
    local num = 1 
    local time_start = socket.gettime()
    -------------------------------------------------
    local plotLoss ={}
    for i = 5,250,10 do
        opt.testData = '/home/yxy/base/CFO/data/test.rel.FB5M.'..i..'.light.torch'
        local testLoader = RankingDataLoader(opt.testData, flog)
        local singleCount,singleNum = unpack(testfunc(opt,model,testLoader,criterion,num))
        table.insert(plotLoss,singleCount/singleNum)
    end
    local losslines
    xpcall(function(inp) losslines=torch.load(inp) end,function() losslines=-1 end,'compare.negsize') 
    if type(losslines) ~= 'table' then
        losslines={}
    end
    local loss_xline = torch.range(5,250,10)
    local lossline = torch.Tensor(plotLoss)
    local name = opt.iniMethod
    losslines[#losslines+1]={name,loss_xline,lossline}
    torch.save('compare.negsize',losslines)
    gnuplot.pngfigure('compare_negsize'..'.png')
    gnuplot.plot(unpack(losslines))
    -- gnuplot.axis({30,1000,0.87,0.92})
    -- gnuplot.axis({100,1000,0.84,0.88})
    gnuplot.xlabel('negative sampling size')
    gnuplot.ylabel('Accuracy')
    gnuplot.movelegend('left','bottom')
    -- gnuplot.pngfigure('cfoloss.v2'..'.png')
    gnuplot.plotflush()
    --------------------------------------------------
    local time_end = socket.gettime()
    -- local testLoader = RankingDataLoader(opt.testData, flog)
    -- local singleCount,singleNum = unpack(testfunc(opt,model,testLoader,criterion,num))

end

-- main()

--********************************
-- total test acc is 0.88650488310151   
-- total number is 20274   
--********************************
-- total test acc is 0.87879819145316  
-- total number is 20569   
--********************************

    -- require 'cutorch'
    -- require 'gnuplot'
    -- a=torch.load('compare.negsize')
    -- gnuplot.pngfigure('compare_negsize'..'.png')
    -- gnuplot.plot(unpack(a))
    -- -- gnuplot.axis({30,1000,0.87,0.92})
    -- -- gnuplot.axis({100,1000,0.84,0.88})
    -- gnuplot.xlabel('Negative predicate size')
    -- gnuplot.ylabel('Accuracy')
    -- gnuplot.movelegend('right','bottom')
    -- -- gnuplot.pngfigure('cfoloss.v2'..'.png')
    -- gnuplot.plotflush()