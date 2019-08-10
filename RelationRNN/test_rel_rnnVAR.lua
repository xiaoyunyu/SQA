--******************2RNN+kernel pool+relation word*************************

function testfunc(opt,model,loader,num)
    -- extract sub models
    local seqModel = model.seqModel
    local normLayer = model.normLayer
    local latentLayer = model.latentLayer
    local sampler = model.sampler
    local rel_vlayer = model.rel_vlayer
----------------syn_attn-----------------
    local train_syn_attn = model.syn_attn
    local config_attn = {}
    config_attn.hiddenSize = opt.hiddenSize
    config_attn.relWordLen = opt.relWordLen 
    config_attn.negSize = loader.negSize
    config_attn.dropoutRate_2 = opt.dropoutRate_2 
    local syn_attn = attn_model_single(config_attn)
    local parameters,_ = syn_attn:getParameters()
    local train_parameters,_ = train_syn_attn:getParameters()
    parameters:copy(train_parameters)
-----------------------------------------
    local cnnLayer = model.cnnLayer
    local relBranch = model.relBranch
    local relCombLayer = model.relCombLayer
    local scoreModel = TripleScore_vec4test(loader.negSize+1)
    local relWordEmbedTab = model.relWordEmbedTab
    local reshape_h = nn.Sequential():add(nn.Replicate(loader.negSize+1)):add(nn.Contiguous()):add(nn.View(-1,opt.hiddenSize))
    
    local flog = loader.logger
    local loss = 0
    local count = 0 
    local stastic = {}
    local lastBatchlen = loader.lastBatchlen
    local preCal,preCal_embed
    local seqModel = model.seqModel
    local usePreCal = false
    if usePreCal then
        preCal = torch.load('preCal4rel.torch')
        preCal_embed = cudacheck(nn.LookupTable(opt.relSize,opt.hiddenSize))
        preCal_embed.weight = preCal
    end
    if lastBatchlen == 0 then
    	lastBatchlen = loader.batchSize
    end
    local savePreCal = false
    if savePreCal then
        preCal = torch.Tensor(opt.relSize,opt.hiddenSize)
        preCal:uniform(-1,1)
    end
    seqModel:evaluate()
    normLayer:evaluate()
    latentLayer:evaluate()
    sampler:evaluate()
    rel_vlayer:evaluate()
    syn_attn:evaluate()
    cnnLayer:evaluate()
    relBranch:evaluate()
    relCombLayer:evaluate()
    scoreModel:evaluate()
    relWordEmbedTab:evaluate()
    reshape_h:evaluate()

    -- core testing loop
    --for i = 1, loader.numBatch do
    fo=io.open('predicted_relation.txt','w')
    -- local time_start = socket.gettime()
    for i = 1, loader.numBatch do
        xlua.progress(i, loader.numBatch)
        ----------------------- load minibatch ------------------------
        --rel_r:[negsize * batchsize]
        local seq, rel_w, rel_r,rel_s = loader:nextBatch(1)
        rel_w=rel_w:transpose(1,2)
        rel_s=rel_s:transpose(1,2)
        -- if i == 15000 then
        -- 	print(rel_r)
        -- end
        --print(rel_r:size())
        --print(rel_r[{{},{127}}])
        --print(rel_r)--1 4
        local negSize = rel_r:size(1)-1
        ------------------------ forward pass -------------------------    
        -- sequence vectors [n_batch x n_dim]
        local seqVec = seqModel:forward(seq)
        local normVec = normLayer:forward(seqVec)
        local mean, log_var = unpack(latentLayer:forward(seqVec))
        local relscores = torch.zeros(loader.batchSize,negSize+1)

        local rel_em_tab = relWordEmbedTab:forward({rel_w,rel_s})
        local rel_em = rel_em_tab[1]
        local syn_em = rel_em_tab[2]
        local rel_wMat = cnnLayer:forward(rel_em)
        local rel_sMat = rel_vlayer:forward(syn_em)
        local rel_rMat = relBranch:forward(rel_r)
        for i = 1,opt.sample_num do
            local h_ = sampler:forward({mean, log_var})
            local h = reshape_h:forward(h_)
            local rel_Mat
            local rel_sAttn = syn_attn:forward({h,rel_sMat})
            rel_Mat = relCombLayer:forward({rel_wMat,rel_rMat,rel_sAttn})
       
            --print(rel_Mat:size())
            --[batchsize * negsize * hiddensize]
            rel_Mat = rel_Mat:view(-1,loader.batchSize,opt.hiddenSize)

            --print(rel_Mat[5][1]-rel_Mat[5][3])
            -- local posOut = rel_Mat:view(-1,loader.batchSize,opt.hiddenSize):narrow(1,1,1):transpose(1,2)
            -- local negOut = rel_Mat:view(-1,loader.batchSize,opt.hiddenSize):narrow(1,2,negSize):transpose(1,2)
            relscores = relscores + scoreModel:forward({normVec,rel_Mat}):squeeze(3):transpose(1,2)
        end
        -- local relscores = maxLayer_pos:forward(relProd)

        -- local Posscores = tmnLayer_pos:forward(posProd)
        -- local Negscores = tmnLayer_neg:forward(negProd)

        -- local Posscores = maxLayer_pos:forward(posProd)
        -- local Negscores = maxLayer_neg:forward(negProd)

        -- Posscores = Posscores:expand(Posscores:size(1),Negscores:size(2))
        -- local sorted,_ =(Posscores-Negscores):sort(2,false)
        --  count = count + torch.ge(sorted[{{},1}],0):sum()


        -- sequence matrix  [n_neg x n_batch x n_dim]
        --print(relscores[1][100]-relscores[1][110])
        local negscores = relscores[{{},{2,negSize+1}}]
        local sorted,argSort =negscores:sort(2,true)
        argSort = argSort[{{},{1}}]
        if i==loader.numBatch then
            for idx = 1,loader.batchSize do
                fo:write(argSort[idx][1])
                fo:write('\t')
                if rel_r[argSort[idx][1]+1][idx] == rel_r[1][idx] then
                    count=count+1
                end
            end
        else
            for idx = 1,loader.batchSize do
                    fo:write(argSort[idx][1])
                    fo:write('\t')
                if rel_r[argSort[idx][1]+1][idx] == rel_r[1][idx] then
                    count=count+1
                end
                -- elseif loader.batchSize == 1 then
                --     -- flog.info(string.format('the wrong problem id is %d', i))
                --     -- table.insert(stastic, i)
                
                -- end
            end
        end
        collectgarbage()


    end
    -- local time_end = socket.gettime()
    local totalLen = ((loader.numBatch-1)*loader.batchSize)+lastBatchlen
    -- local totalLen = (loader.numBatch*loader.batchSize)
    -- local ave = (time_end - time_start)/totalLen
    -- print(string.format('average time is %f s',ave))
    -- score_file:close()
    -- rank_file:close()
    seqModel:training()
    normLayer:training()
    latentLayer:training()
    sampler:training()
    rel_vlayer:training()
    syn_attn:training()
    cnnLayer:training()
    relBranch:training()
    relCombLayer:training()
    scoreModel:training()
    relWordEmbedTab:training()
    reshape_h:training()

    

    flog.info(string.format('********test acc is %f', count/totalLen))
    -- flog.info(string.format('********test loss is %f', loss/loader.numBatch))
    print(string.format('********test acc is %f', count/totalLen))
    if num then
        print(string.format('the batchsize is %d',num))
    end
  
    collectgarbage()
    if savePreCal then
        torch.save('preCal4rel.torch',preCal)
    end
    return {count,totalLen}

end

function final_testData(path,bs)
    wordVocab = torch.load('../vocab/vocab.word.t7')
    relationVocab = torch.load('../vocab/vocab.rel.t7')
    relWordVocab = torch.load('../vocab/vocab.relword.t7')
    print(wordVocab.size)
    print(relationVocab.size)
    print(relWordVocab.size)
--../Inference/relAN.final.test.txt
    createRankingData(path, '../data/final.valid.rel.FB5M.torch', wordVocab, relationVocab, bs, relWordVocab,199,9,40)--4)

end

function main()
    require '../init3.lua'
    require 'socket'

    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Training a Recurrent Neural Network to embed a sentence')
    cmd:text()
    cmd:text('Options')
    cmd:option('-kernelNum',4)--k>=3
    cmd:option('-dropoutRate_1',0.3,'dropout rate')--*******************************
    cmd:option('-dropoutRate_2',0.1,'dropout rate')--*******************************
    cmd:option('-dropoutRate_r',0.5,'dropout rate')--*******************************
    cmd:option('-sigm',0.3)
    cmd:option('-random_init',false)
    cmd:option('-back','')
    cmd:option('-attn_mode',false)
    cmd:option('-soft',false)
    cmd:option('-soft_update',false)
    ---------------------------------------------------------------
    cmd:option('-syn',true)
    cmd:option('-sample_num',1)
    cmd:option('-eps',0.0005)
    cmd:option('-iniMethod','VA')
    ---------------------------------------------------------------
    cmd:option('-config','')

    cmd:option('-vocabSize',100003,'number of words in dictionary')

    cmd:option('-relSize',7524,'number of relations in dictionary')
    cmd:option('-relWordLen',9,'number of relations in dictionary')
    cmd:option('-relEmbedSize',256,'size of rel embedding')
    cmd:option('-relWordSize',7735,'number of relations in dictionary')
    cmd:option('-rareRelWordSize',210,'number of relations in dictionary')
    cmd:option('-initRange_rwl',0.09,'the range of uniformly initialize parameters')--********
    cmd:option('-initRange_rl',0.1,'the range of uniformly initialize parameters')--*********
    cmd:option('-initRange_sl',0.5,'the range of uniformly initialize parameters')
    cmd:option('-initRange_rqk',0.15,'the range of uniformly initialize parameters')

    cmd:option('-wrdEmbedSize',300,'size of word embedding')
    cmd:option('-wrdEmbedPath','../vocab/word.glove100k.t7','pretained word embedding path')

    cmd:option('-numLayer',2,'number of timesteps to unroll to')
    cmd:option('-num_feat_maps',100,'number of timesteps to unroll to')

    cmd:option('-maxSeqLen',40,'number of timesteps to unroll to')
    cmd:option('-hiddenSize',300,'size of RNN internal state')
    cmd:option('-dropoutRate',0.5,'dropout rate')--********************************************

    cmd:option('-negSize',0,'number of negtive samples for each iteration')
    cmd:option('-maxEpochs',300,'number of full passes through the training data')
    cmd:option('-initRange',0.006,'the range of uniformly initialize parameters')--************
    cmd:option('-costMargin',0.1,'the margin used in the ranking cost')
    cmd:option('-useGPU',1,'whether to use gpu for computation')

    cmd:option('-printEvery',200,'how many steps/minibatches between printing out the loss')
    cmd:option('-saveEvery',100,'how many epochs between auto save trained models')
    cmd:option('-saveFile','/home/yxy/base/CFO/relRNN/model.rel.stackBiRNN','filename to autosave the model (protos) to')
    cmd:option('-logFile','logs/rel.stackBiRNN.','log file to record training information')
    cmd:option('-testLogFile','logs/test.','log file to record training information')
    cmd:option('-dataFile', '../data/train.relword.50.torch','training data file')
    cmd:option('-testData','../data/test.rel.FB5M.torch','run test on which data set')
    cmd:option('-testOn',true)
    cmd:option('-round','var')
    cmd:option('-seed',321,'torch manual random number generator seed')
    cmd:text()

    local opt = cmd:parse(arg)    
    
    if opt.useGPU > 0 then
        require 'cutorch'
        require 'cunn'
        cutorch.setDevice(opt.useGPU)
        torch.setdefaulttensortype('torch.CudaTensor')    
    end
    local curr_time = 7227
    local flog = logroll.file_logger(opt.testLogFile..opt.round..'.'..curr_time..'.log')
    -- local wordVocab = torch.load('../vocab/vocab.word.t7')
    -- local relationVocab = torch.load('../vocab/vocab.rel.t7')
    -- local relWordVocab = torch.load('../vocab/vocab.relword.t7')
    -- print(wordVocab.size)
    -- print(relationVocab.size)
    -- print(relWordVocab.size)


    local model = torch.load(opt.saveFile..'.'..opt.round..'.'..curr_time)


        -----------------------------------------------
    -- local plotLoss ={}
    -- for i = 5,250,10 do
    --     opt.testData = '/home/yxy/base/CFO/data/test.rel.FB5M.'..i..'.torch'
    --     local testLoader = RankingDataLoader(opt.testData, opt.relSize, flog)
    --     local singleCount,singleNum = unpack(testfunc(opt,model,testLoader,64))
    --     table.insert(plotLoss,singleCount/singleNum)
    -- end
    -- local losslines
    -- xpcall(function(inp) losslines=torch.load(inp) end,function() losslines=-1 end,'compare.negsize') 
    -- if type(losslines) ~= 'table' then
    --     losslines={}
    -- end
    -- local loss_xline = torch.range(5,250,10)
    -- local lossline = torch.Tensor(plotLoss)
    -- local name = opt.iniMethod
    -- losslines[#losslines+1]={name,loss_xline,lossline}
    -- torch.save('compare.negsize',losslines)
    -- gnuplot.pngfigure('compare_negsize'..'.png')
    -- gnuplot.plot(unpack(losslines))
    -- -- gnuplot.axis({30,1000,0.87,0.92})
    -- -- gnuplot.axis({100,1000,0.84,0.88})
    -- gnuplot.xlabel('negative sampling size')
    -- gnuplot.ylabel('Accuracy')
    -- gnuplot.movelegend('left','bottom')
    -- -- gnuplot.pngfigure('cfoloss.v2'..'.png')
    -- gnuplot.plotflush()
    ------------------------------------------------
        
    for bs = 64,64,64 do
        -- final_testData('../Inference/relAN.final.test.txt',bs)
        -- createRankingData('../Inference/relAN.final.test.txt', opt.testData, wordVocab, relationVocab, bs, relWordVocab,199,9,40)
        opt.testData='../data/final.test.rel.FB5M.torch'
        local singletestLoader = RankingDataLoader(opt.testData, opt.relSize,flog)
        -- local time_start = socket.gettime()
        local singleCount,singleNum = unpack(testfunc(opt,model,singletestLoader,bs))
        -- local time_end = socket.gettime()
        -- local ave = (time_end - time_start)/singleNum
        -- print(string.format('average time is %f s',ave))
    end


end
main()
collectgarbage()

--********************************
-- total test acc is 0.88650488310151   
-- total number is 20274   
--********************************
-- total test acc is 0.87879819145316  
-- total number is 20569   
--********************************