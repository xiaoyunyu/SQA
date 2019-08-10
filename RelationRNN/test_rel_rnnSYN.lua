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
    local seqEmbed = model.seqEmbed
    local v   = model.v
    local q   = model.q
    local k   = model.k 
    local attnModel  = model.attnModel
    --local ffnLayer = model.ffnLayer
    local normLayer = model.normLayer
    local latentLayer = model.latentLayer
    local sampler = model.sampler
    local rel_qlayer = model.rel_qlayer
    local rel_klayer = model.rel_klayer
    local rel_vlayer = model.rel_vlayer
    local rel_attnModel = model.rel_attnModel
    -- local relWordBranch  = model.relWordBranch
    local relWordEmbed = model.relWordEmbed
    -- local relposEmbed = model.relposEmbed
    local relBranch = model.relBranch
    local relCombLayer = model.relCombLayer
    local prodLayer = TripleProd()
    local scoreModel = TripleScore_vec4test(loader.negSize+1)
    local relEmbed = model.relEmbed
    if opt.soft then
        tmnLayer = model.tmnLayer_pos
    --local tmnLayer = model.maxLayer_pos
    else
        tmnLayer =model.maxLayer_pos
    end
    local maxLayer_neg = model.maxLayer_neg
    local cnnLayer = model.cnnLayer
    local relWordEmbedTab = model.relWordEmbedTab
    -- local RNN_res = model.RNN_res
 
    -- local wordEmbed = model.wordEmbed 
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
    seqEmbed:evaluate()
    v:evaluate()
    q:evaluate()
    k:evaluate()
    attnModel:evaluate()
    --ffnLayer:evaluate()
    normLayer:evaluate()
    relBranch:evaluate()
    -- relWordBranch:evaluate()
    relWordEmbed:evaluate()
    -- relposEmbed:evaluate()
    relCombLayer:evaluate()
    prodLayer:evaluate()
    scoreModel:evaluate()
    relEmbed:evaluate()
    tmnLayer:evaluate()

    maxLayer_neg:evaluate()
    latentLayer:evaluate()
    sampler:evaluate()
    rel_qlayer:evaluate()
    rel_klayer:evaluate()
    rel_attnModel:evaluate()
    rel_vlayer:evaluate()
    seqModel:evaluate()
    cnnLayer:evaluate()
    -- RNN_res:evaluate()

    -- core testing loop
    --for i = 1, loader.numBatch do
    fo=io.open('predicted_relation.txt','w')
    -- local time_start = socket.gettime()
    for i = 1, loader.numBatch do
        xlua.progress(i, loader.numBatch)
        ----------------------- load minibatch ------------------------
        --rel_r:[negsize * batchsize]
        local seq, rel_w, rel_r,rel_s = loader:nextBatch(1)
        -- if i == 15000 then
        -- 	print(rel_r)
        -- end
        --print(rel_r:size())
        --print(rel_r[{{},{127}}])
        --print(rel_r)--1 4
        local negSize = rel_r:size(1)-1
        ------------------------ forward pass -------------------------    
        -- sequence vectors [n_batch x n_dim]
        local wrdEmd = seqEmbed:forward(seq)
        if opt.attn_mode then
            local vMat = v:forward(wrdEmd)
            local qMat = q:forward(wrdEmd)
            local kMat = k:forward(wrdEmd)
            -- local res_vMat = RNN_res:forward(vMat)
            -- res_vMat = res_vMat + vMat
            local attnMat_ = attnModel:forward({qMat,kMat,vMat}) 
            attnMat_:add(wrdEmd:transpose(1,2))
            --attnMat = attnMat + wrdEmd:transpose(1,2)
            -- local ffnMat = ffnLayer:forward(attnMat)
            -- ffnMat = ffnMat + attnMat
            normMat = normLayer:forward(attnMat_)
            normMat = normMat:view(loader.batchSize,-1,opt.hiddenSize)
        else 
            seqVec = seqModel:forward(wrdEmd)
        end
        -- local mean, log_var = unpack(latentLayer:forward(vMat))

        -- local h = sampler:forward({mean, log_var})

        -- local rel_qMat = rel_qlayer:forward(h)
        
        -- local rel_seq_em,rel_pos_em = unpack(relWordBranch:forward({rel_w,rel_pos}))
        local rel_Mat
        if not usePreCal then
            if opt.syn then
                rel_w = rel_w:transpose(1,2)
                rel_em_tab = relWordEmbedTab:forward({rel_w})
                rel_em = rel_em_tab[1]
                -- syn_em = rel_em_tab[2]
                rel_wMat = cnnLayer:forward(rel_em)
                -- rel_sMat = rel_vlayer:forward(syn_em)
                rel_rMat = relBranch:forward(rel_r)
                rel_Mat = relCombLayer:forward({rel_wMat,rel_rMat})
            else
                local rel_em = relWordEmbed:forward(rel_w)
                -- local rel_pos_em = relposEmbed:forward(rel_pos)
                -- local rel_em = rel_seq_em+rel_pos_em
                local rel_seq = rel_vlayer:forward(rel_em)
                -- local rel_seq = relWordBranch:forward(rel_w)

                -- local rel_kMat = rel_klayer:forward(rel_em)

                -- local rel_wMat = rel_attnModel:forward({rel_qMat,rel_kMat,rel_seq})
                --local temp = rel_wMat:view(-1,loader.batchSize,opt.hiddenSize):transpose(1,2)
                --print(temp[5][1]-temp[5][3])
                local rel_rMat = relBranch:forward(rel_r)
                --local temp = rel_rMat:view(-1,loader.batchSize,opt.hiddenSize):transpose(1,2)
                --print(temp[5][1]-temp[5][3])
                rel_Mat = relCombLayer:forward({rel_seq,rel_rMat})
            end
            --print(rel_Mat:size())
            --[batchsize * negsize * hiddensize]
            rel_Mat = rel_Mat:view(-1,loader.batchSize,opt.hiddenSize)
        else
            rel_Mat = preCal_embed:forward(rel_r:transpose(1,2):contiguous())
        end
        if savePreCal then
            for negId = 1,200 do
                preCal[{rel_r[negId][1]}]=rel_Mat[{1,negId}]
            end
        end
        --print(rel_Mat[5][1]-rel_Mat[5][3])
        -- local posOut = rel_Mat:view(-1,loader.batchSize,opt.hiddenSize):narrow(1,1,1):transpose(1,2)
        -- local negOut = rel_Mat:view(-1,loader.batchSize,opt.hiddenSize):narrow(1,2,negSize):transpose(1,2)
        if opt.attn_mode then
            local relProd = prodLayer:forward({normMat, rel_Mat:transpose(1,2)}) 
            --print(relProd[5][1]-relProd[5][3])
            relscores = tmnLayer:forward(relProd)
        else
            relscores = scoreModel:forward({seqVec,rel_Mat}):squeeze(3):transpose(1,2)
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
    -- local totalLen = ((loader.numBatch-1)*loader.batchSize)+lastBatchlen
    local totalLen = (loader.numBatch*loader.batchSize)
    -- local ave = (time_end - time_start)/totalLen
    -- print(string.format('average time is %f s',ave))
    -- score_file:close()
    -- rank_file:close()
    seqEmbed:training()
    v:training()
    q:training()
    k:training()
    attnModel:training()
    --ffnLayer:training()
    normLayer:training()
    relBranch:training()
    -- relWordBranch:training()
    relWordEmbed:training()
    -- relposEmbed:training()
    relCombLayer:training()
    prodLayer:training()
    scoreModel:training()
    relEmbed:training()
    tmnLayer:training()

    maxLayer_neg:training()
    latentLayer:training()
    sampler:training()
    rel_qlayer:training()
    rel_klayer:training()
    rel_attnModel:training()
    rel_vlayer:training()
    seqModel:training()
    cnnLayer:training()
    -- RNN_res:training()

    

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
    cmd:option('-hiddenSize',300,'size of RNN internal state')
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
    cmd:option('-testData','../data/test.rel.FB5M.torch','run test on which data set')
    cmd:option('-round',888)
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
    -- local wordVocab = torch.load('../vocab/vocab.word.t7')
    -- local relationVocab = torch.load('../vocab/vocab.rel.t7')
    -- local relWordVocab = torch.load('../vocab/vocab.relword.t7')
    -- print(wordVocab.size)
    -- print(relationVocab.size)
    -- print(relWordVocab.size)
    local tab = {114}
    for _,x in ipairs(tab) do
        local model = torch.load(opt.saveFile..'.'..opt.round..'.'..x)
            
        --local multitestLoader = RankingDataLoader(opt.multitestData, opt.relSize, flog)
        for bs = 64,64,64 do
            -- final_testData('../Inference/relAN.final.test.txt',bs)
            -- createRankingData('../Inference/relAN.final.test.txt', opt.testData, wordVocab, relationVocab, bs, relWordVocab,199,9,40)
            local singletestLoader = RankingDataLoader(opt.testData, opt.negSize, flog)
            -- local time_start = socket.gettime()
            local singleCount,singleNum = unpack(testfunc(opt,model,singletestLoader,bs))
            -- local time_end = socket.gettime()
            -- local ave = (time_end - time_start)/singleNum
            -- print(string.format('average time is %f s',ave))
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