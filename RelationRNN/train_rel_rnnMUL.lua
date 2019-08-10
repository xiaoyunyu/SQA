--*************************self atten-add+pool********************
require '../init3.lua'
require 'gnuplot'
include('test_rel_rnnMUL.lua')

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
cmd:option('-back','.back')
cmd:option('-soft',true)
cmd:option('-soft_update',false)
cmd:option('-attn_mode',true)
cmd:option('-syn',true)
---------------------------------------------------------------
cmd:option('-eps',0.0005)
cmd:option('-num_feat_maps',100,'number of timesteps to unroll to')
cmd:option('-kernels','{1,2,3}','number of timesteps to unroll to')
---------------------------------------------------------------
cmd:option('-config','')

cmd:option('-vocabSize',100003,'number of words in dictionary')

cmd:option('-relSize',7524,'number of relations in dictionary')
cmd:option('-relWordLen',9,'number of relations in dictionary')
cmd:option('-relEmbedSize',256,'size of rel embedding')
cmd:option('-relWordSize',3390,'number of relations in dictionary')
cmd:option('-rareRelWordSize',210,'number of relations in dictionary')
cmd:option('-initRange_rwl',0.09,'the range of uniformly initialize parameters')--********
cmd:option('-initRange_rl',0.1,'the range of uniformly initialize parameters')--*********
cmd:option('-initRange_sl',0.5,'the range of uniformly initialize parameters')
cmd:option('-initRange_rqk',0.15,'the range of uniformly initialize parameters')

cmd:option('-wrdEmbedSize',300,'size of word embedding')
cmd:option('-wrdEmbedPath','../vocab/word.glove100k.t7','pretained word embedding path')

cmd:option('-numLayer',2,'number of timesteps to unroll to')


cmd:option('-maxSeqLen',40,'number of timesteps to unroll to')
cmd:option('-hiddenSize',300,'size of RNN internal state')
cmd:option('-dropoutRate',0.5,'dropout rate')--********************************************

cmd:option('-negSize',0,'number of negtive samples for each iteration')
cmd:option('-maxEpochs',200,'number of full passes through the training data')
cmd:option('-initRange',0.006,'the range of uniformly initialize parameters')--************
cmd:option('-costMargin',0.1,'the margin used in the ranking cost')
cmd:option('-useGPU',1,'whether to use gpu for computation')

cmd:option('-printEvery',200,'how many steps/minibatches between printing out the loss')
cmd:option('-saveEvery',100,'how many epochs between auto save trained models')
cmd:option('-saveFile','/home/yxy/base/CFO/relRNN/model.rel.stackBiRNN','filename to autosave the model (protos) to')
cmd:option('-logFile','logs/rel.stackBiRNN.','log file to record training information')
cmd:option('-testLogFile','logs/test.','log file to record training information')
cmd:option('-dataFile', '../data/train.relword.torch','training data file')
cmd:option('-testData','../data/valid.rel.FB5M.torch','run test on which data set')
cmd:option('-testOn',true)
cmd:option('-round','multi')

cmd:option('-seed',321,'torch manual random number generator seed')
cmd:text()

----------------------------- parse params -----------------------------
local curr_time = os.time()%10000
local opt = cmd:parse(arg)
local flog = logroll.file_logger(opt.logFile..curr_time..'.log')
flog.info(opt)
--local flog = logroll.print_logger()
if opt.useGPU > 0 then
    cutorch.setDevice(opt.useGPU)
    torch.setdefaulttensortype('torch.CudaTensor')
    torch.manualSeed(curr_time)
    cutorch.manualSeed(curr_time)
end

----------------------------- define loader -----------------------------
local loader = SeqRankingLoader(opt.dataFile, opt.relSize, flog,opt.negSize)
local testLoader = RankingDataLoader(opt.testData, opt.relSize,flog)
----------------------------- define models -----------------------------
-- word embedding model
local wordEmbed = cudacheck(nn.LookupTable(opt.vocabSize, opt.wrdEmbedSize))--v
loadPretrainedEmbed(wordEmbed, opt.wrdEmbedPath)
wordEmbed.weight[opt.vocabSize]:zero()
wordEmbed.weight[opt.vocabSize-2]:uniform(-1,1)
local ininorm = wordEmbed.weight[opt.vocabSize-2]:norm()
wordEmbed.weight[opt.vocabSize-2]=wordEmbed.weight[opt.vocabSize-2]*5/ininorm
wordEmbed.weight[opt.vocabSize-1]:uniform(-1,1)
local ininorm = wordEmbed.weight[opt.vocabSize-1]:norm()
wordEmbed.weight[opt.vocabSize-1]=wordEmbed.weight[opt.vocabSize-1]*5/ininorm
-- rel embedding model
local relEmbed = cudacheck(nn.LookupTable(opt.relSize, opt.relEmbedSize))
if not opt.random_init then
    loadPretrainedEmbed(relEmbed,'../vocab/relEmbed.t7')
else
    relEmbed.weight:uniform(-1,1)
    local ininorm = relEmbed.weight:norm(2,2):expand(relEmbed.weight:size(1),opt.relEmbedSize)
    relEmbed.weight:mul(5):cdiv(ininorm)
    relEmbed.weight[opt.relSize]:zero()
end
local rel2word = cudacheck(nn.LookupTable(opt.relSize, 9))
loadPretrainedEmbed(rel2word,'../vocab/rel2word'..opt.back..'.t7')
-- rel word embedding model
local relWordEmbed = cudacheck(nn.LookupTable(opt.relWordSize, opt.wrdEmbedSize))
loadPretrainedEmbed(relWordEmbed,'../vocab/relWord_embed'..opt.back..'.t7')
rareWordEmbed = relWordEmbed.weight[{{opt.relWordSize-2-opt.rareRelWordSize+1,opt.relWordSize},{}}]
rareWordEmbed:uniform(-1,1)
local ininorm = rareWordEmbed:norm(2,2):expand(rareWordEmbed:size(1),opt.wrdEmbedSize)
rareWordEmbed:mul(5):cdiv(ininorm)
relWordEmbed.weight[opt.relWordSize]:zero()
-- rel word position embedding model
-- local relposEmbed = cudacheck(nn.LookupTable(opt.maxSeqLen, opt.wrdEmbedSize))
-- relposEmbed.weight = position_encoding_init(opt.maxSeqLen, opt.wrdEmbedSize)
--sequence q k v attention model
----------------------------attn model--------------------------------
local config_s = {}
config_s.hiddenSize = opt.hiddenSize
config_s.maxSeqLen  = opt.maxSeqLen
config_s.maxBatch   = 256
config_s.logger     = flog
config_s.inputSize  = opt.wrdEmbedSize
local RNN = {}
RNN[1] = BiGRU(config_s)--v
RNN[2] = RNN[1]:clone()--q
RNN[3] = RNN[1]:clone()--k
--config.inputSize = opt.hiddenSize * 2
--batchsize*len*2hiddensize
local v = nn.Sequential():add(nn.Dropout(opt.dropoutRate)):add(RNN[1]):add(nn.Dropout(opt.dropoutRate))--:add(nn.Transpose({1,2}))
local q = nn.Sequential():add(nn.Dropout(opt.dropoutRate)):add(RNN[2]):add(nn.Transpose({1,2}))--:add(Linear(opt.hiddenSize*2, 1, true,false)):add(nn.Transpose({1,2}))
local k = nn.Sequential():add(nn.Dropout(opt.dropoutRate)):add(RNN[3]):add(nn.Transpose({1,2}))--:add(Linear(opt.hiddenSize*2, 1, true,false)):add(nn.Transpose({1,2}))

-- local config_res = {}
-- config_res.hiddenSize = opt.hiddenSize
-- config_res.maxSeqLen  = opt.maxSeqLen
-- config_res.maxBatch   = 256
-- config_res.logger     = flog
-- config_res.inputSize  = opt.hiddenSize*2
-- RNN_res = BiGRU(config_res)
-- RNN[5] = RNN_res
--attention model's layer
local qLiLayer = Linear(opt.hiddenSize*2, 1, true,false)
local kLiLayer = Linear(opt.hiddenSize*2, 1, true,false)
q:add(qLiLayer):add(nn.Dropout(opt.dropoutRate_2))
k:add(kLiLayer):add(nn.Dropout(opt.dropoutRate_2))
local vLiLayer = Linear(2 * opt.hiddenSize, opt.hiddenSize, false)
local attnModel = attn_model(opt,vLiLayer)
-- local ffnLiLayer = Linear(opt.wrdEmbedSize, opt.wrdEmbedSize, false)
-- local ffnLayer = nn.Sequential():add(ffnLiLayer):add(nn.Dropout(opt.dropoutRate_3))
local normLayer = nn.Sequential():add(nn.View(-1,opt.hiddenSize)):add(nn.Normalize(2,1e-6))
-------relation attention---------------
local rel_vlayer = nn.Sequential():add(nn.Dropout(opt.dropoutRate_r))
                                :add(Linear(opt.wrdEmbedSize, opt.hiddenSize,false,false):reset(opt.initRange_rl))
                                :add(nn.Tanh())
                                :add(nn.Max(1))
                                :add(nn.Normalize(2,1e-6))
local config_cnn ={}
config_cnn.dropout_p = opt.dropoutRate_r
loadstring("kernels  = " .. opt.kernels)()
config_cnn.kernels = kernels
config_cnn.num_feat_maps = opt.num_feat_maps
config_cnn.vec_size = opt.wrdEmbedSize
config_cnn.hidden_size = opt.hiddenSize
local cnnLayer = nn.Sequential():add(convLayer(config_cnn)):add(nn.Normalize(2,1e-6))
local relWordEmbedTab = nn.MapTable()
relWordEmbedTab:add(relWordEmbed)

local relLiLayer = Linear(opt.relEmbedSize, opt.hiddenSize,false,false)

local relBranch = nn.Sequential():add(nn.View(-1)):add(relEmbed):add(nn.Dropout(opt.dropoutRate_r))
                    :add(relLiLayer):add(nn.Normalize(2,1e-6))--:add(nn.ReLU(true))--[(33*batchsize) * hiddenSize]

-- ranking score model
local prodLayer = TripleProd()

--kernel pooling
local transLayer = nn.MapTable():add(nn.Sequential():add(nn.View(loader.negSize+1,loader.batchSize,-1)):add(nn.Transpose({1,2})))
local poolLayer = nn.MapTable():add(nn.Sequential():add(prodLayer):add(kernelPool(opt.kernelNum,opt.sigm,opt.sigm)):add(nn.Log()))--**************************
local sumLayer = Linear(opt.kernelNum*2,1,true)
local tmnLayer = nn.Sequential():add(poolLayer):add(nn.JoinTable(3)):add(sumLayer):add(nn.Tanh())

scoreModel = TripleScore(loader.negSize)


-- put all models together
local model = {}
model.seqEmbed = wordEmbed
model.v   = v
model.q   = q
model.k   = k 
model.attnModel  = attnModel
-- model.ffnLayer = ffnLayer
model.normLayer = normLayer

model.rel_vlayer = rel_vlayer
-- model.relWordBranch  = relWordBranch
model.relWordEmbed = relWordEmbed
-- model.relposEmbed = relposEmbed
model.relBranch  = relBranch

model.tmnLayer   = tmnLayer
-- model.tmnLayer = tmnLayer
model.scoreModel = scoreModel
model.relEmbed = relEmbed
model.cnnLayer = cnnLayer
model.relWordEmbedTab = relWordEmbedTab
-- margin ranking criterion
local criterion  = nn.MarginRankingCriterion(opt.costMargin)
-- put together parms and grad pointers in optimParams and optimGrad tables
local optimParams, optimGrad = {}, {}
for l=1,#RNN do--4个RNN
    local rnnParams, rnnGrad = RNN[l]:getParameters()
    rnnParams:uniform(-opt.initRange, opt.initRange)
    -- RNN[l].bias:fill(0)
    optimParams[l], optimGrad[l] = rnnParams, rnnGrad
end

local vRNN_paramID = 1
local qRNN_paramID = 2
local kRNN_paramID = 3
-- local resRNN_paramID = 5
-- optimParams[#optimParams+1], optimGrad[#optimGrad+1] = relWordLiLayer:getParameters()--5
-- relWordLiLayer.weight:uniform(-opt.initRange_rwl, opt.initRange_rwl)
-- local rwLL_paramID = #optimParams
optimParams[#optimParams+1], optimGrad[#optimGrad+1] = relLiLayer:getParameters()--6
relLiLayer.weight:uniform(-opt.initRange_rl, opt.initRange_rl)
local rLL_paramID = #optimParams
optimParams[#optimParams+1], optimGrad[#optimGrad+1] = qLiLayer:getParameters()--7
qLiLayer.weight:uniform(-opt.initRange_rl, opt.initRange_rl)
local ql_paramID = #optimParams
optimParams[#optimParams+1], optimGrad[#optimGrad+1] = kLiLayer:getParameters()--8
kLiLayer.weight:uniform(-opt.initRange_rl, opt.initRange_rl)
local kl_paramID = #optimParams
optimParams[#optimParams+1], optimGrad[#optimGrad+1] = vLiLayer:getParameters()--9
vLiLayer.weight:uniform(-opt.initRange_rwl, opt.initRange_rwl)
vLiLayer.bias:fill(0)
local vl_paramID = #optimParams

optimParams[#optimParams+1], optimGrad[#optimGrad+1] = rel_vlayer:getParameters()
-- rel_vlayer.weight:uniform(-opt.initRange_rl, opt.initRange_rl)
local relv_paramID = #optimParams 
optimParams[#optimParams+1], optimGrad[#optimGrad+1] = cnnLayer:getParameters()
local cnn_paramID = #optimParams 
-- optimParams[#optimParams+1], optimGrad[#optimGrad+1] = wordEmbed:getParameters()
-- local we_paramID = #optimParams
-- optimParams[#optimParams+1], optimGrad[#optimGrad+1] = relWordEmbed:getParameters()
-- local rwe_paramID = #optimParams
-- optimParams[#optimParams+1], optimGrad[#optimGrad+1] = relEmbed:getParameters()
-- local re_paramID = #optimParams

-- optimParams[#optimParams+1], optimGrad[#optimGrad+1] = ffnLiLayer:getParameters()--9
-- ffnLiLayer.weight:uniform(-opt.initRange_rwl, opt.initRange_rl)
-- ffnLiLayer.bias:fill(0)
-- local ffn_paramID = #optimParams
optimParams[#optimParams+1], optimGrad[#optimGrad+1] = sumLayer:getParameters()--10
sumLayer.weight:uniform(-opt.initRange_sl,opt.initRange_sl)
sumLayer.bias:fill(0)
local sum_paramID = #optimParams

-- optimParams[#optimParams+1], optimGrad[#optimGrad+1] = sumLayer:getParameters()--10
-- sumLayer.weight:uniform(-opt.initRange_sl,opt.initRange_sl)

--optimParams[#optimParams+1], optimGrad[#optimGrad+1] = linearLayer:getParameters()
-- optimization configurations [subject to change]
local lrWrd, lrRel = 1e-3, 3e-3--***********************************************************

local optimConf = {['lr'] = {},['momentum'] = 0.9}--****************************************
-- local optimConf = {['lr'] = {}}
for l = 1, #optimParams do optimConf['lr'][l] = 1e-3 end--**********************************
local optimizer = AdaGrad(optimGrad, optimConf)

-- prepare for training
local sumLoss, epochLoss  = 0, 0
local sumKLDLoss, epochKLDLoss = 0, 0
local maxIters = opt.maxEpochs * loader.numBatch
local ones = torch.ones(loader.batchSize, loader.negSize)
local testLoss = 0
local optLoss = 9999
local optEpLoss = 100000
local plotLoss = {}
local last = 0
local max_pred = 0
-- core training loop
for i = 1, maxIters do
-- for i = 1, 1 do
    xlua.progress(i, maxIters)
    -- in the beginning of each loop, clean the grad_params
    relEmbed:zeroGradParameters()
    relWordEmbed:zeroGradParameters()--57
    wordEmbed:zeroGradParameters()

    for i = 1, #optimGrad do optimGrad[i]:zero() end
    ----------------------- load minibatch ------------------------
    local seq, rel_w, rel_r
    if opt.negSize==0 then
        seq, rel_w, rel_r = loader:nextBatch()
    else
        seq, rel_r = loader:nextBatch()
        rel_w = rel2word:forward(rel_r):transpose(3,1,2):contiguous():view(opt.relWordLen,-1)
    end
    -- print(rel_pos)
    -- local rel_w1 = rel_w[{{},{1,32},{}}]
    -- local rel_w2 = rel_w[{{},{33,64},{}}]
    -- local rel_r1 = rel_r[{{1,32},{}}]
    -- local rel_r2 = rel_r[{{33,64},{}}]
    local loss = 0
    local KLDerr = 0
    local paramNorm = 0
    local gradNorm = 0
    ------------------------ forward pass -------------------------
    -------question layers-----------------------
    -- v q k [n_batch x seqLength × 2*n_dim]
    local wrdEmd = wordEmbed:forward(seq)

    vMat = v:forward(wrdEmd)
    qMat = q:forward(wrdEmd)
    kMat = k:forward(wrdEmd)
    -- local res_vMat = RNN_res:forward(vMat)
    -- res_vMat = res_vMat + vMat
    --attenMat [batchsize * seqLength * 2*hidden]
    attnMat = attnModel:forward({qMat,kMat,vMat}) 
    attnMat:add(wrdEmd:transpose(1,2))
    -- attnMat = attnMat + wrdEmd:transpose(1,2)
    -- local ffnMat = ffnLayer:forward(attnMat)
    -- ffnMat = ffnMat + attnMat
    normMat = normLayer:forward(attnMat)
    normMat = normMat:view(loader.batchSize,-1,opt.hiddenSize)

    -------------latent layer---------------------
    --mean,log-var,h[batchsize x hidden]
    -- local mean, log_var = unpack(latentLayer:forward(vMat))
    -- local h = sampler:forward({mean, log_var})
    --rel_qMat[9 x (negsize x batchsize)]
    -- local rel_qMat = rel_qlayer:forward(h)
    -- KLDerr = KLD:forward(mean, log_var)
    -----------relation word branch---------------
    --rel_seq [9 x (negsize x batchsize) x hidden]
    -- relation level branch [33*n_batch x n_dim]
    local rel_rMat = relBranch:forward(rel_r)

    -- rel_w = rel_w:transpose(1,2)
    rel_em_tab = relWordEmbedTab:forward{rel_w}
    rel_em = rel_em_tab[1]
    -- syn_em = rel_em_tab[2]
    rel_wMat = rel_vlayer:forward(rel_em)
    -- rel_sMat = rel_vlayer:forward(syn_em)


    -- local rel_pos_em = relposEmbed:forward(rel_pos)

    -- local rel_em = rel_seq_em+rel_pos_em


    --rel_kMat[9 x (negsize x batchsize)]
    -- local rel_kMat = rel_klayer:forward(rel_em)
    --rel_wMat[(negsize x batchsize) x hidden]
    -- local rel_wMat = rel_attnModel:forward({rel_qMat,rel_kMat,rel_seq})
    --local rel_Mat_trans = rel_Mat:view(loader.negSize+1,loader.batchSize,opt.hiddenSize):transpose(1,2)
    -- posProd[b * 1 * l] negProd[b * negSize * l]

    rel_rMatT,rel_wMatT = unpack(transLayer:forward({rel_rMat,rel_wMat}))
  -----------trunk----------------------------
    scoreMat = tmnLayer:forward({{normMat,rel_rMatT},{normMat,rel_wMatT}})
    posPool = scoreMat[{{},{1}}]
    negPool = scoreMat[{{},{2,loader.negSize+1}}]
    --score
    scores = scoreModel:forward({posPool,negPool})

    --loss
    loss = criterion:forward(scores, ones)
    --local iloss = criterion:forward(iscores, ones)
    --table.insert(plotLoss,loss)
    ------------------------ backward pass -------------------------
    ------------trunck-------------------------------
    -- d_scores table {[1] = d_postive_scores, [2] = d_negative_scores}
    local d_scores 
    d_scores = criterion:backward(scores, ones)

    dposPool,dnegPool = unpack(scoreModel:backward({posPool,negPool}, d_scores))

    dscoreMat = torch.Tensor(loader.batchSize,loader.negSize+1)
    dscoreMat[{{},{1}}]=dposPool
    dscoreMat[{{},{2,loader.negSize+1}}]=dnegPool

    drel_rMatPair,drel_wMatPair = unpack(tmnLayer:backward({{normMat,rel_rMatT},{normMat,rel_wMatT}},dscoreMat))
    dnormMat,drel_rMatT = unpack(drel_rMatPair)
    dnormMat2,drel_wMatT = unpack(drel_wMatPair)
    -- dnormMat3,drel_sMatT = unpack(drel_sMatPair)
    dnormMat:add(dnormMat2)
    -- dnormMat:add(dnormMat3)

    gradNorm = gradNorm + optimGrad[sum_paramID]:norm()^2

----------------------relation branch-------------
    drel_rMat, drel_wMat = unpack(transLayer:backward({rel_rMat,rel_wMat},{drel_rMatT, drel_wMatT})) 


    relBranch:backward(rel_r,drel_rMat)
    gradNorm = gradNorm + relEmbed.gradWeight:norm()^2 + optimGrad[rLL_paramID]:norm()^2
    -- dsyn_em = rel_vlayer:backward(syn_em,drel_sMat)
    -- gradNorm = gradNorm + optimGrad[relv_paramID]:norm()^2
    drel_em = rel_vlayer:backward(rel_em,drel_wMat)
    gradNorm = gradNorm + optimGrad[relv_paramID]:norm()^2
    relWordEmbedTab:backward({rel_w},{drel_em})
    gradNorm = gradNorm + relWordEmbed.gradWeight:norm()^2

    -------------latent layer------------------------
    -- local dKLD_mean, dKLD_log_var = unpack(KLD:backward(mean, log_var))
    -- local dh = rel_qlayer:backward(h,drel_qMat)
    -- gradNorm = gradNorm + optimGrad[relq_paramID]:norm()^2
    -- local dmean, dlog_var = unpack(sampler:backward({mean, log_var},dh))
    -- dmean:add(dKLD_mean)
    -- dlog_var:add(dKLD_log_var)
    -- local dvMat_lat = latentLayer:backward(vMat,{dmean, dlog_var})
    -- gradNorm = gradNorm + optimGrad[lat_paramID]:norm()^2
    -------------question branch---------------------

    dnormMat = dnormMat:view(-1,opt.hiddenSize)
    dattnMat = normLayer:backward(attnMat,dnormMat)
-- local dattnMat = ffnLayer:backward(attnMat,dffnMat)
-- dattnMat = dattnMat + dffnMat
-- gradNorm = gradNorm + optimGrad[ffn_paramID]:norm()^2
    dqMat,dkMat,dvMat = unpack( attnModel:backward({qMat,kMat,vMat},dattnMat) )
-- dvMat:add(dvMat_lat)
    gradNorm = gradNorm + optimGrad[vl_paramID]:norm()^2

-- local dvMat = RNN_res:backward(vMat,dres_vMat)
-- gradNorm = gradNorm + optimGrad[resRNN_paramID]:norm()^2
-- dvMat = dvMat + dres_vMat
    dwrdEmd = v:backward(wrdEmd,dvMat)
    gradNorm = gradNorm + optimGrad[vRNN_paramID]:norm()^2--+ wordGrad[vw_paramID]:norm()^2
    dwrdEmd = dwrdEmd + q:backward(wrdEmd,dqMat)
    gradNorm = gradNorm + optimGrad[qRNN_paramID]:norm()^2+ optimGrad[ql_paramID]:norm()^2--+ wordGrad[qw_paramID]:norm()^2
    dwrdEmd = dwrdEmd + k:backward(wrdEmd,dkMat)
    gradNorm = gradNorm + optimGrad[kRNN_paramID]:norm()^2+ optimGrad[kl_paramID]:norm()^2--+wordGrad[kw_paramID]:norm()^2
    dwrdEmd =  dwrdEmd + dattnMat:transpose(1,2)

    -- dwrdEmd = dwrdEmd + dattnMat:transpose(1,2)
    wordEmbed:backward(seq,dwrdEmd)
    gradNorm = gradNorm + wordEmbed.gradWeight:norm()^2

    gradNorm=gradNorm^0.5

    ----------------------- parameter update ----------------------
    -- sgd with scheduled anealing (override with sparse update)
        --wordEmbed[l].gradWeight[{{100001,100002}}]=0
    wordEmbed.gradWeight[wordEmbed.weight:size(1)]:zero()
    wordEmbed:updateParameters(lrWrd / (1 + 0.0001 * i))
    paramNorm = paramNorm + wordEmbed.weight:norm()^2
    
    relWordEmbed.gradWeight[relWordEmbed.weight:size(1)]:zero()
    relWordEmbed:updateParameters(lrWrd / (1 + 0.0001 * i))
    paramNorm = paramNorm + relWordEmbed.weight:norm()^2

    relEmbed.gradWeight[opt.relSize]:zero()
    relEmbed:updateParameters(lrRel / (1 + 0.0001 * i))
    paramNorm = paramNorm + relEmbed.weight:norm()^2
    -- adagrad for rnn, projection    
    for l = 1, #RNN do optimGrad[l]:clamp(-10, 10) end
    -- optimGrad[5]:clamp(-10, 10) 
    optimizer:updateParams(optimParams, optimGrad)

    for l = 1, #optimParams do paramNorm = paramNorm + optimParams[l]:norm()^2 end
    paramNorm = paramNorm^0.5

    -- accumulate loss
    sumLoss   = sumLoss + loss
    epochLoss = epochLoss + loss
    sumKLDLoss =0
    epochKLDLoss =0
    if i==1 and opt.testOn then
        testLoss = testfunc(opt,model,testLoader,i)
        -- table.insert(plotLoss,testLoss[1]/testLoss[2])
    end


    
    if i % loader.numBatch == 0 then
        local now = socket.gettime()
        print(string.format('cost time is %f',(now-last)*1000))
        last = now
    -- if true then
        flog.info(string.format('epoch %3d, loss %6.8f, grad_norm %6.8f, param_norm %6.8f', i / loader.numBatch, epochLoss / loader.numBatch / loader.negSize + epochKLDLoss /  loader.numBatch, gradNorm, paramNorm))
        print(string.format('epoch %3d, loss %6.8f, grad_norm %6.8f, param_norm %6.8f', i / loader.numBatch, epochLoss / loader.numBatch / loader.negSize + epochKLDLoss /  loader.numBatch, gradNorm, paramNorm))
        --torch.save(opt.saveFile..'.'..opt.round..'.'..1, model)
        -- if i/loader.numBatch==43 then 
        --     torch.save(opt.saveFile..'.'..opt.round..'.'..i/loader.numBatch, model)
        -- end
        if opt.testOn  then
            local timer = torch.Timer()
            local last = timer:time()['real']
            testLoss = testfunc(opt,model,testLoader)
            local now = timer:time()['real']
            print(string.format('test cost time is %f',(now-last)))
            flog.info(string.format('test cost time is %f',(now-last)))
            local curr_pred = testLoss[1]/testLoss[2]
            table.insert(plotLoss,curr_pred)
            if curr_pred>0.932 and curr_pred-max_pred>opt.eps  then
                max_pred = curr_pred
                local epoch = i / loader.numBatch
                print('')
                print('saving model after epoch', epoch)
                torch.save(opt.saveFile..'.'..opt.round..'.'..curr_time, model)
            end
        end
        epochLoss = 0
        epochKLDLoss = 0
        if i / loader.numBatch >= 10 then
            optimizer:updateMomentum(math.min(optimizer.momentum + 0.3, 0.99))
        end        
    end

    ------------------------ training info ------------------------
    if i % opt.printEvery == 0 then
        flog.info(string.format("iter %4d, loss = %6.8f, grad_norm %6.8f, param_norm %6.8f", i, sumLoss / opt.printEvery / loader.negSize + sumKLDLoss / opt.printEvery, gradNorm, paramNorm))
        print(string.format("iter %4d, loss = %6.8f, grad_norm %6.8f, param_norm %6.8f", i, sumLoss / opt.printEvery / loader.negSize + sumKLDLoss / opt.printEvery, gradNorm, paramNorm))
        -- print(string.format("iter %4d, param_norm %6.8f %6.8f %6.8f %6.8f", i, optimGrad[1]:norm(), optimGrad[2]:norm(), optimGrad[3]:norm(), optimGrad[4]:norm()))
        -- if opt.testOn and i/loader.numBatch>13 then
        --     testLoss = testfunc(opt,model,testLoader,i)--***************
        --     table.insert(plotLoss,testLoss[1]/testLoss[2])
        -- end
        --table.insert(plotItLoss,sumLoss)
        sumLoss = 0
        sumKLDLoss = 0
    end
    -- if i % (loader.numBatch * opt.saveEvery) == 0 then
    --     -- save model after each epoch
    --     local epoch = i / loader.numBatch
    --     print('')
    --     print('saving model after epoch', epoch)
    --     torch.save(opt.saveFile..'.'..opt.round..'.'..epoch, model)

    -- end
    collectgarbage()
end

print('max accuracy prob is '..math.max(unpack(plotLoss)))
flog.info('max accuracy prob is '..math.max(unpack(plotLoss)))
print(opt.config)

-- if opt.testOn then

--     local loss_xline = torch.range(1,#plotLoss)
--     local lossline = torch.Tensor(plotLoss)
--     local losslines = torch.load('compare.loss')
--     if type(losslines) ~= 'table' then
--         losslines={}
--     end
--     local name
--     if opt.random_init then
--         name='Random initialization alg.'
--     else
--         name='Heuristic label-embedding alg.'
--     end
--     losslines[#losslines+1]={name,loss_xline,lossline,'~'}
--     print('')
--     print('saving loss table')
--     torch.save('compare.loss',losslines)

    -- require 'cutorch'
    -- require 'gnuplot'
    -- a=torch.load('compare.loss')
    -- b={}
    -- b[1]=a[2]
    -- b[2]=a[1]
    -- gnuplot.pngfigure('comparition_init'..'.png')
    -- gnuplot.plot(unpack(b))
    -- gnuplot.xlabel('# of training epochs')
    -- gnuplot.ylabel('Accuracy')
    -- gnuplot.movelegend('right','bottom')
    -- gnuplot.plotflush()
    -- gnuplot.axis{1,100,0.87,0.94}
-- end


collectgarbage()


-- f=open('shsa.tmp','r')
-- a=f.readlines()
-- x1=x2=list(range(1,101))
-- y1=a[0].strip().split('\t')
-- y1=[float(x) for x in y1]
-- y2=a[1].strip().split('\t')
-- y2=[float(x) for x in y2]
-- font1 = {'family' : 'Times New Roman',
-- 'weight' : 'normal',
-- 'size'   : 45,
-- }
-- plt.clf()
-- plt.rcParams['savefig.dpi'] = 50 #图片像素
-- -- plt.figure(figsize=[14,12])
-- plt.tick_params(labelsize=40)
-- plt.plot(x1,y1,color='blue',linewidth='2', label='HLB')
-- plt.plot(x2,y2,color='green',linewidth='2',label='Rand')
-- plt.legend(prop=font1)
-- plt.xlabel('# of training epochs',font1)
-- plt.xticks(list(range(0,101,10)),rotation=45)
-- plt.ylabel('Accuracy',font1)
-- plt.xlim((1,100))
-- plt.ylim((0.87,0.94))
-- plt.tight_layout()
-- plt.savefig('comparition_init.png')
