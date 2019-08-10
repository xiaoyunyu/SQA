--*************************self atten-add+pool********************
require '../init3.lua'
require 'gnuplot'
include('test_rel_rnnVAR.lua')

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
cmd:option('-num_feat_maps',300,'number of timesteps to unroll to')

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
cmd:option('-round','var')
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
local relWordEmbedTab = nn.MapTable()
relWordEmbedTab:add(relWordEmbed)
-----------------------seq model---------------------------
local config = {}
config.hiddenSize = opt.hiddenSize
config.maxSeqLen  = opt.maxSeqLen
config.maxBatch   = 256
config.logger     = flog

local RNN_seq = {}
for l = 1, opt.numLayer do
    config.inputSize = l == 1 and opt.wrdEmbedSize or opt.hiddenSize * 2
    RNN_seq[l] = BiGRU(config)
end

local selectLayer = BiRNNSelect()
local seqModel = nn.Sequential()
seqModel:add(wordEmbed)
for l = 1, opt.numLayer do
    seqModel:add(nn.Dropout(opt.dropoutRate))--************可改************
    seqModel:add(RNN_seq[l])
end
seqModel:add(selectLayer)
--------------------v_q------------------------------
local linearLayer = Linear(2 * opt.hiddenSize, opt.hiddenSize)
local normLayer = nn.Sequential():add(linearLayer):add(nn.Normalize(2,1e-6))--:add(nn.ReLU(true))--
------VAE layer-----
local latentLayer = nn.Sequential():add(nn.Dropout(dropoutRate_1))
    :add(Linear(2 * opt.hiddenSize,opt.hiddenSize):reset(opt.initRange_rwl))
    :add(nn.ReLU(true))

local mean_logvar = nn.ConcatTable()
mean_logvar:add(Linear(opt.hiddenSize, opt.hiddenSize):reset(opt.initRange_rl))
mean_logvar:add(Linear(opt.hiddenSize, opt.hiddenSize):reset(opt.initRange_rl))

latentLayer:add(mean_logvar)

local sampler = nn.Sampler()
local reshape_h = nn.Sequential():add(nn.Replicate(loader.negSize+1)):add(nn.Contiguous()):add(nn.View(-1,opt.hiddenSize))
--------phi layer------
local config_phi = {}
config_phi.hiddenSize = opt.hiddenSize
config_phi.classNum = 2
config_phi.negSize = loader.negSize
config_phi.dropoutRate_2 = opt.dropoutRate_2 
local p_phi = PhiLayer(config_phi)
local y_label = label_init(loader.negSize,loader.batchSize)
local y_label_vec = torch.add(y_label,-1):view(-1)
local config_KLD = {}
config_KLD.batchSize = loader.batchSize
config_KLD.hiddenSize = opt.hiddenSize
config_KLD.negSize = loader.negSize
local KLDLayer = KLDCriterion(config_KLD)
-------relation---------------
------------------------synlayer---------------------------------
local rel_vlayer = nn.Sequential():add(nn.Dropout(opt.dropoutRate_r))
                                :add(Linear(opt.wrdEmbedSize, opt.hiddenSize,false,false):reset(opt.initRange_rl))
                                :add(nn.Tanh())
local config_attn = {}
config_attn.hiddenSize = opt.hiddenSize
config_attn.relWordLen = opt.relWordLen 
config_attn.negSize = loader.negSize
config_attn.dropoutRate_2 = opt.dropoutRate_2 
local syn_attn = attn_model_single(config_attn)
---------------------wordlayer--------------------------------
local config_cnn ={}
config_cnn.dropout_p = opt.dropoutRate_r
config_cnn.kernels = {1,2}
config_cnn.num_feat_maps = opt.num_feat_maps
config_cnn.vec_size = opt.wrdEmbedSize
config_cnn.hidden_size = opt.hiddenSize
local cnnLayer = convLayer(config_cnn)
---------------------------------------------------------------
local relLiLayer = Linear(opt.relEmbedSize, opt.hiddenSize,false,false)
local relBranch = nn.Sequential():add(nn.View(-1)):add(relEmbed):add(nn.Dropout(opt.dropoutRate_r))
                    :add(relLiLayer)--:add(nn.ReLU(true))--[(33*batchsize) * hiddenSize]
local relCombLayer = nn.Sequential():add(nn.CAddTable()):add(nn.Normalize(2,1e-6))--:add(nn.ReLU(true))--

-- ranking score model
local scoreModel = TripleScore_vec(loader.negSize)

-- put all models together
local model = {}

model.seqModel = seqModel
model.normLayer = normLayer
model.latentLayer = latentLayer
model.sampler = sampler
model.rel_vlayer = rel_vlayer
model.syn_attn = syn_attn
model.cnnLayer = cnnLayer
model.relBranch  = relBranch
model.relCombLayer = relCombLayer
model.scoreModel = scoreModel
model.relWordEmbedTab = relWordEmbedTab
-- margin ranking criterion
-- local norm_tensor = torch.ones(loader.negSize+1,loader.batchSize):fill(1/loader.negSize)
-- norm_tensor[1]:fill(1)
-- norm_tensor = norm_tensor:view(-1)
local criterion  = nn.MarginRankingCriterion(opt.costMargin)
-- local criterion = nn.BCECriterion()
-- put together parms and grad pointers in optimParams and optimGrad tables
local optimParams, optimGrad = {}, {}

for l=1,#RNN_seq do--4个RNN
    local rnnParams, rnnGrad = RNN_seq[l]:getParameters()
    rnnParams:uniform(-opt.initRange, opt.initRange)
    -- RNN[l].bias:fill(0)
    optimParams[l], optimGrad[l] = rnnParams, rnnGrad
end

optimParams[#optimParams+1], optimGrad[#optimGrad+1] = linearLayer:getParameters()--7
local ll_paramID = #optimParams
optimParams[#optimParams+1], optimGrad[#optimGrad+1] = latentLayer:getParameters()
local lat_paramID = #optimParams
optimParams[#optimParams+1], optimGrad[#optimGrad+1] = rel_vlayer:getParameters()
local relv_paramID = #optimParams 
optimParams[#optimParams+1], optimGrad[#optimGrad+1] = syn_attn:getParameters()--7
optimParams[#optimParams]:uniform(-opt.initRange_rqk, opt.initRange_rqk)
local synAttn_paramID = #optimParams
optimParams[#optimParams+1], optimGrad[#optimGrad+1] = cnnLayer:getParameters()
local cnn_paramID = #optimParams 
optimParams[#optimParams+1], optimGrad[#optimGrad+1] = relLiLayer:getParameters()--6
relLiLayer.weight:uniform(-opt.initRange_rl, opt.initRange_rl)
local rLL_paramID = #optimParams
optimParams[#optimParams+1], optimGrad[#optimGrad+1] = p_phi:getParameters()--6
local phi_paramID = #optimParams
-- optimization configurations [subject to change]
local lrWrd, lrRel = 3e-4, 1e-3--***********************************************************

local optimConf = {['lr'] = {},['momentum'] = 0.9}--****************************************
-- local optimConf = {['lr'] = {}}
for l = 1, #optimParams do optimConf['lr'][l] = 3e-4 end--**********************************
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
    local seq, rel_w, rel_r,rel_s
    if opt.negSize==0 then
        ----------seq:seqlen*batchsize,rel_w/rel_s:seqlen*(negsize*batchsize),rel_r:negsize*batchsize----
        seq, rel_w, rel_r, rel_s = loader:nextBatch()
        rel_w=rel_w:transpose(1,2):contiguous()
        rel_s=rel_s:transpose(1,2):contiguous()
    else
        seq, rel_r = loader:nextBatch()
        rel_w = rel2word:forward(rel_r):transpose(3,1,2):contiguous():view(opt.relWordLen,-1)
    end

    local loss = 0
    local KLDerr = 0
    local paramNorm = 0
    local gradNorm = 0
    ------------------------ forward pass -------------------------
    -------question layers-----------------------
    -- seqVec [n_batch × 2*n_dim]
    local seqVec = seqModel:forward(seq)
    --normVec [batchsize*hiddensize]
    local normVec = normLayer:forward(seqVec)
    -------------latent layer---------------------
    --mean,log-var,h[batchsize x hidden]
    local mean_theta, log_var_theta = unpack(latentLayer:forward(seqVec))
    local h_ = sampler:forward({mean_theta, log_var_theta})
    local h = reshape_h:forward(h_)
    --rel_qMat[9 x (negsize x batchsize)]
    -- KLDerr = KLD:forward(mean, log_var)
    -----------relation layers---------------
    --rel_seq [9 x (negsize x batchsize) x hidden]
    -- rel_rMat [(negsize*n_batch)*hidden]
    local rel_rMat = relBranch:forward(rel_r)

    local rel_em_tab = relWordEmbedTab:forward{rel_w,rel_s}
    local rel_em = rel_em_tab[1]
    local syn_em = rel_em_tab[2]
    --rel_wMat [(neg*batch)*hidden]
    local rel_wMat = cnnLayer:forward(rel_em)
    --rel_sMat [(neg*batch)*seqlen*hidden]
    local rel_sMat = rel_vlayer:forward(syn_em)
    --rel_sAttn [(neg*batch)*hidden]
    -- local mean, log_var = unpack(p_phi:forward({normVec,rel_rMat,rel_wMat,rel_sMat,y_label}))
    -- KLDerr = KLDerr + KLDLayer:forward({mean_theta,log_var_theta,mean,log_var})[1]
    -- local h = sampler:forward({mean, log_var})
    local rel_sAttn = syn_attn:forward({h,rel_sMat})

    local rel_Mat = relCombLayer:forward({rel_wMat,rel_rMat,rel_sAttn})

    -- -- posProd[b * 1 * l] negProd[b * negSize * l]
    -- local candMat = rel_Mat:view(loader.negSize+1,loader.batchSize,-1)
    -- --(negsize+1*batchsize)
    -- local scores = scoreModel:forward({normVec,candMat})
    -- --loss
    -- loss = criterion:forward(scores, y_label_vec)
    posOut = rel_Mat:view(loader.negSize+1,loader.batchSize,-1):narrow(1,1,1):squeeze(1)
    negOut = rel_Mat:view(loader.negSize+1,loader.batchSize,-1):narrow(1,2,loader.negSize)
    scores = scoreModel:forward({normVec,posOut,negOut})
    -- --loss
    loss = criterion:forward(scores, ones)
    ------------------------ backward pass -------------------------
    ------------trunck-------------------------------
    -- d_scores table {[1] = d_postive_scores, [2] = d_negative_scores}
    local d_scores = criterion:backward(scores, ones)
   	-- local d_scores = criterion:backward(scores, y_label_vec)
    -- dnormVec,dcandMat = unpack(scoreModel:backward({normVec,candMat}, d_scores))
    -- drel_Mat = dcandMat:view(-1,opt.hiddenSize)

 	dnormVec,dposOut,dnegOut = unpack(scoreModel:backward({normVec,posOut,negOut}, d_scores))
    drel_Mat = torch.Tensor(loader.negSize+1,loader.batchSize,opt.hiddenSize)
    drel_Mat[{{1}}] = dposOut
    drel_Mat[{{2,loader.negSize+1}}] = dnegOut
    drel_Mat = drel_Mat:view(-1,opt.hiddenSize)

    drel_wMat,drel_rMat,drel_sAttn = unpack(relCombLayer:backward({rel_wMat,rel_rMat,rel_sAttn},drel_Mat))
    
    local dh,drel_sMat = unpack(syn_attn:backward({h,rel_sMat},drel_sAttn))
    gradNorm = gradNorm + optimGrad[synAttn_paramID]:norm()^2
    -- local dmean, dlog_var = unpack(sampler:backward({mean, log_var},dh))
    -- local dmean_theta,dlog_var_theta,dmean2,dlog_var2 = unpack(KLDLayer:backward({mean_theta,log_var_theta,mean,log_var},torch.ones(1)))
    -- dmean:add(dmean2)
    -- dlog_var:add(dlog_var2)
    -- local dnormVec2,drel_rMat2,drel_wMat2,drel_sMat2,_ = 
                        -- unpack(p_phi:backward({normVec,rel_rMat,rel_wMat,rel_sMat,y_label},{dmean, dlog_var}))
    -- gradNorm = gradNorm + optimGrad[phi_paramID]:norm()^2
    -- dnormVec:add(dnormVec2)
    -- drel_rMat:add(drel_rMat2)
    -- drel_wMat:add(drel_wMat2)
    -- drel_sMat:add(drel_sMat2)
    relBranch:backward(rel_r,drel_rMat)
    gradNorm = gradNorm + relEmbed.gradWeight:norm()^2 + optimGrad[rLL_paramID]:norm()^2
    
    dsyn_em = rel_vlayer:backward(syn_em,drel_sMat)
    gradNorm = gradNorm + optimGrad[relv_paramID]:norm()^2
    drel_em = cnnLayer:backward(rel_em,drel_wMat)
    gradNorm = gradNorm + optimGrad[cnn_paramID]:norm()^2
    relWordEmbedTab:backward({rel_w,rel_s},{drel_em,dsyn_em})
    gradNorm = gradNorm + relWordEmbed.gradWeight:norm()^2

    -------------latent layer------------------------
    -- local dKLD_mean, dKLD_log_var = unpack(KLD:backward(mean, log_var))
    local dh_ = reshape_h:backward(h_,dh)
    local dmean_theta, dlog_var_theta = unpack(sampler:backward({mean_theta, log_var_theta},dh_))
    -- -- dmean:add(dKLD_mean)
    -- -- dlog_var:add(dKLD_log_var)
    local dseqVec = latentLayer:backward(seqVec,{dmean_theta, dlog_var_theta})
    -- local dseqVec = torch.zeros(loader.batchSize,opt.hiddenSize*2)
    gradNorm = gradNorm + optimGrad[lat_paramID]:norm()^2
    -------------question branch---------------------
    local dseqVec2 = normLayer:backward(seqVec,dnormVec)
    gradNorm = gradNorm + optimGrad[ll_paramID]:norm()^2
    dseqVec:add(dseqVec2)
    seqModel:backward(seq,dseqVec)
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
    for l = 1, #RNN_seq do optimGrad[l]:clamp(-10, 10) end
    optimizer:updateParams(optimParams, optimGrad)

    for l = 1, #optimParams do paramNorm = paramNorm + optimParams[l]:norm()^2 end
    paramNorm = paramNorm^0.5

    -- accumulate loss
    sumLoss   = sumLoss + loss
    epochLoss = epochLoss + loss
    sumKLDLoss = sumKLDLoss + KLDerr
    epochKLDLoss =epochKLDLoss + KLDerr
    if i==1 and opt.testOn then
        testLoss = testfunc(opt,model,testLoader,i)
    end


    
    if i % loader.numBatch == 0 then
        local now = socket.gettime()
        print(string.format('cost time is %f',(now-last)*1000))
        last = now
    -- if true then
        flog.info(string.format('epoch %3d, loss %6.8f, grad_norm %6.8f, param_norm %6.8f', i / loader.numBatch, epochLoss / loader.numBatch / loader.negSize + epochKLDLoss /  loader.numBatch, gradNorm, paramNorm))
        print(string.format('epoch %3d, loss %6.8f, grad_norm %6.8f, param_norm %6.8f', i / loader.numBatch, epochLoss / loader.numBatch / loader.negSize + epochKLDLoss /  loader.numBatch, gradNorm, paramNorm))        if opt.testOn  then
            local timer = torch.Timer()
            local last = timer:time()['real']
            testLoss = testfunc(opt,model,testLoader)
            local now = timer:time()['real']
            print(string.format('test cost time is %f',(now-last)))
            flog.info(string.format('test cost time is %f',(now-last)))
            local curr_pred = testLoss[1]/testLoss[2]
            table.insert(plotLoss,curr_pred)
            if curr_pred>0.933 and curr_pred-max_pred>opt.eps  then
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