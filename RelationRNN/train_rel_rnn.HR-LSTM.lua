--*************************self atten-add+pool********************
require '../init3.back.lua'
require 'gnuplot'
include('test_rel_rnn.HR-LSTM.lua')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a Recurrent Neural Network to embed a sentence')
cmd:text()
cmd:text('Options')
cmd:option('-kernelNum',4)--k>=3
cmd:option('-dropoutRate_2',0.1,'dropout rate')--*******************************
cmd:option('-dropoutRate_r',0.5,'dropout rate')--*******************************
cmd:option('-sigm',0.3)
cmd:option('-back','.back')

cmd:option('-vocabSize',100003,'number of words in dictionary')

cmd:option('-relSize',7524,'number of relations in dictionary')
cmd:option('-relWordLen',9,'number of relations in dictionary')
cmd:option('-relEmbedSize',256,'size of rel embedding')
cmd:option('-relWordSize',3390,'number of relations in dictionary')
cmd:option('-rareRelWordSize',210,'number of relations in dictionary')
cmd:option('-initRange_rwl',0.09,'the range of uniformly initialize parameters')--********
cmd:option('-initRange_rl',0.1,'the range of uniformly initialize parameters')--*********
cmd:option('-initRange_sl',0.5,'the range of uniformly initialize parameters')

cmd:option('-wrdEmbedSize',300,'size of word embedding')
cmd:option('-wrdEmbedPath','../vocab/word.glove100k.t7','pretained word embedding path')

cmd:option('-maxSeqLen',40,'number of timesteps to unroll to')
cmd:option('-hiddenSize',256,'size of RNN internal state')
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
cmd:option('-dataFile', '../data/train.relword.torch','training data file')
cmd:option('-testData','../data/valid.rel.FB5M.torch','run test on which data set')
cmd:option('-round','hr-bilstm')
cmd:option('-testOn',true)

cmd:option('-seed',321,'torch manual random number generator seed')
cmd:text()

----------------------------- parse params -----------------------------
local curr_time = os.time()%1000
local opt = cmd:parse(arg)
local flog = logroll.file_logger(opt.logFile..curr_time..'.log')
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
-- loadPretrainedEmbed(relEmbed,'../vocab/relEmbed.t7')
-- relEmbed.weight[opt.relSize]:zero()
relEmbed.weight:uniform(-1,1)
local ininorm = relEmbed.weight:norm(2,2):expand(relEmbed.weight:size(1),opt.relEmbedSize)
relEmbed.weight:mul(5):cdiv(ininorm)
relEmbed.weight[opt.relSize]:zero()
local rel2word = cudacheck(nn.LookupTable(opt.relSize, 9))
loadPretrainedEmbed(rel2word,'../vocab/rel2word'..opt.back..'.t7')
local relWordEmbed = cudacheck(nn.LookupTable(opt.relWordSize, opt.wrdEmbedSize))
loadPretrainedEmbed(relWordEmbed,'../vocab/relWord_embed'..opt.back..'.t7')
rareWordEmbed = relWordEmbed.weight[{{opt.relWordSize-2-opt.rareRelWordSize+1,opt.relWordSize},{}}]
rareWordEmbed:uniform(-1,1)
local ininorm = rareWordEmbed:norm(2,2):expand(rareWordEmbed:size(1),opt.wrdEmbedSize)
rareWordEmbed:mul(5):cdiv(ininorm)
relWordEmbed.weight[opt.relWordSize]:zero()

local RNN = {}
RNN[1] = cudnn.BLSTM(opt.wrdEmbedSize,opt.hiddenSize,1,false,opt.dropoutRate)--v
RNN[2] = cudnn.BLSTM(opt.hiddenSize*2,opt.hiddenSize,1,false,opt.dropoutRate)--q
local input = nn.Identity()()
local we_inp = wordEmbed(input)
local r1 = RNN[1](we_inp)
local r2 = RNN[2](r1)
local sum_r = nn.CAddTable()({r1,r2})
local output = nn.Normalize(2,1e-6)(nn.Max(1)(sum_r))
local seqModel = nn.gModule({input},{output})

RNN[3] = cudnn.BLSTM(opt.wrdEmbedSize,opt.hiddenSize,1,false,opt.dropoutRate)

local relWordBranch = nn.Sequential():add(nn.View(opt.relWordLen,-1)):add(relWordEmbed)
                        :add(nn.Dropout(opt.dropoutRate_r)):add(RNN[3])

local relLiLayer = Linear(opt.relEmbedSize, opt.hiddenSize*2,false,false)
local relBranch = nn.Sequential():add(nn.View(1,-1)):add(relEmbed):add(nn.Dropout(opt.dropoutRate_r))
                    :add(relLiLayer)--[(33*batchsize) * hiddenSize]
local relCombLayer = nn.Sequential():add(nn.JoinTable(1)):add(nn.Max(1)):add(nn.Normalize(2,1e-6))

local scoreModel = TripleScore_vec(loader.negSize)

-- put all models together
local model = {}
model.seqModel = seqModel
model.relWordBranch  = relWordBranch
model.relBranch  = relBranch
model.relCombLayer = relCombLayer
model.scoreModel = scoreModel

-- margin ranking criterion
local criterion  = nn.MarginRankingCriterion(opt.costMargin)
-- put together parms and grad pointers in optimParams and optimGrad tables
local optimParams, optimGrad = {}, {}
for l=1,#RNN do--4个RNN
    local rnnParams, rnnGrad = RNN[l]:getParameters()
    rnnParams:uniform(-opt.initRange, opt.initRange)
    optimParams[l], optimGrad[l] = rnnParams, rnnGrad
end

local r1RNN_paramID = 1
local r2RNN_paramID = 2
local wrRNN_paramID = 3


optimParams[#optimParams+1], optimGrad[#optimGrad+1] = relLiLayer:getParameters()--6
relLiLayer.weight:uniform(-opt.initRange_rl, opt.initRange_rl)
local rLL_paramID = #optimParams

local lrWrd, lrRel = 1e-3, 3e-4--***********************************************************

local optimConf = {['lr'] = {},['momentum'] = 0.9}--****************************************
-- local optimConf = {['lr'] = {}}
for l = 1, #optimParams do optimConf['lr'][l] = 1e-3 end--**********************************
local optimizer = AdaGrad(optimGrad, optimConf)

-- prepare for training
local sumLoss, epochLoss  = 0, 0
local maxIters = opt.maxEpochs * loader.numBatch
local ones = torch.ones(loader.batchSize, loader.negSize)
local testLoss = 0
local optLoss = 9999
local optEpLoss = 100000
local plotLoss = {}
local timer = torch.Timer()
local now = timer:time()['real']
local last
-- core training loop
for i = 1, maxIters do
--for i = 1, 1 do
    xlua.progress(i, maxIters)
    -- in the beginning of each loop, clean the grad_params
    relEmbed:zeroGradParameters()
    relWordEmbed:zeroGradParameters()--57
    wordEmbed:zeroGradParameters()

    for i = 1, #optimGrad do optimGrad[i]:zero() end

    ----------------------- load minibatch ------------------------
    local seq, rel_w, rel_r = loader:nextBatch()
    local loss = 0
    local paramNorm = 0
    local gradNorm = 0
    ------------------------ forward pass -------------------------
    -- v q k [n_batch x seqLength × 2*n_dim]
    local seqVec = seqModel:forward(seq)
    --relation word level branch [33*n_batch x n_dim]
    local rel_wMat = relWordBranch:forward(rel_w)
    -- relation level branch [33*n_batch x n_dim]
    local rel_rMat = relBranch:forward(rel_r)
    --relation combination [33*n_batch x n_dim]
    local rel_Mat = relCombLayer:forward({rel_wMat,rel_rMat})
    --local rel_Mat_trans = rel_Mat:view(loader.negSize+1,loader.batchSize,opt.hiddenSize):transpose(1,2)
    local posOut = rel_Mat:view(loader.negSize+1,loader.batchSize,-1):narrow(1,1,1):squeeze(1)
    local negOut = rel_Mat:view(loader.negSize+1,loader.batchSize,-1):narrow(1,2,loader.negSize)
    local scores = scoreModel:forward({seqVec,posOut,negOut})
    --loss
    local loss = 0
    loss = criterion:forward(scores, ones)
    --local iloss = criterion:forward(iscores, ones)
    --table.insert(plotLoss,loss)
    ------------------------ backward pass -------------------------
    -- d_scores table {[1] = d_postive_scores, [2] = d_negative_scores}
    local d_scores 
    d_scores = criterion:backward(scores, ones)
    local dposPool,dnegPool = unpack(scoreModel:backward({posPool,negPool}, d_scores))

    local dseqVec,dposOut,dnegOut = unpack(scoreModel:backward({seqVec,posOut,negOut}, d_scores))
    local drel_Mat = torch.Tensor(loader.negSize+1,loader.batchSize,opt.hiddenSize*2)
    drel_Mat[{{1}}] = dposOut
    drel_Mat[{{2,loader.negSize+1}}] = dnegOut
    drel_Mat = drel_Mat:view(-1,opt.hiddenSize)

    drel_wMat, drel_rMat = unpack(relCombLayer:backward({rel_wMat,rel_rMat},drel_Mat))
    relWordBranch:backward(rel_w,drel_wMat)
    gradNorm = gradNorm + optimGrad[wrRNN_paramID]:norm()^2 + relWordEmbed.gradWeight:norm()^2
    relBranch:backward(rel_r,drel_rMat)
    gradNorm = gradNorm + relEmbed.gradWeight:norm()^2 + optimGrad[rLL_paramID]:norm()^2

    seqModel:backward(seq,dseqVec)
    gradNorm = gradNorm + wordEmbed.gradWeight:norm()^2 + optimGrad[r1RNN_paramID]:norm()^2+optimGrad[r2RNN_paramID]:norm()^2

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

    if i==1 and opt.testOn then
        testLoss = testfunc(opt,model,testLoader)
        -- table.insert(plotLoss,testLoss[1]/testLoss[2])
    end



    if i % loader.numBatch == 0 then
        last = now
        now = timer:time()['real']
        print(string.format('training cost time is %f',(now-last)))
    -- if true then
        flog.info(string.format('epoch %3d, loss %6.8f, grad_norm %6.8f, param_norm %6.8f', i / loader.numBatch, epochLoss / loader.numBatch / loader.negSize, gradNorm, paramNorm))
        print(string.format('epoch %3d, loss %6.8f, grad_norm %6.8f, param_norm %6.8f', i / loader.numBatch, epochLoss / loader.numBatch / loader.negSize, gradNorm, paramNorm))
        --torch.save(opt.saveFile..'.'..opt.round..'.'..1, model)
        -- if i/loader.numBatch==43 then 
        --     torch.save(opt.saveFile..'.'..opt.round..'.'..i/loader.numBatch, model)
        -- end
        if opt.testOn  then
            testLoss = testfunc(opt,model,testLoader)
            table.insert(plotLoss,testLoss[1]/testLoss[2])
            if testLoss[1]/testLoss[2]>0.935 then
                local epoch = i / loader.numBatch
                print('')
                print('saving model after epoch', epoch)
                torch.save(opt.saveFile..'.'..curr_time..'.'..epoch, model)
            end
        end
        epochLoss = 0
        if i / loader.numBatch >= 10 then
            optimizer:updateMomentum(math.min(optimizer.momentum + 0.3, 0.99))
        end        
    end

    ------------------------ training info ------------------------
    if i % opt.printEvery == 0 then
        flog.info(string.format("iter %4d, loss = %6.8f, grad_norm %6.8f, param_norm %6.8f", i, sumLoss / opt.printEvery / loader.negSize, gradNorm, paramNorm))
        print(string.format("iter %4d, loss = %6.8f, grad_norm %6.8f, param_norm %6.8f", i, sumLoss / opt.printEvery / loader.negSize, gradNorm, paramNorm))
        -- print(string.format("iter %4d, param_norm %6.8f %6.8f %6.8f %6.8f", i, optimGrad[1]:norm(), optimGrad[2]:norm(), optimGrad[3]:norm(), optimGrad[4]:norm()))
        -- if opt.testOn and i/loader.numBatch>13 then
        --     testLoss = testfunc(opt,model,testLoader,i)--***************
        --     table.insert(plotLoss,testLoss[1]/testLoss[2])
        -- end
        --table.insert(plotItLoss,sumLoss)
        sumLoss = 0
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
print('average accuracy prob is '..math.max(unpack(plotLoss)))
if opt.testOn then

    local loss_xline = torch.range(1,#plotLoss)
    local lossline = torch.Tensor(plotLoss)
    local losslines = torch.load('valid.loss')
    if type(losslines) ~= 'table' then
        losslines={}
    end
    losslines[#losslines+1]={'loss'..#losslines+1,loss_xline,lossline,'~'}
    print('')
    print('saving loss table')
    torch.save('valid.loss',losslines)


    gnuplot.pngfigure('validloss'..'.png')
    gnuplot.plot(unpack(losslines))
    gnuplot.xlabel('iter')
    gnuplot.ylabel('Loss')
    gnuplot.title('valid loss')
    gnuplot.plotflush()
    --gnuplot.axis{'','',0.85,1}
end


collectgarbage()