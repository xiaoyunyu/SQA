require '../init2.lua'
require 'gnuplot'
include('test_rel_rnn.CFO.lua')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a Recurrent Neural Network to embed a sentence')
cmd:text()
cmd:text('Options')
cmd:option('-updateRel',true,'number of words in dictionary')
cmd:option('-iniMethod','heuristic','number of words in dictionary')

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
cmd:option('-testData','../data/valid.rel.FB5M.torch','run test on which data set')
cmd:option('-round','cfo')
cmd:option('-testOn',true)
cmd:option('-plt',false)
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:text()

----------------------------- parse params -----------------------------
local curr_time = os.time()%1000
local opt = cmd:parse(arg)
local flog = logroll.file_logger(opt.logFile..opt.round..'.'..curr_time..'.log')
flog.info(opt)
--local flog = logroll.print_logger()
if opt.useGPU > 0 then
    cutorch.setDevice(opt.useGPU)
    torch.setdefaulttensortype('torch.CudaTensor')
    torch.manualSeed(curr_time)
    cutorch.manualSeed(curr_time)
end

----------------------------- define loader -----------------------------
local loader = SeqRankingLoader(opt.dataFile, opt.negSize, opt.relSize, flog)
local testLoader = RankingDataLoader(opt.testData, flog)
----------------------------- define models -----------------------------
-- word embedding model
local wordEmbed = cudacheck(nn.LookupTable(opt.vocabSize, opt.wrdEmbedSize))
loadPretrainedEmbed(wordEmbed, opt.wrdEmbedPath)

-- rel embedding model
-- local relEmbed = torch.load('../TransE/model.60').RelEmbed
local relEmbed = nn.Sequential()
local relEmbed_i = cudacheck(nn.LookupTable(opt.relSize, opt.relEmbedSize))
if opt.iniMethod=='heuristic' then
	loadPretrainedEmbed(relEmbed_i,'../vocab/relEmbed.t7')
else
	relEmbed_i.weight[opt.relSize]:uniform(-opt.initRange, opt.initRange)--************可改************
	relEmbed_i.weight[opt.relSize]:view(1,-1):renorm(2, 1, 1)
end


local relEmbed_l = Linear(opt.relEmbedSize,opt.relEmbedSize)
local RelDrop = nn.Dropout(0.3)--************可改************
if opt.addLinear then
    relEmbed:add(relEmbed_i):add(relEmbed_l):add(RelDrop)
else
    relEmbed:add(relEmbed_i):add(RelDrop)
end

--local posRelDrop = nn.Dropout(0.3)--************可改************
--local posNorm = nn.Normalize(2,1e-6)
--local negNorm = nn.Normalize(2,1e-6)

-- multi-layer (stacked) Bi-RNN
local config = {}
config.hiddenSize = opt.hiddenSize
config.maxSeqLen  = opt.maxSeqLen
config.maxBatch   = 256
config.logger     = flog

local RNN = {}
for l = 1, opt.numLayer do
    config.inputSize = l == 1 and opt.wrdEmbedSize or opt.hiddenSize * 2
    RNN[l] = BiGRU(config)
end

local selectLayer = BiRNNSelect()
local linearLayer = nn.Linear(2 * opt.hiddenSize, opt.relEmbedSize)

local seqModel = nn.Sequential()
seqModel:add(wordEmbed)
for l = 1, opt.numLayer do
    seqModel:add(nn.Dropout(opt.dropoutRate))--************可改************
    seqModel:add(RNN[l])
end
seqModel:add(selectLayer)
seqModel:add(linearLayer)


--seqModel:add(nn.Normalize(2,1e-6))

-- ranking score model
local scoreModel = TripleScore(opt.negSize)

-- put all models together
local model = {}
model.seqModel   = seqModel
model.relEmbed   = relEmbed
--model.posRelDrop = posRelDrop
--model.negRelDrop = negRelDrop
model.scoreModel = scoreModel
--model.posNorm = posNorm
--model.negNorm = negNorm



-- margin ranking criterion
local criterion  = nn.MarginRankingCriterion(opt.costMargin)

-- put together parms and grad pointers in optimParams and optimGrad tables
local optimParams, optimGrad = {}, {}
for l = 1, opt.numLayer do
    local rnnParams, rnnGrad = RNN[l]:getParameters()
    rnnParams:uniform(-opt.initRange, opt.initRange)--************可改************
    optimParams[l], optimGrad[l] = rnnParams, rnnGrad
end
optimParams[#optimParams+1], optimGrad[#optimGrad+1] = linearLayer:getParameters()
optimParams[#optimParams+1], optimGrad[#optimGrad+1] = relEmbed_l:getParameters()
-- optimization configurations [subject to change]
local lrWrd, lrRel = 1e-3, 3e-4--************可改************

local optimConf = {['lr'] = {},['momentum'] = 0.3}--*****************修改*******************
-- local optimConf = {['lr'] = {}}
for l = 1, #optimParams do optimConf['lr'][l] = 1e-3 end
local optimizer = AdaGrad(optimGrad, optimConf)--************可改************

-- prepare for training
local sumLoss, epochLoss  = 0, 0
local maxIters = opt.maxEpochs * loader.numBatch
local ones = torch.ones(loader.batchSize, loader.negSize)
local testLoss = 0
local optLoss = 9999
local optEpLoss = 100000
local plotLoss = {}
local max_prec = 0


-- core training loop
for i = 1, maxIters do
    xlua.progress(i, maxIters)
    -- in the beginning of each loop, clean the grad_params
    
    relEmbed:zeroGradParameters()
    wordEmbed:zeroGradParameters()
    for i = 1, #optimGrad do optimGrad[i]:zero() end

    ----------------------- load minibatch ------------------------
    local seq, pos, negs = loader:nextBatch()
    seq = seq:transpose(1,2)
    negs = negs:transpose(1,2)
    local rel = torch.Tensor(loader.negSize+1,loader.batchSize)
    rel[{{1},{}}]=pos
    rel[{{2,loader.negSize+1},{}}]=negs
    local currSeqLen = seq:size(1)
    local loss = 0

    ------------------------ forward pass -------------------------
    -- sequence vectors [n_batch x n_dim]
    -- print(seq:size())
    -- print(seq)
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
    local loss = criterion:forward(scores, ones)

    --local iloss = criterion:forward(iscores, ones)
    --table.insert(plotLoss,loss)
    
    -- d_scores table {[1] = d_postive_scores, [2] = d_negative_scores}
    local d_scores = criterion:backward(scores, ones)
    -- d_seqVec [n_batch x n_dim], d_posVec [n_batch x n_dim], d_negMat [n_neg x n_batch x n_dim]
    -- local d_seqVec, d_posVec, d_negMat = unpack(scoreModel:backward({seqVec, posVec, negMat}, d_scores))
    local d_seqVec, d_posVec, d_negMat = unpack(scoreModel:backward({seqVec, posVec, negMat
--        :view(negDropMat_:size(1),negDropMat_:size(2),-1)
        }, d_scores))
    local drelMat = torch.Tensor(loader.negSize+1,loader.batchSize,opt.relEmbedSize)
    drelMat[{{1},{},{}}]=d_posVec
    drelMat[{{2,loader.negSize+1},{},{}}]=d_negMat
    relEmbed:backward(rel,drelMat)
--    local d_negMat_ = negNorm:backward(negDropMat_:view(negDropMat_:size(1)
--        *negDropMat_:size(2),-1),d_negDropMat:view(d_negDropMat:size(1)
--        *d_negDropMat:size(2),-1))
--     local d_negMat = negRelDrop:backward(negMat, d_negDropMat
-- --        d_negMat_:view(d_negDropMat:size(1),d_negDropMat:size(2),-1)
--         )
    
--    local d_posVec_ = posNorm:backward(posDropVec_,d_posDropVec)
    -- local d_posVec = posRelDrop:backward(posVec, d_posDropVec)

    -- grad due to negative matrix
--    relEmbed:backward(negs, d_negMat)

    -- grad due to positive vectors
--    relEmbed:backward(pos, d_posVec)

    -- grad to the sequence model
    -- seqModel:backward(dropedSeq, d_seqVec)
    seqModel:backward(seq, d_seqVec)


    ----------------------- parameter update ----------------------
    -- sgd with scheduled anealing
    --relEmbed.gradWeight:clamp(-10,10)
    -- if i <= loader.numBatch then
    --     relEmbed:updateParameters(lrRel / (1 + 0.0001 * i))
    --     --relEmbed.weight:renorm(2, 1, 1)
    -- end

    -- renorm rel embeding into normal ball
    --relEmbed.weight:renorm(2, 2, 1)
    if opt.updateRel then
        -- sgd with scheduled anealing
        relEmbed_i:updateParameters(lrRel / (1 + 0.0001 * i))
        -- renorm rel embeding into normal ball
        relEmbed_i.weight:renorm(2, 2, 1)
    end
    -- sgd with scheduled anealing (override with sparse update)
    wordEmbed:updateParameters(lrWrd / (1 + 0.0001 * i))

    -- adagrad for rnn, projection     
    for l = 1, opt.numLayer do optimGrad[l]:clamp(-10, 10) end
    optimizer:updateParams(optimParams, optimGrad)
    if i==1 then
    	testLoss = testfunc(opt,model,testLoader,criterion,i)
    end
        
    -- accumulate loss
    sumLoss   = sumLoss + loss
    epochLoss = epochLoss + loss

    -- scheduled anealing the momentum rate after each epoch
    if i % loader.numBatch == 0 then
        flog.info(string.format('epoch %3d, loss %6.8f', i / loader.numBatch, epochLoss / loader.numBatch / loader.negSize))
        print(string.format('epoch %3d, loss %6.8f', i / loader.numBatch, epochLoss / loader.numBatch / loader.negSize))
        --table.insert(plotEpLoss,epochLoss)
        if opt.testOn then
            local timer = torch.Timer()
            local last = timer:time()['real']
            testLoss = testfunc(opt,model,testLoader,criterion,i)--***************
            local now = timer:time()['real']
            print(string.format('test cost time is %f',(now-last)))
            flog.info(string.format('test cost time is %f',(now-last)))
            local prec = testLoss[1]/testLoss[2]
            table.insert(plotLoss,prec)
            if prec-max_prec>0.001 and prec>0.86 then
                max_prec = prec
                local epoch = i / loader.numBatch
                print('')
                print('saving model after epoch'..epoch)
                flog.info('saving model after epoch'..epoch)
                torch.save(opt.saveFile..'.'..opt.round..'.'..opt.iniMethod, model)
            end
        end
        epochLoss = 0
        if i / loader.numBatch >= 10 then
            optimizer:updateMomentum(math.min(optimizer.momentum + 0.3, 0.99))
        end        
    end

    ------------------------ training info ------------------------
    if i % opt.printEvery == 0 then
        flog.info(string.format("iter %4d, loss = %6.8f", i, sumLoss / opt.printEvery / opt.negSize))
        print(string.format("iter %4d, loss = %6.8f", i, sumLoss / opt.printEvery / opt.negSize))
        --table.insert(plotItLoss,sumLoss)
        sumLoss = 0
    end
    -- if i % (loader.numBatch * opt.saveEvery) == 0 then
    --     -- save model after each epoch
    --     local epoch = i / loader.numBatch
    --     print('')
    --     print('saving model after epoch', epoch)
    --     torch.save(opt.saveFile..'.'..opt.useGPU..'.'..epoch, model)

    -- end
end

print('max accuracy prob is '..math.max(unpack(plotLoss)))
flog.info('max accuracy prob is '..math.max(unpack(plotLoss)))
if opt.plt then
    local loss_xline = torch.range(1,#plotLoss,1)
    local lossline = torch.Tensor(plotLoss)
    local losslines
    xpcall(function(inp) losslines=torch.load(inp) end,function() losslines=-1 end,'cfo.loss.v2') 
    if losslines==-1 then
        losslines={}
    end
    local method
    if opt.updateRel then
        method = opt.iniMethod..' w update'
    else
        method = opt.iniMethod..' w/o update'
    end

    losslines[#losslines+1]={method,loss_xline,lossline,'~'}
    print('')
    print('saving loss table')
    torch.save('cfo.loss.v2',losslines)

    -- require 'cutorch'
    -- require 'gnuplot'
    -- a=torch.load('cfo.loss')
    -- b={}
    -- b[1]=a[1]
    -- b[2]=a[2]
    gnuplot.pngfigure('cfoloss.v2'..'.png')
    gnuplot.plot(unpack(losslines))
    -- gnuplot.axis({30,1000,0.87,0.92})
    gnuplot.axis({100,1000,0.84,0.88})
    gnuplot.xlabel('# of training epochs')
    gnuplot.ylabel('Accuracy')
    gnuplot.movelegend('right','bottom')
    -- gnuplot.pngfigure('cfoloss.v2'..'.png')
    gnuplot.plotflush()


    collectgarbage()
end
-- #python
-- import matplotlib as mpl 
-- mpl.use('Agg')
-- import matplotlib.pyplot as plt

-- f=open('cfo.tmp','r')
-- a=f.readlines()
-- x1=x2=list(range(1,1001))
-- x1=x2=x1[::4]
-- y1=a[0].strip().split('\t')
-- y1=[float(x) for x in y1]
-- y1=y1[::4]
-- y2=a[1].strip().split('\t')
-- y2=[float(x) for x in y2]
-- y2=y2[::4]
-- font1 = {'family' : 'Times New Roman',
-- 'weight' : 'normal',
-- 'size'   : 45,
-- }
-- plt.clf()
-- plt.rcParams['savefig.dpi'] = 50 #图片像素
-- -- plt.figure(figsize=[14,12])
-- plt.tick_params(labelsize=40)
-- plt.plot(x1,y1,color='blue',linewidth='2', label='HLB label-embedding')
-- plt.plot(x2,y2,color='green',linewidth='2',label='Random initialization')
-- plt.legend(prop=font1)
-- plt.xlabel('# of training epochs',font1)
-- plt.xticks(list(range(100,1001,100)),rotation=45)
-- plt.ylabel('Accuracy',font1)
-- plt.xlim((30,1000))
-- plt.ylim((0.87,0.92))
-- plt.tight_layout()
-- plt.savefig('cfoloss.png')
