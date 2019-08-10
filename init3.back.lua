require 'torch'
require 'nn'
require 'nngraph'
require 'logroll'
require 'socket'
require 'gnuplot'
local ok, err = pcall( function () require 'cutorch' end )
if ok then
	require 'cunn'
	require 'cudnn'
end

include('src/model/CRF.lua')
include('src/model/BiRNN.lua')
include('src/model/BiGRU.lua')
include('src/model/BiRNNSelect.lua')
include('src/model/Linear.lua')
include('src/model/BatchDot.lua')
include('src/model/TripleScore3.lua')
include('src/model/kernelPool.lua')
include('src/model/model_utils.lua')
include('src/model/attn_model.lua')
include('src/model/BatchAdd.lua')
include('src/model/BatchComb.lua')
include('src/model/Sampler.lua')
include('src/model/KLDCriterion.lua')
include('src/model/convLayer.lua')
include('src/model/PhiLayer.lua')
include('src/model/MulConst.lua')
include('src/optim/AdaGrad.lua')
include('src/optim/SGD.lua')

include('src/data/RankingDataLoader3.back.lua')
include('src/model/BiLSTM.lua')

include('src/data/SeqMultiLabelLoader.lua')
include('src/data/SeqLabelingLoader.lua')
include('src/data/SeqRankingLoader3.back.lua')

include('src/data/Vocab.lua')
