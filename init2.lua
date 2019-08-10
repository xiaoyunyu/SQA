require 'torch'
require 'nn'
require 'nngraph'
require 'logroll'
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
include('src/model/TripleScore.lua')
include('src/model/kernelPool.lua')
include('src/model/model_utils.lua')
include('src/model/attn_model.lua')
include('src/model/BatchAdd.lua')
include('src/optim/AdaGrad.lua')
include('src/optim/SGD.lua')

include('src/data/RankingDataLoader2.lua')

include('src/data/SeqMultiLabelLoader.lua')
include('src/data/SeqLabelingLoader2.lua')
include('src/data/SeqRankingLoader2.lua')

include('src/data/Vocab.lua')
print('init done')
