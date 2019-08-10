#!/bin/bash
# for ((i=1;i<10;i++));do
# 	th train_rel_rnn7.lua
# done
# th train_rel_rnnVAR.lua 
#th train_rel_rnnMUL.lua -num_feat_maps 150
th train_rel_rnn7.back.lua -rnn cudnn.gru
th train_rel_rnn7.back.lua -rnn cudnn.lstm
th train_rel_rnn7.back.lua