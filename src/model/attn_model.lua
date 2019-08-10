function attn_model(opt,lilayer)
  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local q = inputs[1]
  local k = inputs[2]
  local v = inputs[3]
  local v_t = nn.Transpose({1,2})(v)
  local attn = nn.Transpose({1,3})(BatchAdd()({q,k}))
  -- local tensor_q = nn.Replicate(q:size(2),3)(q)
  -- local tensor_k = nn.Replicate(k:size(2),2)(k)
  -- local attn =nn.Transpose({1,3})(nn.add()({tensor_q,tensor_k}))
  local softmax_attn = nn.SoftMax()
  softmax_attn.name = 'softmax_attn'
  local attn_sf = softmax_attn(attn)
  local attn_t = nn.Transpose({1,3})(attn_sf)
  local attn_d = nn.Dropout(opt.dropoutRate_2)(attn_t)

  local v_combined = nn.MM()({attn_d, v_t})
  if lilayer then
    v_combined = lilayer(v_combined)
  end

  table.insert(outputs,v_combined)

  return nn.gModule(inputs, outputs)
end

function attn_model_single(opt)
  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
----------q (negsize+1*bastchsize)*hiddensize--------------
  local q = inputs[1]
----------v (negsize+1*bastchsize)*seqlen*hiddensize----------
  local v = inputs[2]
  local linear_q = Linear(opt.hiddenSize,1,true,false)
  local linear_v = Linear(opt.hiddenSize,1,true,false)
  local q_ = nn.Replicate(opt.relWordLen,2)(nn.Squeeze(2)(
                linear_q(q)))
  local v_ = linear_v(v)
  local attn = nn.CAddTable()({q_,v_})
  -- local tensor_q = nn.Replicate(q:size(2),3)(q)
  -- local tensor_k = nn.Replicate(k:size(2),2)(k)
  -- local attn =nn.Transpose({1,3})(nn.add()({tensor_q,tensor_k}))
  local softmax_attn = nn.SoftMax()
  softmax_attn.name = 'softmax_attn'
  local attn = softmax_attn(attn)
  local attn = nn.Unsqueeze(2)(attn)
  local attn = nn.Dropout(opt.dropoutRate_2)(attn)

  local v_combined = nn.MM()({attn, v})
  local v_combined = nn.Squeeze(2)(v_combined)
----------------------output (negsize+1*batchsize)*hiddensize-------------------
  table.insert(outputs,v_combined)

  return nn.gModule(inputs, outputs)
end
