function convLayer(cfg)

  local input = nn.Identity()()

  local input_dp = nn.Dropout(cfg.dropout_p)(input)
  -- kernels is an array of kernel sizes
  local kernels = cfg.kernels
  local layer1 = {}
  for i = 1, #kernels do
    local conv
    local conv_layer
    local max_time
    conv = cudnn.SpatialConvolution(1, cfg.num_feat_maps, cfg.vec_size, kernels[i])

    conv_layer = nn.Reshape(cfg.num_feat_maps, -1, true)(
        conv(
        nn.Reshape(1, -1, cfg.vec_size, true)(
        input_dp)))
    max_time = nn.Max(3)(cudnn.ReLU()(conv_layer))

    conv.weight:uniform(-0.01, 0.01)
    conv.bias:zero()
    conv.name = 'convolution'
    table.insert(layer1, max_time)
  end


  local conv_layer_concat
  if #layer1 > 1 then
    conv_layer_concat = nn.JoinTable(2)(layer1)
  end

  local last_layer = conv_layer_concat

  -- simple MLP layer
  local linear = nn.Linear((#layer1) * cfg.num_feat_maps, cfg.hidden_size)
  linear.weight:normal():mul(0.01)
  linear.bias:zero()

  local output = nn.ReLU(true)(linear(nn.Dropout(cfg.dropout_p)(last_layer)))
  model = nn.gModule({input}, {output})
  return model
end