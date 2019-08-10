function PhiLayer(cfg)
	local inputs ={}
	table.insert(inputs,nn.Identity()())
	table.insert(inputs,nn.Identity()())
	table.insert(inputs,nn.Identity()())
	table.insert(inputs,nn.Identity()())
	table.insert(inputs,nn.Identity()())
	--batchsize*hiddensize
	local inp_q =inputs[1]
	--(negsize+1*batchsize)*hiddensize
	local inp_pr = inputs[2]
	local inp_pw = inputs[3]
	--(negsize+1*batchsize)*seqlen*hiddensize
	local inp_ps_mat = inputs[4]
	local inp_ps = nn.Max(2)(inp_ps_mat)
	--negsize+1*batchsize
	local inp_y = inputs[5]
	--layers init
	local linear_q = Linear(cfg.hiddenSize,cfg.hiddenSize)
	local linear_p = Linear(cfg.hiddenSize,cfg.hiddenSize)
	local linear_y = Linear(cfg.classNum,cfg.hiddenSize)
	local y_embed = cudacheck(nn.LookupTable(cfg.classNum, cfg.classNum))
	y_embed.weight:uniform(-1.22,1.22)

	local v_q = nn.View(-1,cfg.hiddenSize)(nn.Contiguous()(
		nn.Replicate(cfg.negSize+1)(
			linear_q(
				inp_q))))
	local v_p = linear_p(nn.CAddTable()({inp_pr,inp_pw,inp_ps}))

	local v_y = linear_y(
		nn.View(-1,cfg.classNum)(
			y_embed(
				inp_y)))

	local v = nn.CAddTable()({v_q,v_p,v_y})

	local mean_logvar = nn.ConcatTable()
	mean_logvar:add(Linear(cfg.hiddenSize, cfg.hiddenSize))
	mean_logvar:add(Linear(cfg.hiddenSize, cfg.hiddenSize))
	--(negsize+1*batchsize)*hiddensize
	local outputs = mean_logvar(v)

	return nn.gModule(inputs, {outputs})
end