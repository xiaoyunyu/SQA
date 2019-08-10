function KLDCriterion(opt)
    local inputs = {}
    local outputs = {}
    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())
----------p_theta(h|q) bastchsize*hiddensize--------------
    local inp_mean_theta = inputs[1]
    local inp_logvar_theta = inputs[2]
----------q_phi(h|q,p,y) (negsize+1*bastchsize)*hiddensize----------
    local inp_mean_phi = inputs[3]
    local mean_phi = nn.View(-1,opt.batchSize,opt.hiddenSize)(inp_mean_phi)
    local inp_logvar_phi = inputs[4]  
    local logvar_phi = nn.View(-1,opt.batchSize,opt.hiddenSize)(inp_logvar_phi)
    local var_phi = nn.Exp()(logvar_phi)

    local m_theta_expand = nn.Replicate(opt.negSize+1)(inp_mean_theta)
    local var_theta_expand = nn.Replicate(opt.negSize+1)(nn.Exp()(inp_logvar_theta))
    --negsize+1*batchsize*hiddensize
    local dm_squ = nn.Square()(nn.CSubTable()({m_theta_expand,mean_phi}))
    local tmp = nn.CAddTable()({dm_squ,var_phi})
    local std_squ = nn.CDivTable()({tmp,var_theta_expand})
    --negsize+1*batchsize
    local sum_logvar_phi = nn.Sum(3)(logvar_phi)
    local sum_logvar_theta = nn.Replicate(opt.negSize+1)(nn.Sum(2)(inp_logvar_theta))
    local sum_other = nn.Sum(3)(std_squ)
    --negsize+1*batchsize
    local d_log = nn.CSubTable()({sum_logvar_theta,sum_logvar_phi})
    local half_tensor = torch.zeros(opt.negSize+1,opt.batchSize):fill(0.5)
    local outMat = MulConst(half_tensor)(nn.CAddTable()({d_log,sum_other}))
    local norm_tensor = torch.ones(opt.negSize+1,opt.batchSize):fill(1/opt.negSize)
    norm_tensor[1]:fill(1)
    local out = nn.Mean(1)(nn.Mean(1)(MulConst(norm_tensor)(outMat)))
----------------------output negsize+1*batchsize-------------------
  table.insert(outputs,out)

  return nn.gModule(inputs, outputs)
end




-- local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Criterion')

-- function KLDCriterion:updateOutput(mean, log_var)
--     -- Appendix B from VAE paper: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
--     local batch = 1
--     if mean:dim()>1 then
--         batch = mean:size(1)
--     end
--     local mean_sq = torch.pow(mean, 2)
--     local KLDelements = log_var:clone()

--     KLDelements:exp():mul(-1)
--     KLDelements:add(-1, mean_sq)
--     KLDelements:add(1)
--     KLDelements:add(log_var)

--     self.output = -0.5 * torch.sum(KLDelements) / batch

--     return self.output
-- end

-- function KLDCriterion:updateGradInput(mean, log_var)
-- 	self.gradInput = {}

--     self.gradInput[1] = mean:clone()

--     -- Fix this to be nicer
--     self.gradInput[2] = torch.exp(log_var):mul(-1):add(1):mul(-0.5)

--     return self.gradInput
-- end
