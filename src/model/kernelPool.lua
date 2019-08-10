local kernelPool, parent = torch.class('kernelPool', 'nn.Module')

function kernelPool:__init(num,sig0,sig1)
    parent.__init(self)
    self.num = num
    self.weight = torch.Tensor(2,num)
    self.gradWeight = torch.Tensor(2,num)

    self:reset(sig0,sig1)

end 

function kernelPool:reset(sig0,sig1)
	self.weight[1][1] = 1--精确匹配
	self.weight[2][1] = sig0--精确匹配
	self.weight[1][2] = 1-(1/(self.num-1))
	self.weight[2][2] = sig1
    for i = 3,self.num do
        self.weight[1][i]=self.weight[1][i-1]-(2/(self.num-1))
        self.weight[2][i] = sig1
    end
    -- print(self.weight)
end

-- function kernelPool:reset(sig0,sig1)
--     self.weight:fill(0.5/self.num)
--     for i = 2,self.num do
--         self.weight[1][i]=self.weight[1][i-1]+(1/self.num)
--     end
-- end

function kernelPool:updateOutput(input)
	if input:dim() == 3 then
    	local sizes = input:size()
        local length = sizes[1]*sizes[2]*sizes[3]
        self.inp = torch.repeatTensor(input,self.num,1,1,1)
        local temp = self.inp:view(self.num,-1)
        temp:csub(torch.repeatTensor(self.weight[1],length,1):transpose(1,2))-- x-c
        temp:cdiv(torch.repeatTensor(self.weight[2],length,1):transpose(1,2))-- (x-c)/d
    	temp:pow(2):mul(-0.5):exp()
        self.output=self.inp:mean(4):squeeze(4):permute(2,3,1)
    else
    	error('input must be 3D Tensor')
    end
    return self.output
end

function kernelPool:updateGradInput(input, gradOutput)
    self.gradInput:resize(input:size()):zero()
    for i = 1,self.num do
        local expandGradOutput = torch.expand(gradOutput:narrow(3,i,1), input:size())
        self.inp[i] = torch.csub(input,self.weight[1][i]):div((-1)*input:size(3)*(self.weight[2][i]^2)):cmul(self.inp[i])
        self.gradInput:addcmul(self.inp[i],expandGradOutput)   
    end
    return self.gradInput
end

function kernelPool:accGradParameters(input, gradOutput, scale)
    local ngradOutput = gradOutput:permute(3,1,2)
    self.gradWeight[1]:add(self.inp:sum(4):mul(-1):cmul(ngradOutput):sum(3):sum(2))
    for i = 1,self.num do
        local temp = torch.csub(input,self.weight[1][i]):div(self.weight[2][i]):mul(-1)
        self.gradWeight[2][i] = self.gradWeight[2][i] + (temp:cmul(self.inp[i]):sum(3):squeeze():cmul(ngradOutput[i]):sum())
    end
end

