local BatchAdd, parent = torch.class('BatchAdd', 'nn.Module')

function BatchAdd:__init(rep)
    parent.__init(self)
    self.rep = (rep == nil and true) or rep
    self.gradInput = {torch.Tensor(), torch.Tensor()}
    self._viewSize = torch.LongStorage()
end 

function BatchAdd:updateOutput(input)
    if self.rep then
        local bathSize = input[1]:size(1)
    	local length = input[1]:size(2)
    	local q = input[1]:view(bathSize,length,1):repeatTensor(1,1,length)
    	local k = input[2]:view(bathSize,1,length):repeatTensor(1,length,1)
        self.output = torch.add(q, k)
    else
        self.output = torch.add(input[1],input[2])
    end
    return self.output
end

function BatchAdd:updateGradInput(input, gradOutput)
    if self.rep then
        self.gradInput[1] = gradOutput:sum(3):squeeze()
        self.gradInput[2] = gradOutput:sum(2):squeeze()
    else
        self.gradInput[1] = gradOutput:clone()
        self.gradInput[2] = gradOutput:clone()
    end
    return self.gradInput
end
