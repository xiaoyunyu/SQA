local BatchComb, parent = torch.class('BatchComb', 'nn.Module')

function BatchComb:__init()
    parent.__init(self)
    self.gradInput = {torch.Tensor(), torch.Tensor()}

end 

function BatchComb:updateOutput(input)
    self.output = torch.add(unpack(input))
    return self.output
end

function BatchComb:updateGradInput(input, gradOutput)
	self.gradInput[1]:resize(gradOutput:size()):zero()
    self.gradInput[1]:add(gradOutput)
	self.gradInput[2]:resize(gradOutput:size()):zero()
    self.gradInput[2]:add(gradOutput)
    return self.gradInput
end
