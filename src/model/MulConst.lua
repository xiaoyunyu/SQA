local MulConst, parent = torch.class('MulConst', 'nn.Module')

function MulConst:__init(const)
    parent.__init(self)
    self.gradInput = torch.Tensor()
    self.const = const
end 

function MulConst:updateOutput(input)
    self.output = torch.cmul(input, self.const)
    return self.output
end

function MulConst:updateGradInput(input, gradOutput)
    self.gradInput = torch.cmul(gradOutput, self.const)
    return self.gradInput
end
