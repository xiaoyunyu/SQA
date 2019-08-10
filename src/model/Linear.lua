local Linear, parent = torch.class('Linear', 'nn.Module')

function Linear:__init(inputSize, outputSize, sqz, bias)
   parent.__init(self)
   local bias = ((bias == nil) and true) or bias
   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   if bias then
      self.bias = torch.Tensor(outputSize)
      self.gradBias = torch.Tensor(outputSize)
   end
   self:reset()
   self.sqz = sqz or  ((sqz == nil) and false) 
end

function Linear:reset(stdv)
   if stdv then
      stdv = stdv
   else
      stdv = math.sqrt(6)/math.sqrt(self.weight:size(1)+self.weight:size(2))
   end
   -- if nn.oldSeed then
   --    for i=1,self.weight:size(1) do
   --       self.weight:select(1, i):apply(function()
   --          return torch.uniform(-stdv, stdv)
   --       end)
   --    end
   --    if self.bias then
   --       for i=1,self.bias:nElement() do
   --          self.bias[i] = torch.uniform(-stdv, stdv)
   --       end
   --    end
   -- else
   self.weight:uniform(-stdv, stdv)
   if self.bias then self.bias:fill(0) end
   -- end
   return self
end

function Linear:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.weight:size(1))
      if self.bias then self.output:copy(self.bias) else self.output:zero() end
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self.addBuffer = self.addBuffer or input.new()
      if self.addBuffer:nElement() ~= nframe then
         self.addBuffer:resize(nframe):fill(1)
      end
      self.output:addmm(0, self.output, 1, input, self.weight:t())
      if self.bias then self.output:addr(1, self.addBuffer, self.bias) end
   elseif input:dim() >= 3 then
      -- computation happens in 2D views
      local dInput = input:view(-1, self.weight:size(2))
      local nframe = dInput:size(1)
      self.output:resize(nframe, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self.addBuffer = self.addBuffer or input.new()
      if self.addBuffer:nElement() ~= nframe then
         self.addBuffer:resize(nframe):fill(1)
      end
      self.output:addmm(0, self.output, 1, dInput, self.weight:t())
      if self.bias then self.output:addr(1, self.addBuffer, self.bias) end

      -- re-view output according to the input size
      local sizes = input:size()
      sizes[input:dim()] = self.weight:size(1)
      self.output = self.output:view(sizes)
      if self.sqz then
         self.output=self.output:squeeze(input:dim())
      end
   else
      error('input must be 1D, 2D or 3D Tensor')
   end

   return self.output
end

function Linear:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      elseif input:dim() >= 3 then
         local dGradInput = self.gradInput:view(-1, self.weight:size(2))
         local dGradOutput = gradOutput:view(-1, self.weight:size(1))
         dGradInput:addmm(0, 1, dGradOutput, self.weight)
      end

      return self.gradInput
   end
end

function Linear:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      if self.bias then self.gradBias:add(scale, gradOutput) end
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      if self.bias then
         self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      end
   elseif input:dim() == 3 then      
      local dGradOutput = gradOutput:view(-1, self.weight:size(1))
      local dInput      = input:view(-1, self.weight:size(2))
      local nframe      = dGradOutput:size(1)
      self.gradWeight:addmm(scale, dGradOutput:t(), dInput)
      if self.bias then
         self.addBuffer = self.addBuffer or input.new()
         if self.addBuffer:nElement() ~= nframe then
            self.addBuffer:resize(nframe):fill(1)
         end
         self.gradBias:addmv(scale, dGradOutput:t(), self.addBuffer)
      end
   end
end

-- we do not need to accumulate parameters when sharing
--Linear.sharedAccUpdateGradParameters = Linear.accUpdateGradParameters


function Linear:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1)) ..
      (self.bias == nil and ' without bias' or '')
end
