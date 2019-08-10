local test, parent = torch.class('test', 'nn.Module')
function test:__init(num,shared)
    parent.__init(self)
    self.num = num
    self.weight = torch.Tensor(2,num)
    self.gradWeight = torch.Tensor(2,num)
    self.inp = {}
    for i = 1,shared do
        self.inp[i]=torch.rand(2,2)
    end
    self.i=1
    self.pt = self.inp[self.i]
end 