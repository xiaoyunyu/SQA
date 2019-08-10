require '..'


function create_embed(dictlen,veclen)
	local embed = cudacheck(torch.Tensor(dictlen,veclen):zero())
	local file = io.open('glove.6B.300d.txt','rb')
	for i=1,dictlen do
		local line = file:read()
		local tab = string.split(line,' ')
		table.remove(tab,1)
		embed[{i}] = torch.Tensor(tab)
	end
	print(embed[{dictlen}])
	torch.save('test.t7',embed)
	file:close()
end

--create_embed(100000,300)

function index_embed(num_word,veclen)
	local embed = cudacheck(torch.CudaTensor(num_word,veclen):zero())
	local embed_db = torch.load('word.glove100k.t7')
	local relWord = io.open('relWordVocab/relWord_id.txt','rb')
	local totNum = 0
	while true do
		local idx = relWord:read()
		if idx == nil then break end
		totNum = totNum + 1
		idx = tonumber(idx)+1
		embed[{totNum}]=embed_db[{idx}]
	end
	assert(num_word == totNum)
	torch.save('relWord_embed.t7',embed)
	relWord:close()
end

index_embed(7523,300)

function rel_embed(num_word,veclen,path)
	local embed = cudacheck(torch.Tensor(num_word,veclen):zero())
	local relWord = io.open(path,'rb')
	for i =1,num_word do
		inp = relWord:read()
		print(i)
		print(inp)
		local fields=stringx.split(inp,' ')
		embed[i]=torch.Tensor(fields)
	end
	torch.save(path..'.t7',embed)
	relWord:close()
end
-- rel_embed(7524,256,'./relEmbed.orthogonal')