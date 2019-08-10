function TripleProd()
	--tarMat[batchsize*length*hiddensize],posVec[batchsize*hiddensize],negMat[batchsize*neg*hidden]
    local tarMat = nn.Identity()()
    local posVec = nn.Identity()()
    local negMat = nn.Identity()()
    local _posVec = nn.Replicate(1,2)(posVec)
    local scoreVecPos = nn.MM(false,true) ({_posVec, tarMat})
    --local scoreMatPos = nn.Replicate(negBatchSize) (scoreVecPos)

    --local tarMat      = nn.Replicate(negBatchSize) (tarVec)
    local scoreMatNeg = nn.MM(false,true) ({negMat, tarMat})

    return nn.gModule({tarMat, posVec, negMat}, {scoreVecPos, scoreMatNeg})
end

function TripleScore(negBatchSize)
    --posVec[batchsize*1],negMat[batchsize*neg]
    local posVec = nn.Identity()()
    local negMat = nn.Identity()()

    local _posVec = nn.Squeeze(2)(posVec)
    local scoreVecPos = nn.Replicate(negBatchSize,2) (_posVec)
    --local scoreMatPos = nn.Replicate(negBatchSize) (scoreVecPos)

    --local tarMat      = nn.Replicate(negBatchSize) (tarVec)

    return nn.gModule({posVec,negMat}, {scoreVecPos, negMat})
end 
function normLayer(opt)
    local input = nn.Identity()()
    local expa = nn.View(-1,opt.relEmbedSize)(input)
    local norma = nn.Normalize(2,1e-6)(expa)

    return nn.gModule({input},{norma})
end