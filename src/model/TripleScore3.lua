function TripleProd()
	--tarMat[batchsize*length*hiddensize],posMat[batchsize*1*hiddensize],negMat[batchsize*neg*hidden]
    local tarMat = nn.Identity()()
    local relMat = nn.Identity()()
    --local tarMat      = nn.Replicate(negBatchSize) (tarVec)
    local scoreMat = nn.MM(false,true) ({relMat, tarMat})

    return nn.gModule({tarMat, relMat}, {scoreMat})
end

function TripleProd2()
    --tarMat[batchsize*length*hiddensize],posVec[batchsize*hiddensize],negMat[batchsize*neg*hidden]
    local tarMat = nn.Identity()()
    local posVec = nn.Identity()()
    local negMat = nn.Identity()()
    local scoreVecPos = nn.MM(false,true) ({posVec, tarMat})
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

function TripleScore_vec(negBatchSize,active,hiddenSize)
    active = ((active==nil) and false) or active
    if active then
        local linearLayer = Linear(hiddenSize,hiddenSize,false,false)
        local tarVec = nn.Identity()()
        local candMat = nn.Identity()()
        local candMat_poj = nn.Dropout(0.3)(linearLayer(candMat))
        local bias = nn.Add(1,true)
        bias.bias:fill(0)
        local tarMat  = nn.Replicate(negBatchSize+1) (tarVec)
        local scoreVec = nn.View(-1)(nn.Sigmoid()(bias(BatchDot()({tarMat, candMat_poj}))))
        return nn.gModule({tarVec, candMat}, {scoreVec})
    else
        local tarVec = nn.Identity()()
        local posVec = nn.Identity()()
        local negMat = nn.Identity()()
        
        local scoreVecPos = BatchDot() ({tarVec, posVec})
        --negsize*batchsize
        local scoreMatPos = nn.Replicate(negBatchSize) (scoreVecPos)

        local tarMat      = nn.Replicate(negBatchSize) (tarVec)
        local scoreMatNeg = BatchDot() ({tarMat, negMat})
        return nn.gModule({tarVec, posVec, negMat}, {scoreMatPos, scoreMatNeg})
    end
end

function TripleScore_vec4test(negBatchSize)
    local tarVec = nn.Identity()()
    local posVec = nn.Identity()()
    local negMat = nn.Identity()()

    local tarMat      = nn.Replicate(negBatchSize) (tarVec)
    local scoreMatNeg = BatchDot() ({tarMat, negMat})

    return nn.gModule({tarVec, negMat}, {scoreMatNeg})
end