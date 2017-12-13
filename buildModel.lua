function model_residual(gpu, dropout)
    local input_h = 126
    local input_w = 126
    local input_channel = 1
    local conv_kernel = {64, 64, 64, 256 }
    local conv_size = {7, 3, 3, 3}
    local conv_tride = {1, 1, 1, 1}
    local conv_pad = {0, 1, 1, 1}
    local pool_size = {2, 2, 2}
    local pool_tride = {2, 2, 2}
    local LRN_size = 5
    local LRN_alpha = 0.0001
    local LRN_beta = 0.75
    local LRN_k = 2
    local ConvNet = nn.SpatialConvolution
    if gpu then
        ConvNet = nn.SpatialConvolutionMM
    end

    local c1 = nn.Sequential()

    local unit1 = nn.Sequential()
    unit1:add(ConvNet(input_channel, conv_kernel[1], conv_size[1], conv_size[1], conv_tride[1], conv_tride[1], conv_pad[1], conv_pad[1]))
    unit1:add(nn.ReLU())
    
    c1:add(unit1)
    c1:add(nn.SpatialCrossMapLRN(LRN_size, LRN_alpha, LRN_beta, LRN_k))
    c1:add(nn.SpatialMaxPooling(pool_size[1], pool_size[1], pool_tride[1], pool_tride[1]))


    local unit2 = nn.Sequential()
    local cat2 = nn.ConcatTable()
    unit2:add(ConvNet(conv_kernel[1], conv_kernel[2], conv_size[2], conv_size[2], conv_tride[2], conv_tride[2], conv_pad[2], conv_pad[2]))
    unit2:add(nn.ReLU())
    cat2:add(unit2)
    cat2:add(nn.Identity())

    c1:add(cat2)
    c1:add(nn.CAddTable())
    c1:add(nn.ReLU())

    c1:add(nn.SpatialCrossMapLRN(LRN_size, LRN_alpha, LRN_beta, LRN_k))
    c1:add(nn.SpatialMaxPooling(pool_size[2], pool_size[2], pool_tride[2], pool_tride[2]))


    local unit3 = nn.Sequential()
    local cat3 = nn.ConcatTable()
    unit3:add(ConvNet(conv_kernel[2], conv_kernel[3], conv_size[3], conv_size[3], conv_tride[3], conv_tride[3], conv_pad[3], conv_pad[3]))
    unit3:add(nn.ReLU())
    cat3:add(unit3)
    cat3:add(nn.Identity())

    c1:add(cat3)
    c1:add(nn.CAddTable())
    c1:add(nn.ReLU())

    c1:add(nn.SpatialCrossMapLRN(LRN_size, LRN_alpha, LRN_beta, LRN_k))
    c1:add(nn.SpatialMaxPooling(pool_size[3], pool_size[3], pool_tride[3], pool_tride[3]))

    if gpu then
        c1:cuda()
    end
    local c2 = c1:clone('weight', 'bias', 'gradWeight', 'gradBias')

    local two = nn.ParallelTable()
    two:add(c1)
    two:add(c2)
    if gpu then
        two:cuda()
    end

    local merge = nn.Sequential()
    merge:add(nn.CSubTable())
    merge:add(nn.Abs())

    local c3 = nn.Sequential()
    c3:add(ConvNet(conv_kernel[3], conv_kernel[4], conv_size[4], conv_size[4], conv_tride[4], conv_tride[4], conv_pad[4], conv_pad[4]))
    c3:add(nn.ReLU())

    local fully = conv_kernel[4] * 15 * 15
    c3:add(nn.Reshape(fully))
    c3:add(nn.Dropout(dropout))
    c3:add(nn.Linear(fully, 2))
    c3:add(nn.LogSoftMax())

    local model = nn.Sequential()
    model:add(two)
    model:add(merge)
    model:add(c3)
    if gpu then
        model:cuda()
    end
    local crit = nn.ClassNLLCriterion()
    return model, crit
end