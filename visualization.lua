require 'torch'
require 'nn'
require 'nnx'
require 'optim'

require 'image'
require 'paths'
require 'rnn'

cmd = torch.CmdLine()
cmd:option('-modelname','max','max,wuzifeng,wuzifengcopy model name you want to load')
cmd:option('-dropout',0.0,'fraction of dropout to use between layers')
cmd:option('-seed',1,'random seed')
cmd:option('-datapath','/Volumes/Passport/data/gait-rnn', 'base data path')
cmd:option('-loadmodel', '', 'load fullmodel, rnn model, cnn model')
cmd:option('-gpu', false, 'use GPU')
cmd:option('-gpudevice', 1, 'set gpu device')
cmd:option('-debug', false, 'debug? this will output more information which will slow the program')
cmd:option('-diffdim', 0, 'additional diff dimension')
cmd:option('-imageid1', '6218964-IDList_OULP-C1V1-A-55_Probe.csv', 'input image1')
cmd:option('-imageid2', '6218964-IDList_OULP-C1V1-A-85_Probe.csv', 'input image2')

opt = cmd:parse(arg)
print(opt)

require 'buildModel'
require 'train'
require 'test'
require 'tool'

local prepDataset = require 'prepareDataset'

-- set the GPU
if opt.gpu then
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(opt.gpudevice)
end

torch.manualSeed(opt.seed)
if opt.gpu then
    cutorch.manualSeed(opt.seed)
end

local dataset = prepDataset.prepareDatasetOULP(opt.datapath, opt.modelname)

for i, item in ipairs({'train', 'val', 'test'}) do
    local item_count = dataset[item]._item_count
    local uniq_count = dataset[item]._uniq_item_count
    info('train data instances %05d, uniq  %04d', item_count, uniq_count)
end

local model, crit

-- build the model
if opt.modelname == 'wuzifeng' then
    model, crit = model_wuzifeng(opt.gpu, opt.dropout)
elseif opt.modelname == 'max' then
    model, crit = model_max(opt.gpu, opt.dropout)
elseif opt.modelname == 'mean' then
    model, crit = model_mean(opt.gpu, opt.dropout)
elseif opt.modelname == 'wuzifengcopy' then
    model, crit = model_wuzifengcopy(opt.gpu, opt.dropout)
elseif opt.modelname == 'maxupdate' then
    model, crit = model_maxupdate(opt.gpu, opt.dropout)
end
print(model)
if opt.mode == 'evaluate' then
    model = torch.load(opt.loadmodel)
    info('loaded model from %s', opt.loadmodel)
end

local img1 = dataset['train']:load_item(opt.imageid1)
local img2 = dataset['train']:load_item(opt.imageid2)

local dirname = 'debug'
os.execute(string.format("mkdir -p %s", dirname))

local cnn1 = model:get(1):get(1):get(1)
print(cnn1)

local maxpool = model:get(1):get(1)
print(maxpool)
local imgs_tbl = {img1, img2}


for t, imgs in ipairs(imgs_tbl) do
    for i, img in ipairs(imgs) do
        for j=1,img:size(1) do
            filename = string.format('%s/img%d_%02dth_%dl.png', dirname,t,i,j)
            image.save(filename, img[j]:reshape(1, img:size(2), img:size(3)))
        end

        local res = cnn1:forward({img})
        res = res[1]
        res = res:reshape(64, 27, 27)
        local max_item = res:max()
        local tmp = put2one(res) / max_item
        local filename = string.format(
            '%s/img%d_%02dth_cnn1_feature_map.png'
            , dirname, t, i)
        image.save(filename, tmp)
    end
end

for t, imgs in ipairs(imgs_tbl) do
    local res = maxpool:forward(imgs)
    local max_item = res:max()
    local tmp = put2one(res) / max_item
    local filename = string.format('%s/img%d_pool_feature_map.png'
                    , dirname, t)
    image.save(filename, tmp)
end
