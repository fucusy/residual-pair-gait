require 'tool'

local pro = {1,0.5,0.9,0.99,0.67,0.12,0.8}
local res = {true, false, true, true, false, true, false}
local eer = cal_eer(pro, res)
info('equal error rate is %.02f', eer)

info('add 0.9 to pro, add true to res')
table.insert(pro, 0.9)
table.insert(res, true)
local eer = cal_eer(pro, res)
info('equal error rate is %.02f', eer)
