%作用：假设avg_num为0.2，产生一个0-1之间，且平均数为0.2的随机值
function num = gen_rand_num(avg_num)  

rand_num = rand;
rate = 1-avg_num;
if rand_num < rate
    num = avg_num * rand; %产生一个(0, avg_num)之间的小数
else
    num = (1-avg_num) * rand + avg_num; %产生一个(avg_num, 1)之间的小数
end

end