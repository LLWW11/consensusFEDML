function [time]= gen_tsmlg_time_axis_v1(num_layers, mean_time_interval)

% exprnd(mean_time_interval), time(1)=1;
time = zeros(num_layers, 1);
time(1) = 1;
for i = 1:num_layers-1
    rand_num = exprnd(mean_time_interval);
    time(i+1) = time(i) + round(rand_num);
end

end