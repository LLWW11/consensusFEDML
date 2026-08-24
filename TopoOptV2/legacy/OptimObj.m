function value = OptimObj(group_number, client_number_per_group, gamma, layers)

time1 = layers*16+100; %传输的时间

max_possible_clients = 37;
max_possible_time  = 200;

if isempty(client_number_per_group)
    time2 = group_number*2;
    client_number_ = 0;
else
    time2 = (group_number + max(client_number_per_group))*2; %聚合时间
    client_number_ = sum(client_number_per_group);
end

time = time1+time2;

% 越大越好
% value = client_number_ /(time1+time2);  % 补充
% value = client_number_ + gamma/(time1+time2);
% value = 0.3*client_number_/max_possible_clients + 0.7*max_possible_time/time;
value = client_number_/max_possible_clients - time/max_possible_time;

% 越小越好
% value = time1 +time2;

