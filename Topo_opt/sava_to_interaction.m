% 1. 从原文件加载指定变量
file_name = 'E:\1_JOCN\Code\聚合时间\dynEdge_dynU_v2_3\Result\normalization\-\55\result-U-v2-Obj1_55_epoch_1000 .mat';
load(file_name, 'client_num_HFLSnF', 'group_num_HFLSnF');
client_num_ = client_num_HFLSnF(:,5);
group_num_ = group_num_HFLSnF(:,5);

group_num = zeros(1000,1);
for i = 1:1000
    group = group_num_{i,1};
    group_num(i) = group{1,1};
end
max_group_num = max(group_num);
indices = find(group_num == max_group_num);
client_num = zeros(1000,max_group_num);
for i = 1:1000
    client1 = client_num_{i,1};
    current_client_num = client1{1,1};
    current_client_num(current_client_num ==0) = [];
    % if length(current_client_num) ~= group_data_(i)
    %     disp(i);
    %     % break;
    % end
    client_num(i,1:group_num(i)) = current_client_num;
    if length(current_client_num) ~= group_num(i)
        client_num(i,group_num(i):max_group_num) = 0;
    end
end

% 2. 将这些变量保存到新文件
save('subset_data.mat', 'client_num', 'group_num');