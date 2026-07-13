total_util = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8];
gnum_fix_3nodes = [3, 2.99, 3, 2.92, 2.93, 2.78, 2.4, 1.82];
gnum_fix_4nodes = [4, 3.96, 3.98, 3.95, 3.92, 3.75, 3.5, 2.53];
gnum_fix_5nodes = [4.95, 5, 5, 4.895, 4.8, 4.535, 4.14, 2.495];
gnum_fix_6nodes = [5.97, 5.95, 5.96, 5.912, 5.855, 5.644, 4.934, 3.254];
gnum_fix_8nodes = [8, 7.91, 7.89, 7.74, 7.51, 7.07, 6.02, 3.91];

cnum_fix_3nodes = [36.86, 36.67, 36.78, 35.15, 34.67, 32.38, 24.04, 12.2];
cnum_fix_4nodes = [37.00, 36.58, 36.74, 36.24, 35.99, 33.77, 30.12, 17.04];
cnum_fix_5nodes = [36.62, 36.97, 36.92, 36.1, 34.79, 32.125, 27.86, 13.40];
cnum_fix_6nodes = [37.00, 37.00, 36.67, 36.10, 35.04, 33.32, 28.56, 14.64];
cnum_fix_8nodes = [36.97, 37.00, 36.95, 35.87, 34.53, 32.54, 27.59, 17.48];

time_fix_3nodes = [163.02, 169.66, 176.38, 191, 201.02, 203.64, 198.92, 179.76];
time_fix_4nodes = [156.80, 165.54, 172.60, 181.34, 193.90, 196.84, 201.42, 185.08];
time_fix_5nodes = [157.34, 165.47, 174.68, 185.76, 191.27, 200.77, 203.02, 178.26];
time_fix_6nodes = [157.38, 161.52, 173.08, 179.32, 190.22, 197.56, 203.96, 182.00];
time_fix_8nodes = [156.62, 164.00, 174.52, 178.44, 191.94, 196.82, 196.34, 187.12];
%% obj = client_num + gamma/time
g1k_cnum = [36.98, 36.21, 36.86, 35.69, 35.14, 32, 29.04, 17.21];
g1k_time = [184.38, 190.68, 206.24, 208.46, 213.9, 212.06, 209.68, 185.24];

g4k_cnum = [36.712, 36.84, 36.468, 35.928, 34.43, 32.69, 26.436, 13.764];
g4k_time = [185.348, 195.212, 203.416, 210.576, 214.176, 215.896, 203.796, 171.144];

g6k_cnum = [36.884, 36.206, 35.946, 35.222, 33.436, 31.454, 25.434, 12.196];
g6k_time = [185.624, 193.256, 200.988, 208.496, 212.504, 214.712, 202.58, 164.748];

g8k_cnum = [36.432, 36.07, 34.846, 34.954, 33.554, 28.592, 23.72, 10.62];
g8k_time = [184.54, 191.492, 199.596, 209.616, 214.164, 207.728, 199.892, 159.616];

g10k_cnum = [36.264, 35.944, 34.818, 34.158, 31.244, 26.904, 21.054, 9.562];
g10k_time = [183.596, 193.348, 199.948, 208.644, 209.528, 204.764, 193.592, 157.784];

g100k_cnum = [33.6, 28.06, 28.13, 17.66, 14.07, 4.96, 0.2, 0.15];
g100k_time = [178.84, 174.4, 183.72, 162.46, 149.58, 123.16, 104.02, 103.1];

% obj1 = client_number / time;
obj1_cnum = [36.72, 36.79, 36.48, 35.69, 32.91, 31.06, 24.67, 15.3];
obj1_time = [183.96, 195.24, 202.32, 212.06, 209.94, 216.32, 204.02, 188.66];
%% 分开time和clients优化
% time_pre < time_current  &&  pre_client_num <= current_client_num
% time_pre < time_current && pre_client_num < current_client_num

b0_cnum = [37.00, 37.00, 36.94, 36.87, 35.39, 33.32, 28.57, 18.52]; % 带等于号
b1_cnum = [37.00, 37.00, 36.70, 36.21, 35.31, 34.19, 29.85, 16.55]; % 不带等于号

b0_time = [143.88, 149.06, 158.48, 169.08, 176.20, 182.14, 196.24, 184.30];
b1_time = [157.48, 162.70, 174.94, 181.16, 189.52, 200.38, 203.30, 182.94];
%% 8.23
cnum_plan2 = [36.26, 37, 36.97, 36.88, 34.96, 32.78, 31.74, 20.1];
gnum_plan2 = [5.9, 6.01, 6.02, 6.09, 5.89, 5.81, 5.99, 4.33];
time_plan2 = [154.9, 161.9, 175.1, 183.2, 190.9, 191.6, 204.8, 191.9];

cnum_plan1 = [36.26, 37, 36.74, 36.24, 34.53, 34.19, 30.53, 17.48];
gnum_plan1 = [5.89, 6.01, 5.98, 5.95, 5.8, 5.91, 5.62, 3.69];
time_plan1 = [155.3, 164, 172.6, 181.3, 191.9, 203.8, 202.9, 187.1];

cnum_plan2_1 = [36.26, 37, 36.97, 36.88, 34.96, 32.78, 31.74, 20.1];%带等于号
gnum_plan2_1 = [5.92, 6.02, 6.01, 6.05, 5.91, 5.84, 5.99, 4.36];
time_plan2_1 = [153.8, 160.9, 172.1, 180.2, 188.4, 187, 200.1, 187.1];

cnum_plan1_1 = [36.26, 37, 36.74, 36.24, 34.53, 34.19, 30.53, 17.48];
gnum_plan1_1 = [5.86, 6.02, 5.96, 5.94, 5.83, 5.9, 5.63, 3.66];
time_plan1_1 = [154.3, 164.1, 171.5, 179.2, 188.6, 200.6, 198.2, 181];

%% 归一化 value = 0.3*client_number_/max_possible_clients + 0.7*max_possible_time/time;
cnum_obj37 = [36.97, 36.97, 36.35, 35.78, 35.95, 32.07, 26.50, 15.73];
cnum_obj46 = [36.99, 36.21, 36.35, 36.12, 35.00, 33.85, 28.42, 17.50];
cnum_obj55 = [36.60, 36.78, 36.99, 36.61, 34.51, 33.62, 27.20, 17.63];
cnum_obj64 = [37.00, 37.00, 36.90, 36.47, 34.56, 34.38, 27.68, 17.39];
cnum_obj73 = [36.99, 36.95, 36.66, 36.05, 35.88, 33.47, 27.95, 16.54];

time_obj37 = [156.62, 164.60, 175.64, 182.76, 189.88, 198.92, 198.90, 190.60];
time_obj46 = [158.46, 162.86, 175.64, 186.46, 194.70, 200.34, 203.98, 192.76];
time_obj55 = [154.98, 164.70, 172.28, 179.16, 192.40, 201.30, 201.46, 194.86];
time_obj64 = [157.50, 166.92, 177.10, 187.36, 192.98, 200.82, 203.60, 193.84];
time_obj73 = [157.44, 165.86, 174.90, 182.48, 193.56, 201.46, 201.46, 195.00];

gnum_obj37 = [6.00, 6.00, 5.93, 6.01, 5.97, 5.47, 4.71, 3.49];
gnum_obj46 = [6.03, 6.04, 5.93, 6.03, 5.87, 5.80, 5.06, 3.55];
gnum_obj55 = [5.94, 5.98, 6.00, 6.02, 6.09, 5.71, 4.82, 3.69];
gnum_obj64 = [6.00, 6.20, 6.09, 6.25, 6.12, 5.92, 4.91, 3.63];
gnum_obj73 = [6.16, 6.00, 5.97, 5.97, 5.97, 5.67, 4.91, 3.50];

%% 0.3*client_number_/max_possible_clients - 0.7*time/max_possible_time;
cnum_objv2_19 = [37.00, 36.48, 36.46, 36.38, 35.09, 31.44, 27.46, 11.58];
cnum_objv2_37 = [36.99, 36.58, 36.92, 36.80, 35.35, 33.25, 28.40, 17.46];
cnum_objv2_55 = [36.97, 36.26, 36.92, 36.18, 36.07, 35.20, 29.24, 19.96];
cnum_objv2_73 = [37.00, 36.95, 36.62, 36.81, 36.73, 35.11, 30.63, 21.48];
cnum_objv2_91 = [37.00, 37.00, 36.82, 36.78, 36.15, 35.41, 30.85, 19.71];

time_objv2_19 = [147.50, 151.14, 160.16, 167.12, 176.36, 180.90, 188.32, 151.64];
time_objv2_37 = [146.46, 152.72, 159.42, 167.90, 175.64, 187.44, 190.30, 168.30];
time_objv2_55 = [146.04, 148.50, 160.12, 169.96, 181.56, 188.06, 192.26, 178.50];
time_objv2_73 = [147.30, 152.02, 158.22, 166.98, 178.54, 191.60, 197.68, 190.32];
time_objv2_91 = [145.52, 152.28, 164.28, 169.56, 178.82, 195.64, 198.26, 189.56];

gnum_objv2_19 = [8.32, 8.81, 8.68, 8.54, 8.09, 6.94, 5.57, 2.28];
gnum_objv2_37 = [8.41, 8.76, 8.80, 8.47, 8.33, 7.66, 6.63, 3.82];
gnum_objv2_55 = [8.23, 8.74, 8.64, 8.48, 8.63, 7.95, 6.37, 4.45];
gnum_objv2_73 = [8.47, 8.91, 8.70, 8.64, 8.73, 8.15, 6.95, 4.47];
gnum_objv2_91 = [8.30, 8.66, 8.71, 8.69, 8.74, 7.62, 6.98, 4.09];

%% only time and only client nums
cnum_only_time = [36.23, 36.83, 36.15, 35.51, 33.93, 29.06, 23.93, 7.62];
cnum_only_client = [36.8, 36.99, 36.96, 36.78, 36.25, 35.94, 31.14, 20.9];

time_only_time = [143.3, 148.92, 161.08, 167.12, 177.56, 181.02, 182.98, 146.64];
time_only_client = [157.63, 163.59, 172.92, 185.03, 190.18, 196.83, 204.77, 192.75];

gnum_only_time = [7.3, 7.7, 7.81, 7.63, 7.17, 6.14, 5.11, 1.68];
gnum_only_client = [5.97, 6.02, 6.035, 5.975, 6.005, 6, 5.98, 4.475];
%% 绘图
% 筛选出total_util为0.4, 0.6, 和0.8的数据索引
indices = total_util == 0.4 |total_util == 0.5 | total_util == 0.6 | total_util == 0.7 |total_util == 0.8;
tutil = total_util(indices);

figure(1)
bar(tutil, [cnum_only_time(indices)', cnum_only_client(indices)',cnum_objv2_55(indices)',cnum_fix_6nodes(indices)'], 'group');
title('Cilent Num');
xlabel('Utilization');
% ylabel('Values');
legend({'Only time', 'Only client','Combine','Fixed 6 Edges'}, 'Location', 'best');
ax = gca;
ax.XTick = [0.4, 0.5, 0.6, 0.7, 0.8];
grid on;hold on;

figure(2)
bar(tutil, [time_only_time(indices)', time_only_client(indices)',time_objv2_55(indices)',time_fix_6nodes(indices)'], 'group');
title('Time');
xlabel('Utilization');
% ylabel('Values');
legend({'Only time', 'Only client','Combine','Fixed 6 Edges'}, 'Location', 'best');
ax = gca;
ax.XTick = [0.4, 0.5, 0.6, 0.7, 0.8];
grid on;hold on;

figure(3)
bar(tutil, [gnum_only_time(indices)', gnum_only_client(indices)',gnum_objv2_55(indices)',gnum_fix_6nodes(indices)'], 'group');
title('Group Num');
xlabel('Utilization');
% ylabel('Values');
legend({'Only time', 'Only client','Combine','Fixed 6 Edges'}, 'Location', 'best');
ax = gca;
ax.XTick = [0.4, 0.5, 0.6, 0.7, 0.8];
grid on;hold on;

figure(4)
A = [cnum_fix_3nodes(indices)',cnum_fix_4nodes(indices)', cnum_fix_5nodes(indices)',cnum_fix_6nodes(indices)',cnum_fix_8nodes(indices)',cnum_objv2_55(indices)'];
bar(tutil,A,'group');
title('Clients Num');
xlabel('Utilization');  
legend({'3 Nodes','4 Nodes', '5 Nodes','6 Nodes','8 Nodes','Combine'}, 'Location', 'best');
ax = gca;
ax.XTick = [0.4,0.5, 0.6,0.7, 0.8];
grid on;hold on;

figure(5)
B = [time_fix_3nodes(indices)',time_fix_4nodes(indices)',time_fix_5nodes(indices)', time_fix_6nodes(indices)',time_fix_8nodes(indices)',time_objv2_55(indices)'];
bar(tutil, B,'group');
title('Time');
xlabel('Utilization');  
legend({'3 Nodes','4 Nodes', '5 Nodes','6 Nodes','8 Nodes','Combine'}, 'Location', 'best');
ax = gca;
ax.XTick = [0.4, 0.5, 0.6, 0.7, 0.8];
grid on;hold on;

figure(6)
C = [gnum_fix_3nodes(indices)',gnum_fix_4nodes(indices)',gnum_fix_5nodes(indices)', gnum_fix_6nodes(indices)',gnum_fix_8nodes(indices)',gnum_objv2_55(indices)'];
bar(tutil, C,'group');
title('group num');
xlabel('Utilization');  
legend({'3 Nodes','4 Nodes', '5 Nodes','6 Nodes','8 Nodes','Combine'}, 'Location', 'best');
ax = gca;
ax.XTick = [0.4, 0.5, 0.6, 0.7, 0.8];
grid on;hold on;

% figure(1)
% plot(total_util, cnum_plan1,'-d',...
%      total_util, cnum_plan2,'-s',....
%      total_util, cnum_plan1_1,'-hexagram',...
%      total_util, cnum_plan2_1,'-v');
% title('client num')
% xlabel('Utilization');
% ylabel('number');
% legend('plan1','plan2','plan1(with =)','plan2(with =)');
% grid on;
% 
% figure(2)
% plot(total_util, time_plan1,'-d',...
%      total_util, time_plan2,'-s',....
%      total_util, time_plan1_1,'-hexagram',...
%      total_util, time_plan2_1,'-v');
% title('time')
% xlabel('Utilization');
% ylabel('second');
% legend('plan1','plan2','plan1(with =)','plan2(with =)');
% grid on;
% 
% figure(3)
% plot(total_util, gnum_plan1,'-d',...
%      total_util, gnum_plan2,'-s',....
%      total_util, gnum_plan1_1,'-hexagram',...
%      total_util, gnum_plan2_1,'-v');
% title('group num')
% xlabel('Utilization');
% ylabel('number');
% legend('plan1','plan2','plan1(with =)','plan2(with =)');
% grid on;
