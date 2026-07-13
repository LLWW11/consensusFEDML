function  [IdleTime] = UpdataIdleTime_v3(TSML_BdwMat, time, Nodes, deadline, limited_layers)
Layers = length(time);

if Layers > limited_layers
    TSML_BdwMat(:,:,limited_layers+1) = zeros(Nodes,Nodes);
    TSML_BdwMat = TSML_BdwMat(:,:,1:(limited_layers+1));
    time = time(1:limited_layers);
    time = [time, time(1) + deadline];
    Layers = limited_layers+1;
end
tmp_bdmat = reshape(TSML_BdwMat,Nodes^2,Layers);
IdleTime = zeros(Nodes^2,Layers);
for k = 1:Nodes^2
    position_of_0 = find(tmp_bdmat(k,:)==0);
    position_of_1 = find(tmp_bdmat(k,:)>0);

    size_of_0 = size(position_of_0,2);
    size_of_1 = size(position_of_1,2);

    if size_of_1 >= 1 && size_of_0>=1
        for i = 1:size_of_1
            tmp = position_of_0 - position_of_1(i).*ones(1,size_of_0) ;
            Close_To_1 = find(tmp>0,2);
            if isempty(Close_To_1)
                IdleTime(k,position_of_1(i)) = time(1) + deadline - time(position_of_1(i));

%                 if find(IdleTime > deadline)
%                     disp('debug')
%                 end

            else
                IdleTime(k,position_of_1(i)) = time(tmp(Close_To_1(1))+position_of_1(i)) - time(position_of_1(i));

%                 if find(IdleTime > deadline)
%                     disp('debug')
%                 end

            end
        end
    elseif size_of_1 >= 1 && size_of_0 < 1
        IdleTime(k,position_of_1) = time(1) + deadline - time(position_of_1);

%         if find(IdleTime > deadline)
%             disp('debug')
%         end

    elseif size_of_1 < 1 && size_of_0 >= 1
        IdleTime(k,position_of_0) = 0;
    else
        error('Error in IdleTime Updata !!!');
    end

    %     % --debug
    %         if length(find(IdleTime < 0)) > 0
    %             disp('error, debug');
    %         end
end



IdleTime = reshape(IdleTime,Nodes,Nodes,Layers);
% end