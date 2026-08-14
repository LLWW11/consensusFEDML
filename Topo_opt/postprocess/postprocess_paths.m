function paths = postprocess_paths()
%POSTPROCESS_PATHS 返回拓扑后处理的代码、输入和输出目录。
%   paths = POSTPROCESS_PATHS() 将 Topo_opt 和 postprocess 加入 MATLAB
%   搜索路径，并返回 topology_directory、postprocess_directory 和
%   output_directory。后处理默认读取 Topo_opt 中的原始结果，输出统一保存到
%   postprocess 目录。

postprocess_directory = fileparts(mfilename('fullpath'));
topology_directory = fileparts(postprocess_directory);
addpath(topology_directory, postprocess_directory);

paths = struct();
paths.topology_directory = topology_directory;
paths.postprocess_directory = postprocess_directory;
paths.output_directory = postprocess_directory;
end
