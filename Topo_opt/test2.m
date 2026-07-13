clear;clc;


adj_mat = [                       %   6 node simple topology
    0,1,0,0,0,1;
    1,0,1,0,0,1;
    0,1,0,1,1,0;
    0,0,1,0,1,0;
    0,0,1,1,0,1;
    1,1,0,0,1,0;];
adj_mat = [                       %14 node NSF
    0	1	1	1	0	0	0	0	0	0	0	0	0	0;
    1	0	1	0	0	0	0	1	0	0	0	0	0	0;
    1	1	0	0	0	1	0	0	0	0	0	0	0	0;
    1	0	0	0	1	0	0	0	0	1	0	0	0	0;
    0	0	0	1	0	1	1	0	0	0	0	0	0	0;
    0	0	1	0	1	0	0	0	1	0	0	0	1	0;
    0	0	0	0	1	0	0	1	0	0	0	0	0	0;
    0	1	0	0	0	0	1	0	0	0	1	0	0	0;
    0	0	0	0	0	1	0	0	0	0	1	0	0	0;
    0	0	0	1	0	0	0	0	0	0	0	1	0	1;
    0	0	0	0	0	0	0	1	1	0	0	1	0	1;
    0	0	0	0	0	0	0	0	0	1	1	0	1	0;
    0	0	0	0	0	1	0	0	0	0	0	1	0	1;
    0	0	0	0	0	0	0	0	0	1	1	0	1	0;
    ];


GF = digraph(adj_mat);
src = 1;
dst = 4;
figure(1)
plot(GF)

for i=1:2
    P = shortestpath(GF,src,dst);
    for j=1:length(P)-1
          next_node = P(j+1);
          curr_node = P(j);
    
          x1 = find(GF.Edges.EndNodes(:,1) ==curr_node);
          x2 = find(GF.Edges.EndNodes(:,2) ==next_node);
          x3 = intersect(x1,x2);
          GF.Edges.Weight(x3) = GF.Edges.Weight(x3) - 1;
          if GF.Edges.Weight(x3) <0
            error('Edges.Weight  <0 !!!');
          elseif GF.Edges.Weight(x3) == 0
              GF = rmedge(GF,x3);
          else
              continue;
          end
                  
    end
end
figure(2)
plot(GF)
