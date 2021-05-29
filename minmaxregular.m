function [M] = minmaxregular(M)
M = (M-min(M(:)))./(max(M(:))-min(M(:)));
end