function [score, model] = EMR(data,y0,opts)
% [score, model] = EMR(data,y0,opts): Efficient Manifold Ranking
% Input:
%       - data: the data matrix of size nSmp x nFea, where each row is a sample
%               point
%       - y0:   the initial query vector, e.g., query item =1 and the other all 0; 
%       
%       opts: options for this algorithm
%           - landmarks : given the landmarks
%           - p: the number of landmarks picked (default 1000)
%           - r: the number of nearest landmarks for representation (default 5)
%           - a: weight in manifold ranking, score = (I - aS)^(-1)y, default  0.99
%           - mode: landmark selection method, currently support
%               - 'kmeans': use centers of clusters generated by kmeans (default)
%               - 'random': use randomly sampled points from the original
%                           data set 
%           The following parameters are effective ONLY in mode 'kmeans'
%           - kmNumRep: the number of replicates for initial kmeans (default 1)
%           - kmMaxIter: the maximum number of iterations for initial kmeans (default 5)
%
% Output:
%       - score: the ranking scores for each point
%       - model: the learned model for out-of-sample retrieval
%
% Usage:
%
%      See: http://www.zjucadcg.cn/dengcai/Data/ReproduceExp.html#EMR
%
%Reference:
%
%	 Bin Xu, Jiajun Bu, Chun Chen, Deng Cai, Xiaofei He, Wei Liu, Jiebo
%	 Luo, "Efficient Manifold Ranking for Image Retrieval",in Proceeding of
%	 the 34th International ACM SIGIR Conference on Research and
%	 Development in Information Retrieval (SIGIR), 2011, pp. 525-534.  
%
%   version 2.0 --Feb./2012 
%   version 1.0 --Sep./2010 
%
%   Written by Bin Xu (binxu986 AT gmail.com)
%              Deng Cai (dengcai AT gmail.com)


% Set and parse parameters
if (~exist('opts','var'))
   opts = [];
end

p = 1000;
if isfield(opts,'p')
    p = opts.p;
end

r = 5;
if isfield(opts,'r')
   r = opts.r;
end

a = 0.99;
if isfield(opts,'a')
   a = opts.a;
end

mode = 'kmeans';
if isfield(opts,'mode')
    mode = opts.mode;
end

if isfield(opts,'landmarks')
   landmarks = opts.landmarks; 
end

nSmp =size(data,1);

% Landmark selection
if ~exist('landmarks','var')
    if strcmp(mode,'kmeans')
        kmMaxIter = 5;
        if isfield(opts,'kmMaxIter')
            kmMaxIter = opts.kmMaxIter;
        end
        kmNumRep = 1;
        if isfield(opts,'kmNumRep')
            kmNumRep = opts.kmNumRep;
        end
        [~,landmarks]=kmeans(data,p,'MaxIter',kmMaxIter,'Replicates',kmNumRep);
        clear kmMaxIter kmNumRep
    elseif strcmp(mode,'random')
        indSmp = randperm(nSmp);
        landmarks = data(indSmp(1:p),:);
        clear indSmp
    else
        error('mode does not support!');
    end
end
model.landmarks = landmarks;
model.a = a;
model.r = r;

% Z construction
D = EuDist2(data,landmarks);
dump = zeros(nSmp,r);
idx = dump;
for i = 1:r
    [dump(:,i),idx(:,i)] = min(D,[],2);
    temp = (idx(:,i)-1)*nSmp+[1:nSmp]';
    D(temp) = 1e100;
end
dump = bsxfun(@rdivide,dump,dump(:,r));
dump = 0.75 * (1 - dump.^2);
Gsdx = dump;
Gidx = repmat([1:nSmp]',1,r);
Gjdx = idx;
Z=sparse(Gidx(:),Gjdx(:),Gsdx(:),nSmp,p);

model.Z = Z';

% Efficient Ranking
feaSum = full(sum(Z,1));
D = Z*feaSum';
D = max(D, 1e-12);
D = D.^(-.5);
H = spdiags(D,0,nSmp,nSmp)*Z;

C = speye(p);
A = H'*H-(1/a)*C;

tmp = H'*y0;
tmp = A\tmp;
score = y0 - H*tmp;