%  author: Hanlin Wu, Beijing Normal University, China
%  email: hanlinwu@mail.bnu.edu.cn

%  Implementation of Paper: L. Zhang, H. Wu,
%  "Cosaliency Detection and Region-of-Interest Extraction via Manifold Ranking and MRF in Remote Sensing Images", 
%  in IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING,2021, DOI:10.1109/TGRS.2021.3079441. 

%  If you use this code for academic purposes, please cite the paper above.

clc;clearvars;
%% parameter setting
par.mapnames = {'HS', 'MC', 'MR', 'DSR', 'GR'};

root_path = 'demo';
par.rsltpath = fullfile(root_path, 'results');
par.dataPath = root_path;

par.img_ext = '.bmp';
par.map_ext = '.bmp';
par.energy_lambda = [8,8,1]; % lambda for energy function, lambda_1, lambda_1, lambda_2
par.numsupixel = 50; % superpixel size for regional fusion
par.knn = 10; % coefficient for k-nearest neightbor (1st data)
par.thes = 0.1; 
par.neighbors = 1;
par.clambda = 1;
par.glambda = 1.5; 
par.ccodebook = 50; % how many bins in color histogram 
par.cclusternum = 10; % how many run times in kmeans
par.imgsize = [128,128]; % the size of the input image and the output image
par.output_imgsize = [128, 128];
par.datasetname = 'data';

%% Main
dataSet.name = par.datasetname;

addpath(genpath('utils'));
rmpath(genpath(fullfile('utils','cvx-w64\lib\narginchk_')));

dataSet.path = fullfile(par.dataPath, dataSet.name);
map.names = par.mapnames;
map.num = length(map.names);

MaskList_all = [];
salMapList_all = [];
timeUsed_all = [];
rslt.path = fullfile(par.rsltpath,dataSet.name);

rng(1);
subDataSet.path = dataSet.path;

disp("-----------------------------------------------------------------");
disp('[I] Begin processing image group...');
%create rslt path
rsltpath = par.rsltpath;
if ~exist(rsltpath, 'dir')
   mkdir(rsltpath); 
end

img.path = fullfile(subDataSet.path,'images',['*',par.img_ext]);
img.list = dir(img.path);
img.num = length(img.list);

% read images
IRGB = cell(1,img.num);
imgNames = cell(1,img.num);
for i = 1:img.num
    imgNames{i} = char(regexpi(img.list(i).name,'^\w+','match'));
    img_raw = imread(fullfile(img.list(i).folder,img.list(i).name));
    IRGB{i} = imresize(img_raw,par.imgsize);
end
clear img_raw;

% read-in saliency maps
map.path = fullfile(subDataSet.path,'maps');
map.Mset = read_maps(imgNames,map,par);

% clockTimeStart
clockTimeStart = clock;

% generating features
disp('[I] begin generating features...');
lambda = [2, 4, 6];
[feature.color, feature.gabor] = getFeature(IRGB, lambda);

%superpixes exact
superpixels = cell(1,img.num);
for i = 1:img.num
    superpixels{i} = SLIC_mex(IRGB{i}, par.numsupixel, 20);
end

% K-means clustering for color features
des = feature.color;
randid = floor(size(des,1)*rand(1,par.ccodebook))'+1;
randcen = des(randid,:);
[~,ccenl] = do_kmeans(des',par.ccodebook,par.cclusternum, randcen');
ccenl = ccenl+1;

% K-means clustering for gabor features
des = feature.gabor;
randid = floor(size(des,1)*rand(1,par.ccodebook))'+1;
randcen = des(randid,:);
[~,gcenl] = do_kmeans(des',par.ccodebook,par.cclusternum, randcen');
gcenl = gcenl+1;

% generate the feature histogram for superpixels
pnum = 0;
cfeat = cell(img.num,max(superpixels{i}(:)));
gfeat = cell(img.num,max(superpixels{i}(:)));
for i = 1:img.num
    for j = 1:max(superpixels{i}(:))
        idx = find(superpixels{i}(:) == j);
        cfeat{i,j} = hist(ccenl(idx+pnum),(1:par.ccodebook))/numel(idx);
        gfeat{i,j} = hist(gcenl(idx+pnum),(1:par.ccodebook))/numel(idx);
    end
    pnum = pnum + numel(superpixels{i});
end

% Convert saliency maps to co-saliency maps
disp('[I] Convert saliency maps to co-saliency maps...');
[map.Mset,~,~,~,~,~] = Sal2Cosal([feature.color feature.gabor],superpixels,map.Mset,par);

disp('[I] begin MRF...');
[par.sigma_c,par.sigma_g] = compute_sigma(img.num,superpixels,cfeat,gfeat);% Computer the normalization constant
[affinity,~] = calAffinity(superpixels,img.num,cfeat,gfeat,par);% compute the affinity matrix

% calculateing the first unary term...
A = calUnaryTerm_A(map.Mset, superpixels, par);

% calculating laplacian matrix for pairwise term...
clear D;
% compute the degree matrix
D = ones(size(affinity,1),size(affinity,1));
D(eye(size(affinity,1), 'logical')) = sum(affinity, 2);
L = D - affinity;
[~,E,W] = eig(L);
m = W*E^(1/2);

% second unary term
s_avg = avg_sm(superpixels, img.num, map.num, map.Mset);
B = calUnaryTerm_B(s_avg,img.num,superpixels,map.num);

% optimization
disp('[I] begin optimization...');
[~, x] = optimation(A,B,m,par.energy_lambda);

% fusion
SalMap = fusion(x, superpixels, map, par);

% post deal
for i = 1:img.num
    SalMap{i} = normalize(posdeal2(SalMap{i},0.4,6));
    SalMap{i} = imresize(SalMap{i}, par.output_imgsize);
end

% save rslt
for j = 1:img.num
    name = fullfile(rsltpath,[imgNames{j},'.png']);
    imwrite(SalMap{j},name);
end

rmpath(genpath('utils'));