function [coMset,Repeatness_avg,coMap_1,coMap_1_seg,coMap_2,coMap_2_seg] = Sal2Cosal(cfeat,superpixels,Mset,par)
% convert single image saliency map to co-saliency map

imgsize = par.imgsize;
methodNum = size(Mset,2);
imgNum = size(Mset,1);

opts.kmMaxIter = 100;opts.a = 0.8;opts.p = 100;opts.kmNumRep = 5;
des = cfeat;
randid = floor(size(des,1)*rand(1,opts.p))'+1;
randcen = des(randid,:);
[landmarks,~] = do_kmeans(des',opts.p,opts.kmNumRep, randcen'); % ·Ç³£Âý...
opts.landmarks = landmarks';
coMset = cell(size(Mset));

coMap_1 = cell(methodNum,imgNum);
coMap_1_seg = cell(methodNum,imgNum);

coMap_2 = cell(methodNum,imgNum,imgNum);
coMap_2_seg = coMap_2;
Repeatness_avg = cell(methodNum,imgNum);
for methodIdx = 1:methodNum
    % step 1
    query = [];
    for imgIdx = 1:imgNum
        temp = Mset{imgIdx,methodIdx};
        %thresh = graythresh(temp);
        thresh = mean(temp,'all');
        temp = imbinarize(temp,thresh); 
        temp = reshape(temp,prod(imgsize),1);
        query = [query;temp];
    end
    rslt = EMR(cfeat,query,opts);
    rslt = minmaxregular(rslt);
    for i = 1:imgNum
        idx = (i-1)*prod(par.imgsize);
        coMap_1{methodIdx,i} = reshape(rslt(idx+1:idx+prod(imgsize),1),imgsize);
        coMap_1{methodIdx,i} = minmaxregular(coMap_1{methodIdx,i});
        thresh = mean(coMap_1{methodIdx,i},'all');
        coMap_1_seg{methodIdx,i} = imbinarize(coMap_1{methodIdx,i},thresh);
    end
    
    % step 2
    for imgIdx = 1:imgNum
        %thresh = graythresh(coMap_1{imgIdx});
        query = reshape(coMap_1_seg{methodIdx,imgIdx},prod(imgsize),1);
        Gidx = find(query==1) + prod(imgsize)*(imgIdx-1);
        Gjdx = ones(size(Gidx,1),1);
        query = sparse(Gidx,Gjdx,Gjdx,size(cfeat,1),1);
        rslt = EMR(cfeat,query,opts);
        rslt = full(rslt);
        rslt = minmaxregular(rslt);
        for i = 1:imgNum
            idx = (i-1)*prod(imgsize);
            coMap_2{methodIdx,imgIdx,i} = reshape(rslt(idx+1:idx+prod(imgsize),1),imgsize);
            coMap_2{methodIdx,imgIdx,i} = minmaxregular(coMap_2{methodIdx,imgIdx,i});
            coMap_2_seg{methodIdx,imgIdx,i} = imbinarize(coMap_2{methodIdx,imgIdx,i},'global');
        end
    end
    
    % fusion
    for i = 1:imgNum
       temp = zeros(imgsize);
       for j = 1:imgNum
           temp = temp + coMap_2_seg{methodIdx,j,i};
       end
       Repeatness = temp >=2;
       % avg
       temp = zeros(imgsize);
       for spIdx = 1:max(superpixels{i}(:))
           idx = superpixels{i}==spIdx;
           temp(idx) = mean(Repeatness(idx),'all');
       end
       temp(temp >=0.8) = 1;
       Repeatness_avg{methodIdx,i} = temp;
       coMset{i,methodIdx} = Mset{i,methodIdx}.*temp;
    end
end
end