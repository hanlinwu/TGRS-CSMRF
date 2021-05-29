function A = calUnaryTerm_A(Mset, superpixels, par)

[img_num,map_num] = size(Mset);
bins = 128;
knn = par.knn;
thes = par.thes;

w = cell(img_num,par.numsupixel);

A = [];
for i = 1:img_num
    spnum = max(superpixels{i}(:)); % the actual number of the superpixels
    avg_p = zeros(spnum,2); % Ô¤·ÖÅäÄÚ´æ
    
    for l = 1:spnum
        [xx,yy] = find(superpixels{i} == l);
        avg_p(l,:) = mean([xx,yy]);
    end
    
    [nb,~] = knnsearch(avg_p,avg_p,'k',knn,'distance','euclidean');
    
    F = cell(map_num,spnum);
    parfor p = 1:map_num
        mest = Mset{i,p};
        mest = im2double(mest);
        avg = zeros(1,spnum);
        for j=1:spnum
            h = superpixels{i} == j;
            % calculate the average saliency score for each superpixel
            avg(j) = mean(mest(h));
        end
        % compute the color histogram for the p-th map of the i-th image
        for sp = 1:spnum
            window = nb(sp,:);
            sign = avg(window) >= thes * max(avg(window)); % sailency thresholding
            slabels = window(sign); % record the superpixel label over the theshold
            fhis = [];
            for q = 1:numel(slabels)
                h = (superpixels{i} == slabels(q));
                H = (mest(h));
                his = hist(H(:),(0:1:bins-1)/bins);
                fhis = [fhis;his];
            end
            F{p,sp} = sum(fhis);%feature sp on p-th map
        end
    end
    
    parfor sp = 1:spnum
        f_matrix = zeros(map_num,bins);
        for p = 1:map_num
            f_matrix(p,:) = F{p,sp};
        end
        f_matrix = f_matrix ./ 10000; 
        % to reduce the scale of each bin, for accelerating the speed.
        %---------------------RPCA---------------------------%
        % lamda is used to control the weight of the saprsity of E
        lamda = 0.05;
        [~ ,E] = exact_alm_rpca(f_matrix',lamda);
        S = double(E');
        w{i,sp} = sqrt(sum(abs(S).^2,2));
        w{i,sp} = w{i,sp} / (max(w{i,sp})-min(w{i,sp})+1e-10); %normalization
        w{i,sp} = exp(-w{i,sp})+1e-10;
        sum_w = sum(w{i,sp},1);
        w{i,sp} = w{i,sp} / (sum_w);
    end
    
    parfor sp = 1:max(superpixels{i}(:))
    A_vec1 = ones(1,map_num)-w{i,sp}';
    A_vec2 = exp(A_vec1)./sum(exp(A_vec1));
    A = vertcat(A,A_vec2);
    end
end
end