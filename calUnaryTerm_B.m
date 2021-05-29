function B = calUnaryTerm_B(s_avg, img_num, superpixels, method_num)
B = [];
for id = 1:img_num
    spnum = max(superpixels{id}(:));
    temp = [s_avg{id,:}];
    f_matrix = reshape(temp,[method_num,floor(size(temp,2)/method_num)]);
    %f_matrix = f_matrix ./ 10000;
    % to reduce the scale of each bin, for accelerating the speed.
    %---------------------RPCA---------------------------%
    % lamda is used to control the weight of the saprsity of E
    lamda = 0.05;
    [~ ,E] = exact_alm_rpca(f_matrix',lamda);
    S = double(E');
    w{id} = sqrt(sum(abs(S).^2,2));
    w{id} = w{id} / (max(w{id})-min(w{id})+eps); %normalization
    w{id} = exp(-w{id})+eps;
    sum_w = sum(w{id},1);
    w{id} = w{id} / (sum_w);
    temp = w{id};
    B = [B;repmat(temp',[spnum,1])];
end
