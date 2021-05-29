function SalMap = fusion(x,supixels,map,par)

img_num = size(supixels,2);
% calculate weight matrix
addnum = 0;
weight = cell(img_num,par.numsupixel);
for i = 1:img_num
    spnum = max(supixels{i}(:));
    for j = 1:spnum
        weight{i,j} = x(j+addnum,:);
    end
    addnum = addnum + spnum;
end

% begin fusion
SalMap = cell(1,img_num);
for j = 1:img_num
    temp_map = zeros(size(map.Mset{j,1}));
    for q=1:max(supixels{j}(:))
        h = supixels{j} == q;
        for t = 1:map.num
            temp_map = temp_map + (weight{j,q}(t)*h).*double(map.Mset{j,t});
        end
    end
    rs = normalize(temp_map);
    SalMap{j} = rs;
end
end