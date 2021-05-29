function Mset = read_maps(imgNames,map,par)
imgNum = length(imgNames);
Mset = cell(imgNum, map.num);

for m=1:imgNum
    for n=1:map.num
        % read-in saliency maps
        mapPath = fullfile(map.path,map.names{n},strcat(imgNames{m},par.map_ext));
        map_raw = imread(mapPath);
        Mset{m,n} = im2double(imresize(map_raw,par.imgsize));
    end
end 