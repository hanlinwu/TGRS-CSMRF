function [color,gabor] = getFeature_new(IRGB, lambda)
%GENFEATURE 获取图像的颜色和纹理特征、
imgNum = numel(IRGB);
color = [];
gabor = [];
for i = 1:imgNum
    RGB = im2double(IRGB{i});
    rgb = reshape(RGB,size(RGB,1)*size(RGB,2),size(RGB,3));
    color = [color;rgb];
    gabor = [gabor;create_feature_space(RGB, lambda)];
end
end

function feaImg=create_feature_space(im, lambda)
[n, r, ~] = size(im);
im_gray = rgb2gray(im);

% choice of bandwidth and orientation
% f = [0,2];
% f is 1/lambda where lambda is the wavelength (pixels). Valid numbers are
% between 2 and 256
% lambda = [3,5,7];
% lambda = sqrt(2)*[2 6 8];  % old version

f = 1./lambda;
% theta = [0 pi/6 pi/3 3*pi/4];
theta = 0:pi/8:(pi-pi/8);
%theta = [3*pi/4];
gamma = [0, -0.5*pi];

feaImg=[];
for i=1:length(f)
    for j=1:length(theta)
        for g = 1:length(gamma)
            [~, Gimg]=gaborfilterVikas(im_gray(:,:),f(i),theta(j),gamma(g));
        end
        Gimg=reshape(Gimg,n*r,1);
        feaImg=cat(2,feaImg,Gimg);
    end
end
end