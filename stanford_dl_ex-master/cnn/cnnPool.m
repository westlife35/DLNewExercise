function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

%%% YOUR CODE HERE %%%
filter = ones(poolDim);  
for imageNum=1:numImages  
    for filterNum=1:numFilters  
        %test1=squeeze(convolvedFeatures(:, :,filterNum,imageNum));
        %test2=convolvedFeatures(:, :,filterNum,imageNum);
        %im = squeeze(squeeze(convolvedFeatures(:,
        %:,filterNum,imageNum)));%貌似squeeze不要也可以
        %经过测试，test1==test2==im,所以im简化为如下
        im=convolvedFeatures(:, :,filterNum,imageNum);
        pooledImage =conv2(im, filter,'valid');  % 问题，这里是做conv2块呢？还是直接加起来快呢？感觉直接加起来会因为for语句的存在而影响效率吧
        pooledImage = pooledImage(1:poolDim:end,1:poolDim:end);%取中间部分  
        pooledImage = pooledImage./(poolDim*poolDim);  % get mean
  
        %pooledImage = sigmoid(pooledImage); %不需要sigmoid
        %感觉不是不需要，而是sigmoid已经在前边做了
        %pooledImage = reshape(pooledImage,convolvedDim / poolDim,convolvedDim / poolDim, 1, 1);%2维变维4维  not necessary to squeeze
%         if isequal(pooledImage, reshape(pooledImage,convolvedDim / poolDim,convolvedDim / poolDim, 1, 1))
%             disp('same without squeeze');
%         end
        
        pooledFeatures(:, :, filterNum, imageNum) = pooledImage;  
    end  
end

end

