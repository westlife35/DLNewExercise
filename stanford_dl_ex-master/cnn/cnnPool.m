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
        %:,filterNum,imageNum)));%ò��squeeze��ҪҲ����
        %�������ԣ�test1==test2==im,����im��Ϊ����
        im=convolvedFeatures(:, :,filterNum,imageNum);
        pooledImage =conv2(im, filter,'valid');  % ���⣬��������conv2���أ�����ֱ�Ӽ��������أ��о�ֱ�Ӽ���������Ϊfor���Ĵ��ڶ�Ӱ��Ч�ʰ�
        pooledImage = pooledImage(1:poolDim:end,1:poolDim:end);%ȡ�м䲿��  
        pooledImage = pooledImage./(poolDim*poolDim);  % get mean
  
        %pooledImage = sigmoid(pooledImage); %����Ҫsigmoid
        %�о����ǲ���Ҫ������sigmoid�Ѿ���ǰ������
        %pooledImage = reshape(pooledImage,convolvedDim / poolDim,convolvedDim / poolDim, 1, 1);%2ά��ά4ά  not necessary to squeeze
%         if isequal(pooledImage, reshape(pooledImage,convolvedDim / poolDim,convolvedDim / poolDim, 1, 1))
%             disp('same without squeeze');
%         end
        
        pooledFeatures(:, :, filterNum, imageNum) = pooledImage;  
    end  
end

end

