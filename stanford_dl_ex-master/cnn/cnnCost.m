function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,numFilters,numImages);

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

%%% YOUR CODE HERE %%%
activations = cnnConvolve(filterDim, numFilters, images, Wc, bc);  
activationsPooled = cnnPool(poolDim, activations); 

% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);

%%% YOUR CODE HERE %%%
z = Wd*activationsPooled;  
z = bsxfun(@plus,z,bd);  
%z = Wd * activationsPooled+repmat(bd,[1,numImages]);   
z = bsxfun(@minus,z,max(z,[],1));%��ȥ���ֵ������һ��ά��  �������Ⱑ��Ψ��û���١�����Ч������ˣ�
z = exp(z);  
probs = bsxfun(@rdivide,z,sum(z,1));  
preds = probs;

%�Ѻ�ߵĴ�����ǰ����������
% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;%������أ��о�����Ҫ�ɣ�������
    cost=0;
    return;
end;

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

%%% YOUR CODE HERE %%%
logProbs = log(probs);     
% labelIndex=sub2ind(size(logProbs), labels', 1:size(logProbs,2));%��Ϊ�ҳ���������ʹ�õ����⣬���Ըĳ��±ߵĴ���Ҳ�ǿ��Ե�
labelIndex=sub2ind(size(logProbs), labels, [1:size(logProbs,2)]');
%�ҳ�����logProbs����������������labelsָ��������1:size(logProbs,2)ָ�������������������ظ�labelIndex  
values = logProbs(labelIndex);    
cost = -sum(values);  

weightDecay=0.001;%��ʱ����Ϊ0 ��supervised-dnn�о���0
weightDecayCost = (weightDecay/2) * (sum(Wd(:) .^ 2) + sum(Wc(:) .^ 2));  
cost = cost / numImages+weightDecayCost;   
%Make sure to scale your gradients by the inverse size of the training set   
%if you included this scale in the cost calculation otherwise your code will not pass the numerical gradient check.  

%������Щ�������Ƿŵ�ǰ��һ��Ч�ʱȽϸ�
% % Makes predictions given probs and returns without backproagating errors.
% if pred
%     [~,preds] = max(probs,[],1);
%     preds = preds';
%     grad = 0;
%     return;
% end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% YOUR CODE HERE %%%
%softmax�в�  
targetMatrix = zeros(size(probs));    
targetMatrix(labelIndex) = 1;    
softmaxError = probs-targetMatrix;  
  
%pool��в�  
poolError = Wd'*softmaxError;  
poolError = reshape(poolError, outputDim, outputDim, numFilters, numImages);  
  
unpoolError = zeros(convDim, convDim, numFilters, numImages);  
unpoolingFilter = ones(poolDim);  
poolArea = poolDim*poolDim;  
%չ��poolErrorΪunpoolError  
for imageNum = 1:numImages  
    for filterNum = 1:numFilters  
        e = poolError(:, :, filterNum, imageNum);  
        unpoolError(:, :, filterNum, imageNum) = kron(e, unpoolingFilter)./poolArea;  %����Ϊʲô��Ҫ����poolArea
    end  
end  
  
convError = unpoolError .* activations .* (1 - activations);

%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%
%softmax�ݶ�  
Wd_grad = (1/numImages).*softmaxError * activationsPooled'+weightDecay * Wd; % l+1��в� * l�㼤��ֵ  
bd_grad = (1/numImages).*sum(softmaxError, 2);  
  
% Gradient of the convolutional layer  
bc_grad = zeros(size(bc));  
Wc_grad = zeros(size(Wc));  
  
%����bc_grad  
for filterNum = 1 : numFilters  
    e = convError(:, :, filterNum, :);  
    bc_grad(filterNum) = (1/numImages).*sum(e(:));  
end  
  
%��תconvError  
for filterNum = 1 : numFilters  
    for imageNum = 1 : numImages  
        e = convError(:, :, filterNum, imageNum);  
        convError(:, :, filterNum, imageNum) = rot90(e, 2);  
    end  
end  
  
for filterNum = 1 : numFilters  
    Wc_gradFilter = zeros(size(Wc_grad, 1), size(Wc_grad, 2));  
    for imageNum = 1 : numImages       
        Wc_gradFilter = Wc_gradFilter + conv2(images(:, :, imageNum), convError(:, :, filterNum, imageNum), 'valid');  % why ?? ֻ�ǽ̳�����ôд��ʽ��
    end  
    Wc_grad(:, :, filterNum) = (1/numImages).*Wc_gradFilter;  
end  
Wc_grad = Wc_grad + weightDecay * Wc; 

%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
