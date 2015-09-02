function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
for l=1:numHidden   %隐藏层特征计算  
    if(l == 1)  
        z = stack{l}.W*data;  
    else   
        z = stack{l}.W*hAct{l-1};  
    end  
    z = bsxfun(@plus,z,stack{l}.b);  
    hAct{l}=sigmoid(z);  
end  
  
%输出层(softmax)特征计算  
h = (stack{numHidden+1}.W)*hAct{numHidden};  
h = bsxfun(@plus,h,stack{numHidden+1}.b);  
e = exp(h);  
pred_prob = bsxfun(@rdivide,e,sum(e,1)); %概率表  
hAct{numHidden+1} = pred_prob;  
%[~,pred_labels] = max(pred_prob, [], 1);

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
ceCost =0;  
c= log(pred_prob);  
%fprintf("%d,%d\n",size(labels,1),size(labels,2)); %60000,1  
I=sub2ind(size(c), labels', 1:size(c,2));%找出矩阵c的线性索引，行由labels指定，列由1:size(c,2)指定，生成线性索引返回给I  
values = c(I);  
ceCost = -sum(values);

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
% Cross entroy gradient  
  
%d = full(sparse(labels,1:size(c,2),1));      
d = zeros(size(pred_prob));  
d(I) = 1;  
error = (pred_prob-d); %输出层的残差   
  
%梯度，残差反向传播  
for l = numHidden+1: -1 : 1   
    gradStack{l}.b = sum(error,2);  
    if(l == 1)  
        gradStack{l}.W = error*data';  
        break;%l==1时，即当前层是第一层隐藏层时，不需要再传播残差  
    else   
        gradStack{l}.W = error*hAct{l-1}';  
    end  
    error = (stack{l}.W)'*error .*hAct{l-1}.* (1-hAct{l-1});%后面部分是激活函数偏导数  
    %error = (stack{l}.W)'*error .* (1-hAct{l-1}.*hAct{l-1});%后面部分是激活函数偏导数  
end 

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost = 0;  
for l = 1:numHidden+1  
    wCost = wCost + .5 * ei.lambda * sum(stack{l}.W(:) .^ 2);%所有权值的平方和  
end  
  
cost = ceCost + wCost;  
  
% Computing the gradient of the weight decay.  
for l = numHidden : -1 : 1  
    gradStack{l}.W = gradStack{l}.W + ei.lambda * stack{l}.W;%softmax没用到权重衰减项  
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



