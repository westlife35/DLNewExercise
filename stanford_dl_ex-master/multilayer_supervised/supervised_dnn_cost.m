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
for l=1:numHidden   %���ز���������  
    if(l == 1)  
        z = stack{l}.W*data;  
    else   
        z = stack{l}.W*hAct{l-1};  
    end  
    z = bsxfun(@plus,z,stack{l}.b);  
    hAct{l}=sigmoid(z);  
end  
  
%�����(softmax)��������  
h = (stack{numHidden+1}.W)*hAct{numHidden};  
h = bsxfun(@plus,h,stack{numHidden+1}.b);  
e = exp(h);  
pred_prob = bsxfun(@rdivide,e,sum(e,1)); %���ʱ�  
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
I=sub2ind(size(c), labels', 1:size(c,2));%�ҳ�����c����������������labelsָ��������1:size(c,2)ָ�������������������ظ�I  
values = c(I);  
ceCost = -sum(values);

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
% Cross entroy gradient  
  
%d = full(sparse(labels,1:size(c,2),1));      
d = zeros(size(pred_prob));  
d(I) = 1;  
error = (pred_prob-d); %�����Ĳв�   
  
%�ݶȣ��в�򴫲�  
for l = numHidden+1: -1 : 1   
    gradStack{l}.b = sum(error,2);  
    if(l == 1)  
        gradStack{l}.W = error*data';  
        break;%l==1ʱ������ǰ���ǵ�һ�����ز�ʱ������Ҫ�ٴ����в�  
    else   
        gradStack{l}.W = error*hAct{l-1}';  
    end  
    error = (stack{l}.W)'*error .*hAct{l-1}.* (1-hAct{l-1});%���沿���Ǽ����ƫ����  
    %error = (stack{l}.W)'*error .* (1-hAct{l-1}.*hAct{l-1});%���沿���Ǽ����ƫ����  
end 

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost = 0;  
for l = 1:numHidden+1  
    wCost = wCost + .5 * ei.lambda * sum(stack{l}.W(:) .^ 2);%����Ȩֵ��ƽ����  
end  
  
cost = ceCost + wCost;  
  
% Computing the gradient of the weight decay.  
for l = numHidden : -1 : 1  
    gradStack{l}.W = gradStack{l}.W + ei.lambda * stack{l}.W;%softmaxû�õ�Ȩ��˥����  
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



