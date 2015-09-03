function [ stack ] = initialize_weights( ei )
%INITIALIZE_WEIGHTS Random weight structures for a network architecture
%   eI describes a network via the fields layerSizes, inputDim, and outputDim 
%   
%   This uses Xavier's weight initialization tricks for better backprop
%   See: X. Glorot, Y. Bengio. Understanding the difficulty of training 
%        deep feedforward neural networks. AISTATS 2010.

%% initialize hidden layers
stack = cell(1, numel(ei.layer_sizes));
for l = 1 :numel(ei.layer_sizes)
    if l > 1
        prev_size = ei.layer_sizes(l-1);
    else
        prev_size = ei.input_dim;
    end;
    cur_size = ei.layer_sizes(l);
    % Xaxier's scaling factor
    %s = sqrt(6) / sqrt(prev_size + cur_size);
    s = sqrt(6) / sqrt(prev_size + cur_size+1);
    % good initialization test accuracy: 0.968600 train accuracy: 1.000000
    % Elapsed time is 107.370475 seconds.   test accuracy: 0.972700 train
    % accuracy: 1.000000 Elapsed time is 110.425341 seconds.
    stack{l}.W = rand(cur_size, prev_size)*2*s - s;
    % bad initialization test accuracy: 0.113500 train accuracy: 0.112367 Elapsed time is 17.998085 seconds.
    %stack{l}.W = rand(cur_size, prev_size);
    % good initialization test accuracy: 0.970800 train accuracy: 1.000000 Elapsed time is 155.449660 seconds.
    %stack{l}.W = randn(cur_size, prev_size);
    stack{l}.b = zeros(cur_size, 1);
end
