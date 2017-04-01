function [grad_s_Wi_approx, grad_s_Wo_approx, ...
    grad_s_bi_approx, grad_s_bo_approx] = ...
    gradient_nn_approx(X,Y,Wi,bi,Wo,bo)
% Numerically compute gradient of the logistic loss of the 
% neural network on example X with target label Y, 
% with respect to the parameters Wi,bi,Wo,bo.
%
% Input: X ... 2x1 vector of the input example
%        Y ... 1x1 the target label in {-1,1}   
%        Wi,bi,Wo,bo ... parameters of the network
%        Wi ... [hxd]
%        bi ... [hx1]
%        Wo ... [1xh]
%        bo ... 1x1
%        where h... is the number of hidden units
%              d... is the number of input dimensions (d=2)
%
% Output: 
%  grad_s_Wi_approx [hxd] ... numericl gradient of loss s(Y,Y(X)) w.r.t  Wi
%  grad_s_Wo_approx [1xh] ... numericl gradient of loss s(Y,Y(X)) w.r.t. Wo
%  grad_s_bi_approx [hx1] ... numericl gradient of loss s(Y,Y(X)) w.r.t. bi
%  grad_s_bo_approx [1x1] ... numericl gradient of loss s(Y,Y(X)) w.r.t. bo
%

% Get the neural network dimensions
[h,d] = size(Wi);

% Compute the loss function at the point of interest
eps = 1e-6;
[~,~,loss] = nnet_forward_logloss(X,Y,Wi,bi,Wo,bo);

% Compute the numerical gradient of loss s(Y,Y(X)) w.r.t. bo
[~,~,loss_bo] = nnet_forward_logloss(X,Y,Wi,bi,Wo,bo+eps);
grad_s_bo_approx = (loss_bo-loss)/eps;

% Compute the numerical gradient of loss s(Y,Y(X)) w.r.t. bi
grad_s_bi_approx = zeros(h,1);
for i = 1:h
    bi_i = eps*[zeros(i-1,1);1;zeros(h-i,1)];
    [~,~,loss_bi_k] = nnet_forward_logloss(X,Y,Wi,bi+bi_i,Wo,bo);
    grad_s_bi_approx(i) = (loss_bi_k-loss)/eps;
end

% Compute the numerical gradient of loss s(Y,Y(X)) w.r.t. Wo
grad_s_Wo_approx = zeros(1,h);
for i = 1:h
    Wo_i = eps*[zeros(1,i-1) 1 zeros(1,h-i)];
    [~,~,loss_Wo_k] = nnet_forward_logloss(X,Y,Wi,bi,Wo+Wo_i,bo);
    grad_s_Wo_approx(i) = (loss_Wo_k-loss)/eps;
end

% Compute the numerical gradient of loss s(Y,Y(X)) w.r.t. Wi
grad_s_Wi_approx = zeros(h,d);
for i = 1:h
    for j = 1:d
        Wi_ij = zeros(h,d);
        Wi_ij(i,j) = eps;
        [~,~,loss_Wi_ij] = nnet_forward_logloss(X,Y,Wi+Wi_ij,bi,Wo,bo);
        grad_s_Wi_approx(i,j) = (loss_Wi_ij-loss)/eps;
    end
end
end

