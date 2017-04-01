% Demo script for training two layer fully connected neural network 
% with one hidden layer
% 
% Josef.Sivic@ens.fr
% Adapted from Nicolas Le Roux

clear; % clear all current variables.

% load training data
tmp = load('double_moon_train1000.mat');
Xtr = tmp.X';
Ytr = tmp.Y';

% load the validation data
tmp = load('double_moon_val1000.mat');
Xval = tmp.X';
Yval = tmp.Y';

% train fully connected neural network with 1 hidden layer
h  = 100; % number of hidden units, i.e. the dimensionality of the hidden layer
di = 2; % input dimension (2D) -- do not change
do = 1; % output dimension (1D - classification) -- do not change

lrate     = 0.02; % learning rate
nsamples  = length(Ytr);
visualization_step = 100000; % visualize output only these steps

% randomly initialize parameters of the model
Wi = rand(h,di);
bi = rand(h,1);
Wo = rand(1,h);
bo = rand(1,1);

for i = 1:100*nsamples % hundred passes through the data
    % draw an example at random
    n = randi(nsamples);
    
    X = Xtr(:,n);
    Y = Ytr(:,n); % desired output
    
    % compute gradient 
    [grad_s_Wi, grad_s_Wo, grad_s_bi, grad_s_bo] = ...
        gradient_nn(X,Y,Wi,bi,Wo,bo);

%     % compute numerical gradient 
%     [grad_s_Wi_approx, grad_s_Wo_approx, ...
%         grad_s_bi_approx, grad_s_bo_approx] = ...
%         gradient_nn_approx(X,Y,Wi,bi,Wo,bo);

    % gradient update                                 
    Wi = Wi - lrate.*grad_s_Wi;
    Wo = Wo - lrate.*grad_s_Wo;
    bi = bi - lrate.*grad_s_bi;
    bo = bo - lrate.*grad_s_bo;
    
    % plot training error
    [Po,Yo,loss]    = nnet_forward_logloss(Xtr,Ytr,Wi,bi,Wo,bo);
    Yo_class        = sign(Yo);
    tr_error(i)     = sum((Yo_class - Ytr)~=0)./length(Ytr);
    
    % plot validation error
    [Pov,Yov,lossv]    = nnet_forward_logloss(Xval,Yval,Wi,bi,Wo,bo);
    Yov_class          = sign(Yov);
    val_error(i)       = sum((Yov_class - Yval)~=0)./length(Yval);

    % visualization only every visualiztion_step-th iteration       
    if ~mod(i,visualization_step) 

        % show decision boundary
        plot_decision_boundary(Xtr,Ytr,Wi,bi,Wo,bo);
        
        % plot the evolution of the training and test errors
        if 1            
            figure(3); hold off;
            plot(tr_error);
            title(sprintf('Training error: %.2f %%',tr_error(i)*100));
            grid on;
            
            figure(4); hold off;
            plot(val_error);
            title(sprintf('Validation error: %.2f %%',val_error(i)*100));
            grid on;
        end;
        
        
        % Numerically check the correctness of gradients
        % To be completed 
        %
        
    end;
    
end;

tr_error(end),val_error(end),max(find(val_error==0,1),find(tr_error==0,1))

