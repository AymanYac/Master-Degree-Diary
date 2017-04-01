%% 1 - Build the feature database
clear; % clear all current variables.
setup;
fprintf('feature database creation\n')

% load the pre-trained CNN model
net = load('imagenet-vgg-f.mat');

% get the list of images
files    = dir('data/images/*.jpg');

% initialize the variables to store the layer state for each image
prob     = zeros(size(files,1),size(net.meta.classes.name,2));
fc8      = zeros(size(files,1),net.layers{20}.size(3));
fc7      = zeros(size(files,1),net.layers{18}.size(3)); 
im_names = zeros(size(files,1),1);

% loop through all images
txt_disp = '';
i     = 0;
for file = files'
    % display the progress
    i = i + 1;
    if mod(i,25) == 0
        fprintf(repmat('\b',1,size(txt_disp,2)))
        txt_disp = sprintf('image number %d ouf of %d\n',i,size(files,1));
        fprintf(txt_disp)
    end
    
    % load the image and normalize it
    im  = imread(strcat('data/images/',file.name));
    im_ = single(im);
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
    im_ = im_ - net.meta.normalization.averageImage;

    % compute the forward pass on the image
    res = vl_simplenn(net, im_);

    % store the last layer prob, fc7 and fc8
    prob(i,:)   = res(21).x(:);
    fc8(i,:)    = res(20).x(:);
    fc7(i,:)    = res(18).x(:);
    im_names(i) = str2double(file.name(1:6));
end
fprintf(repmat('\b',1,size(txt_disp,2)))
fprintf('%d images used\nfeature database finished\n',i)


%% 2 - Evaluate the image classification using CNN features and linear SVM

% variables definition
categories  = {'motorbike','aeroplane','person'};
features    = {prob,fc7,fc8};
feat_name   = {'prob','fc7','fc8'};
norms       = {'none','L1','L2'};
C           = 10.^((-2:6)/2);
AP_final    = zeros(numel(categories),numel(features),...
    numel(norms),numel(C));

% initialize iterators
k_cat       = 0;
param_best  = zeros(numel(categories),4);

% loop through all categories
for cat = categories
    % update iterators
    k_cat   = k_cat+1;
    k_feat  = 0;
    
    % load the training and validation labels
    pos_train   = load(sprintf('data/%s_train.txt',string(cat)));
    pos_val     = load(sprintf('data/%s_val.txt',string(cat)));
    neg_train   = load('data/background_train.txt');
    neg_val     = load('data/background_val.txt');
    names_train  = [pos_train; neg_train];

    % loop through all features
    for feature = features
        % update iterators
        k_feat  = k_feat+1;
        k_norm  = 0;
        
        % loop through all norms
        for norm = norms
            % update iterators
            k_norm  = k_norm+1;
            k_C     = 0;
            
            % normalize the features
            feat_norm = feature{1};
            if string(norm) == 'L1'
                feat_norm = ...
                    bsxfun(@times,feat_norm,1./sum(abs(feat_norm),2));
            elseif string(norm) == 'L2'
                feat_norm = ...
                    bsxfun(@times,feat_norm,1./sqrt(sum(feat_norm.^2,2)));
            end
            
            % training dataset
            Y_train  = ...
                [ones(numel(pos_train),1);-ones(numel(neg_train),1)];
            X_train  = ...
                zeros(numel(pos_train)+numel(neg_train),size(feat_norm,2));
            for i = 1:numel(Y_train)
                i_im = find(im_names==names_train(i),1);
                X_train(i,:) = feat_norm(i_im,:);
            end
            
            % validation dataset
            names_val   = [pos_val; neg_val];
            Y_val       = ...
                [ones(numel(pos_val),1);-ones(numel(neg_val),1)];
            X_val       = ...
                zeros(numel(pos_val)+numel(neg_val),size(feat_norm,2));
            for i = 1:numel(Y_val)
                i_im = find(im_names==names_val(i),1);
                X_val(i,:) = feat_norm(i_im,:);
            end

            % initialize the SVM variables
            % AP_C:   average precision for all C
            % p:      permutation used for cross validation
            % n_cv:   number of partitions used in cross validation
            % AP_cv:  average precision for the cross validation segment
            % sep:    delimiter of the cross validation segments
            AP_C      = zeros(1,numel(C));
            p         = randperm(numel(Y_train));
            n_cv      = 10;
            AP_cv     = zeros(1,n_cv);
            sep       = [1, round((1:n_cv-1)*numel(Y_train)/n_cv), ...
                numel(Y_train)+1];

            % determine C with cross validation
            tic
            for C_cv = C
                toc
                tic
                fprintf('%d, %d, %d, %d\n',k_cat,k_feat,k_norm,k_C)
                reshape(AP_final(k_cat,k_feat,:,:),[numel(norms),numel(C)])
                k_C = k_C+1;

                % compute the average precision on each cross validation segment
                for i = 1:n_cv
                    % create the cross validation dataset
                    X_val_cv     = X_train(p(sep(i):sep(i+1)-1),:);
                    Y_val_cv     = Y_train(p(sep(i):sep(i+1)-1),:);
                    X_train_cv   = X_train([p(1:sep(i)-1),p(sep(i+1):sep(end)-1)],:);
                    Y_train_cv   = Y_train([p(1:sep(i)-1),p(sep(i+1):sep(end)-1)]);

                    % train the SVM
                    [w, bias] = trainLinearSVM(X_train_cv.',Y_train_cv.',C_cv);

                    % get the AP
                    scores_cv  = X_val_cv*w + bias.';
                    [~,~,info] = vl_pr(Y_val_cv, scores_cv);
                    AP_cv(i) = double(info.ap);
                end

                % record the average precision for the parameter C
                AP_C(k_C) = mean(AP_cv);
                
                % compute and record the final average precision
                [w, bias] = trainLinearSVM(X_train.',Y_train.',C_cv);
                scores_val = X_val*w + bias.';
                [~,~,info] = vl_pr(Y_val, scores_val);
                AP_final(k_cat,k_feat,k_norm,k_C) = info.ap;
            end
        end
    end
end


%% 3 - Plot the results

% initialize variables
k_cat   = 0;
k_feat  = 0;
k_norm  = 0;
k_C     = 0;
top_false_neg = zeros(numel(categories),3);
top_false_pos = zeros(numel(categories),3);

% plot summarize curves
for k_cat = 1:numel(categories)
    % setup the figure
    f = figure(k_cat); clf;
    set(k_cat,'position',[100 100 1000 600]);
    set(k_cat,'name',sprintf('Average Precision - %s',categories{k_cat}));
    
    % plot the average precision depending on C and the norm
    for k_feat = 1:numel(feat_name)
        subplot(2,round((numel(feat_name)+1)/2),k_feat);
        for k_norm = 1:numel(norms)
            % plot the average precision
            ap = AP_final(k_cat,k_feat,k_norm,:);
            semilogx(C, 100*ap(:).'); hold on;
        end
        
        % setup the subplot
        axis([C(1) C(end) 85 100])
        title(sprintf('Average Precision - feature %s',feat_name{k_feat}))
        xlabel('SVM regularization paramater C')
        ylabel('Average Precision (%)')
        legend(string(norms).')
        legend('show')
        set(gca,'yminorgrid','on')
    end
    
    % plot the average precision depending on the features and the norm
    subplot(2,round((numel(feat_name)+1)/2),numel(feat_name)+1);
    ap = reshape(max(AP_final(k_cat,:,:,:),[],4), ...
        [numel(feat_name),numel(norms)]);
    h = bar(100*ap);
    set(h(1),'facecolor',[0,0.4470,0.7410])
    set(h(2),'facecolor',[0.8500,0.3250,0.0980])
    set(h(3),'facecolor',[0.9290,0.6940,0.1250])
    
    % setup the subplot
    axis([0 1+numel(feat_name) 85 100])
    title('Average Precision - C optimized')
    xlabel('Feature dataset')
    ylabel('Average Precision (%)')
    legend(string(norms).','location','northwest')
    legend('show')
    set(gca,'xticklabel',string(feat_name).')
    set(gca,'yminorgrid','on')
end

% plot precision recall curve
for k_cat = 1:numel(categories)
    % find the best parameters
    ap = AP_final(k_cat,:,:,:);
    [~,k_feat_best,k_norm_best,k_C_best] = ...
        ind2sub(size(ap),find(ap==max(ap(:))));
    
    % set the variables
    cat        = categories{k_cat};
    feat_best  = features{k_feat_best};
    norm_best  = norms{k_norm_best};
    C_best     = C(k_C_best);
    
    % load the training and validation labels
    pos_train   = load(sprintf('data/%s_train.txt',string(cat)));
    pos_val     = load(sprintf('data/%s_val.txt',string(cat)));
    neg_train   = load('data/background_train.txt');
    neg_val     = load('data/background_val.txt');
            
    % normalize the features
    feat_norm = feat_best;
    if string(norm_best) == 'L1'
        feat_norm = ...
            bsxfun(@times,feat_norm,1./sum(abs(feat_norm),2));
    elseif string(norm_best) == 'L2'
        feat_norm = ...
            bsxfun(@times,feat_norm,1./sqrt(sum(feat_norm.^2,2)));
    end
            
    % training dataset
    names_train = [pos_train; neg_train];
    Y_train     = ...
        [ones(numel(pos_train),1);-ones(numel(neg_train),1)];
    X_train     = ...
        zeros(numel(pos_train)+numel(neg_train),size(feat_norm,2));
    for i = 1:numel(Y_train)
        i_im = find(im_names==names_train(i),1);
        X_train(i,:) = feat_norm(i_im,:);
    end
            
    % validation dataset
    names_val   = [pos_val; neg_val];
    Y_val       = ...
        [ones(numel(pos_val),1);-ones(numel(neg_val),1)];
    X_val       = ...
        zeros(numel(pos_val)+numel(neg_val),size(feat_norm,2));
    for i = 1:numel(Y_val)
        i_im = find(im_names==names_val(i),1);
        X_val(i,:) = feat_norm(i_im,:);
    end

    % train the SVM
    [w, bias] = trainLinearSVM(X_train.',Y_train.',C_best);

    % compute the training average precision
    scores_val  = X_val*w + bias.';
    
    % find the best false positive
    [~,v] = sort(scores_val.*(Y_val<0),'descend');
    top_false_pos(k_cat,:) = names_val(v(1:3)).';
%     for n_im = 1:3
%         name_im = top_false_pos(k_cat,n_im);
%         src   = sprintf('data/images/%06d.jpg',name_im);
%         dest  = sprintf('B2-%s-falsepos-%d.jpg',string(categories{k_cat}),n_im);
%         copyfile(src,dest)
%     end
    
    % find the best false negative
    [~,v] = sort(scores_val.*(Y_val>0),'ascend');
    top_false_neg(k_cat,:) = names_val(v(1:3)).';
%     for n_im = 1:3
%         name_im = top_false_neg(k_cat,n_im);
%         src   = sprintf('data/images/%06d.jpg',name_im);
%         dest  = sprintf('B2-%s-falseneg-%d.jpg',string(categories{k_cat}),n_im);
%         copyfile(src,dest)
%     end
    
    % setup the plot
    figure(10+k_cat);
    set(10+k_cat,'position',[100 100 300 300]);
    set(10+k_cat,'name',sprintf('Precision Recall curve - %s',categories{k_cat}));
    vl_pr(Y_val, scores_val);
end