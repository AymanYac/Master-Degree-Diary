% EXERCISE1: basic training and testing of a classifier

% setup MATLAB to use our software
setup ;

% --------------------------------------------------------------------
% Stage A: Data Preparation
% --------------------------------------------------------------------

% Load training data
% encoding = 'bovw' ;
% encoding = 'vlad' ;
encoding = 'fv' ;

% category = 'motorbike' ;
% category = 'aeroplane' ;
% category = 'person' ;

cat = {'motorbike'; 'aeroplane'; 'person'} ;
frac = [.1, .5, +inf] ;
AP = zeros(2,3,3) ;

for m = 1:2
    for i = 1:3
        category = cat{i};
        for j = 1:3
            fraction = frac(j);

            pos = load(['data/' category '_train_' encoding '.mat']) ;
            neg = load(['data/background_train_' encoding '.mat']) ;

            names = {pos.names{:}, neg.names{:}};
            histograms = [pos.histograms, neg.histograms] ;
            labels = [ones(1,numel(pos.names)), - ones(1,numel(neg.names))] ;
            clear pos neg ;

            % Load testing data
            pos = load(['data/' category '_val_' encoding '.mat']) ;
            neg = load(['data/background_val_' encoding '.mat']) ;

            testNames = {pos.names{:}, neg.names{:}};
            testHistograms = [pos.histograms, neg.histograms] ;
            testLabels = [ones(1,numel(pos.names)), - ones(1,numel(neg.names))] ;
            clear pos neg ;

            % For stage G: throw away part of the training data
            % fraction = .1 ;
            % fraction = .5 ;
            % fraction = +inf ;

            sel = vl_colsubset(1:numel(labels), fraction, 'uniform') ;
            names = names(sel) ;
            histograms = histograms(:,sel) ;
            labels = labels(:,sel) ;
            clear sel ;

            % count how many images are there
            fprintf('Number of training images: %d positive, %d negative\n', ...
                    sum(labels > 0), sum(labels < 0)) ;
            fprintf('Number of testing images: %d positive, %d negative\n', ...
                    sum(testLabels > 0), sum(testLabels < 0)) ;

            % For Stage E: Vary the image representation
            % histograms = removeSpatialInformation(histograms) ;
            % testHistograms = removeSpatialInformation(testHistograms) ;

            if m==2
                % For Stage F: Vary the classifier (Hellinger kernel)
                histograms = bsxfun(@times, histograms, 1./sum(abs(histograms),1)) ;
                histograms = sign(histograms).*sqrt(abs(histograms)) ;
                testHistograms = bsxfun(@times, testHistograms, 1./sum(abs(testHistograms),1)) ;
                testHistograms = sign(testHistograms).*sqrt(abs(testHistograms)) ;
            else
                % L2 normalize the histograms before running the linear SVM
                histograms = bsxfun(@times, histograms, 1./sqrt(sum(histograms.^2,1))) ;
                testHistograms = bsxfun(@times, testHistograms, 1./sqrt(sum(testHistograms.^2,1))) ;
                
                % L1 normalize the histograms before running the linear SVM
                % histograms = bsxfun(@times, histograms, 1./sum(abs(histograms),1)) ;
                % testHistograms = bsxfun(@times, testHistograms, 1./sum(abs(testHistograms),1)) ;
            end

            % --------------------------------------------------------------------
            % Stage B: Training a classifier
            % --------------------------------------------------------------------

            % Train the linear SVM. The SVM paramter C should be
            % cross-validated. Here for simplicity we pick a valute that works
            % well with all kernels.
            C = 10 ;
            [w, bias] = trainLinearSVM(histograms, labels, C) ;

            % Evaluate the scores on the training data
            scores = w' * histograms + bias ;

            % Visualize the ranked list of images
            % figure(1) ; clf ; set(1,'name','Ranked training images (subset)') ;
            % displayRankedImageList(names, scores)  ;

            % Visualize the precision-recall curve
            % figure(2) ; clf ; set(2,'name','Precision-recall on train data') ;
            % vl_pr(labels, scores) ;

            % --------------------------------------------------------------------
            % Stage C: Classify the test images and assess the performance
            % --------------------------------------------------------------------

            % Test the linear SVM
            testScores = w' * testHistograms + bias ;

            % Visualize the ranked list of images
            % figure(3) ; clf ; set(3,'name','Ranked test images (subset)') ;
            % displayRankedImageList(testNames, testScores)  ;

            % Visualize visual words by relevance on the first image
            % [~,best] = max(testScores) ;
            % displayRelevantVisualWords(testNames{best},w)

            % Visualize the precision-recall curve
            % figure(4) ; clf ; set(4,'name','Precision-recall on test data') ;
            % vl_pr(testLabels, testScores) ;

            % Print results
            [drop,drop,info] = vl_pr(testLabels, testScores) ;
            fprintf('Test AP: %.2f\n', info.auc) ;

            [drop,perm] = sort(testScores,'descend') ;
            fprintf('Correctly retrieved in the top 36: %d\n', sum(testLabels(perm(1:36)) > 0)) ;

            % Record the AP results
            AP(m,i,j) = info.auc ;

        end
    end
end

% Plot the linear kernel performance
figure(5); clf; set(5,'name','Average Precision with linear kernel') ;
hold on
ap = AP(1,1,:) ;
plot([0.1, 0.5, 1], 100*ap(:).');
ap = AP(1,2,:) ;
plot([0.1, 0.5, 1], 100*ap(:).');
ap = AP(1,3,:) ;
plot([0.1, 0.5, 1], 100*ap(:).');
axis([0.1 1 0 100])
title('Average Precision with linear kernel')
xlabel('Fraction of training data used')
ylabel('Average Precision (%)')
legend('motorbike', 'aeroplane', 'person')
legend('show')

% Plot the Hellinger kernel performance
figure(6); clf; set(6,'name','Average Precision with Hellinger kernel') ;
hold on
ap = AP(2,1,:) ;
plot([0.1, 0.5, 1], 100*ap(:).');
ap = AP(2,2,:) ;
plot([0.1, 0.5, 1], 100*ap(:).');
ap = AP(2,3,:) ;
plot([0.1, 0.5, 1], 100*ap(:).');
axis([0.1 1 0 100])
title('Average Precision with Hellinger kernel')
xlabel('Fraction of training data used')
ylabel('Average Precision (%)')
legend('motorbike', 'aeroplane', 'person')
legend('show')