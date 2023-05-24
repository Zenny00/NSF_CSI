%
% This is an example for the following paper
% Yongsen Ma, Gang Zhou, Shuangquan Wang, Hongyang Zhao, and Woosub Jung. 2018.
% SignFi: Sign Language Recognition Using WiFi.
% Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 2, 1, Article 23 (March 2018), 21 pages.
% DOI: https://doi.org/10.1145/3191755

function [net_info, perf] = signfi_cnn_example(csi,label)
    load('dataset_lab_150.mat'); % load CSI and labels into workspace
    tic; % time of starting
    csi = csi1;
    disp(size(csi));

    % prepare for training data
    csi_abs = abs(csi);
    csi_ang = angle(csi);

    csi_tensor = [csi_abs, csi_ang];
    word = categorical(label(1:1500));
    t0 = toc; % pre-processing time

    index = randperm(size(csi_tensor, 4));
    % [M,N,S]: CSI matrix for each instanceinstance
    % T: the total number of instances
    [M,N,S,T] = size(csi_tensor);
    N = 5;
    Nw = 150; % number of classes
    
    rng(42); % For reproducibility
    n_epoch = 75;
    learn_rate = 0.006;
    l2_factor = 0.01;
    
    % Convolutional Neural Network settings
    layers = [imageInputLayer([M N S]);
              convolution2dLayer(7,22,'Padding',3);
              batchNormalizationLayer();
              reluLayer();

              maxPooling2dLayer(4,'Stride',2); 
              convolution2dLayer(3,58,'Padding',3);
              batchNormalizationLayer();
              reluLayer();
               
              dropoutLayer(0.5); % Dropout layer with a rate of 0.5
              fullyConnectedLayer(Nw);
              softmaxLayer();
              classificationLayer()];
                       
    % get training/testing input
    K = 5;
    cv = cvpartition(T,'kfold',K); % 20% for testing
    k = 1; % for k=1:K
        trainIdx = find(training(cv,k));
        testIdx = find(test(cv,k));
        trainCsi = csi_tensor(:,:,:,trainIdx);
        trainWord = word(trainIdx,1);
        testCsi = csi_tensor(:,:,:,testIdx);
        testWord = word(testIdx,1);
        
        
        trainSize = size(trainIdx, 1);
        pca_csi_tensor = zeros(200, 5, 3, trainSize);
  
        % Loop through all the matricies contained in the original CSI tensor
        for index = 1:trainSize 
            % Print the iteration number, used for debugging
            disp("Iteration: " + index);
    
            % Get the current matrix from the CSI tensor
            csi_matrix = trainCsi(:, :, :, index);
            empty_matrix = zeros(200, 5, 3);
                
            for j = 1:3
                csi_array = csi_matrix(:, :, j); % The current antenna
                csi_array = csi_array - mean(csi_array); % Center the data
    
                % Compute the principal components
                [coefficient_array, score, latent] = pca(csi_array);
    
                explained = cumsum(latent) / sum(latent);
                reduced_array = csi_array * coefficient_array(:, 1:5);
    
                empty_matrix(:, :, j) = reduced_array;
            end
            % Place the updated matrix into the 4-dimensional tensor
            pca_csi_tensor(:, :, :, index) = empty_matrix;
        end
    
        trainCsi = pca_csi_tensor;

        testSize = size(testIdx, 1);
        pca_csi_tensor = zeros(200, 5, 3, testSize);
  
        % Loop through all the matricies contained in the original CSI tensor
        for index = 1:testSize 
            % Print the iteration number, used for debugging
            disp("Iteration: " + index);
    
            % Get the current matrix from the CSI tensor
            csi_matrix = testCsi(:, :, :, index);
            empty_matrix = zeros(200, 5, 3);
                
            for j = 1:3
                csi_array = csi_matrix(:, :, j); % The current antenna
                csi_array = csi_array - mean(csi_array); % Center the data
    
                % Compute the principal components
                [coefficient_array, score, latent] = pca(csi_array);
    
                explained = cumsum(latent) / sum(latent);
                reduced_array = csi_array * coefficient_array(:, 1:5);
    
                empty_matrix(:, :, j) = reduced_array;
            end
            % Place the updated matrix into the 4-dimensional tensor
            pca_csi_tensor(:, :, :, index) = empty_matrix;
        end
    
        testCsi = pca_csi_tensor;
        valData = {testCsi,testWord};
        
        % training options for the Convolutional Neural Network
        options = trainingOptions('sgdm','ExecutionEnvironment','gpu',...
                          'MaxEpochs',n_epoch,...
                          'InitialLearnRate',learn_rate,...
                          'L2Regularization',l2_factor,...
                          'ValidationData',valData,...
                          'ValidationFrequency',10,...
                          'ValidationPatience',Inf,...
                          'Shuffle','every-epoch',...
                          'Verbose',false,...
                          'Plots','training-progress');

        [trainedNet,tr{k,1}] = trainNetwork(trainCsi,trainWord,layers,options);

        t1 = toc; % training end time

        [YTest, scores] = classify(trainedNet,testCsi);
        TTest = testWord;
        test_accuracy = sum(YTest == TTest)/numel(TTest);
        disp(test_accuracy);
        t2 = toc; % testing end time
        
    %end
    
    %plot confusion matrix
    %ttest = dummyvar(double(TTest))';
    %tpredict = dummyvar(double(YTest))';
    %[c,cm,ind,per] = confusion(ttest,tpredict);
    %plotconfusion(ttest,tpredict);

    net_info = tr;
    perf = test_accuracy;
end