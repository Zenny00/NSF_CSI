%
% This is an example for the following paper
% Yongsen Ma, Gang Zhou, Shuangquan Wang, Hongyang Zhao, and Woosub Jung. 2018.
% SignFi: Sign Language Recognition Using WiFi.
% Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 2, 1, Article 23 (March 2018), 21 pages.
% DOI: https://doi.org/10.1145/3191755

function [net_info, perf] = signfi_cnn_example(csi,label)
    load('dataset_lab_276_dl.mat'); % load CSI and labels into workspace
    tic; % time of starting
    % prepare for training data
    csi_abs = abs(csid_lab);
    csi_ang = angle(csid_lab);
    % csi_ang = get_signfi_phase(csi);
    csi_tensor = [csi_abs,csi_ang];
    word = categorical(label_lab);
    t0 = toc; % pre-processing time

    % PCA matrix feature space mapping
    % ===================================

    % First we define a 4-dimensional tensor the hold the output of our
    % algorithm
    
    pca_csi_tensor = zeros(200, 60, 3, 5280);
  
    % Loop through all the matricies contained in the original CSI tensor
    for index = 1:5280 
        % Print the iteration number, used for debugging
        disp("Iteration: " + index);

        % Get the current matrix from the CSI tensor
        csi_matrix = csi_tensor(:, :, :, index);
        empty_matrix = zeros(200, 60, 3);
            
        for j = 1:3
            csi_array = csi_matrix(:, :, j); % The current antenna
            csi_array = csi_array - mean(csi_array); % Center the data

            % Compute the principal components
            [coefficient_array, score, latent] = pca(csi_array);

            new_array = score * coefficient_array';
            empty_matrix(:, :, j) = new_array;
        end
        % Place the updated matrix into the 4-dimensional tensor
        pca_csi_tensor(:, :, :, index) = empty_matrix;
    end
    
    % [M,N,S]: CSI matrix for each instance
    % T: the total number of instances

    csi_tensor = pca_csi_tensor;
    
    % [M,N,S]: CSI matrix for each instanceinstance
    % T: the total number of instances
    [M,N,S,T] = size(csi_tensor);
    Nw = 276; % number of classes
    
    rng(42); % For reproducibility
    n_epoch = 15;
    learn_rate = 0.01;
    l2_factor = 0.01;
    
    % Convolutional Neural Network settings
    layers = [imageInputLayer([M N S]);

              convolution2dLayer(3,3,'Padding',1);
              batchNormalizationLayer();
              reluLayer(); 

              averagePooling2dLayer(3,"Stride",3);

              %convolution2dLayer(5,3,'Padding',1);
              %batchNormalizationLayer();
              %reluLayer(); 

              dropoutLayer(0.6);
              fullyConnectedLayer(Nw);
              softmaxLayer();
              classificationLayer()];
                         

    % get training/testing input
    K = 5;
    cv = cvpartition(T,'kfold',K); % 20% for testing
    accuList =[];
    timeList = [];
    k = 1; % for k=1:K % for k=1:K
    trainIdx = find(training(cv,k));
    testIdx = find(test(cv,k));
    trainCsi = csi_tensor(:,:,:,trainIdx);
    trainWord = word(trainIdx,1);
    testCsi = csi_tensor(:,:,:,testIdx);
    testWord = word(testIdx,1);
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

    %train_start = toc;
    tic;
    [trainedNet,tr{k,1}] = trainNetwork(trainCsi,trainWord,layers,options);
    train_stop = toc; % training end time

    [YTest, scores] = classify(trainedNet,testCsi);
    TTest = testWord;
    test_accuracy = sum(YTest == TTest)/numel(TTest);
    t2 = toc; % testing end time

    accuList = [accuList test_accuracy];
    %timeList = [timeList (train_stop - train_start)];
    timeList = [timeList train_stop];
    
    % plot confusion matrix
%     ttest = dummyvar(double(TTest))';
%     tpredict = dummyvar(double(YTest))';
%     [c,cm,ind,per] = confusion(ttest,tpredict);
%     plotconfusion(ttest,tpredict);

    %net_info = tr;
    %perf = test_accuracy;
end