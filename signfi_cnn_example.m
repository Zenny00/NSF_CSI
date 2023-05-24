%% Yongsen Ma <yma@cs.wm.edu>
% Computer Science Department, The College of William & Mary
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
    
    % disp(size(first_csi_tensor));
    % plot(squeeze(first_csi_tensor(1, 1, :)));
    
    % EXAMPLE 1 %
    %{
    csi_tensor_new = zeros(200, 60, 3, 5280);
    disp(size(csi_tensor_new));
    for i = 1:5280
        disp("Iteration: " + i);
        current_csi_tensor = csi_tensor(:, :, :, i);
        individual_sample = zeros(200, 60, 3);
        for j = 1:60
            sub_carrier = current_csi_tensor(:, j, :); % The current subcarrier
            [m, n, p] = (size(sub_carrier));
            sub_carrier = reshape(sub_carrier, m*n, p);
            [IC, A, W] = fastICA(sub_carrier', 3);
            sub_carrier = reshape(IC', m, n, []);
            individual_sample(:, j, :) = sub_carrier;
        end
        csi_tensor_new(:, :, :, i) = individual_sample;
    end
    
    disp(size(csi_tensor_new));
    csi_tensor = csi_tensor_new;
    %}
    % --------- %
    
    % EXAMPLE 2 %
  

    csi_tensor_new = zeros(200, 60, 3, 5280);
    disp(size(csi_tensor_new));
    for i = 1:5280
        disp("Iteration: " + i);
        current_csi_tensor = csi_tensor(:, :, :, i);
        individual_sample = zeros(200, 60, 3);
        for j = 1:3
            sub_carrier = current_csi_tensor(:, :, j); % The current antenna
            [IC, A, W] = fastICA(sub_carrier', 60);
            individual_sample(:, :, j) = IC';
        end
        csi_tensor_new(:, :, :, i) = individual_sample;
    end
    
    disp(size(csi_tensor_new));
    csi_tensor = csi_tensor_new;
    
    % --------- %

    % plot(squeeze(tensor_3D_ICA(1, 1, :)));
    
    % [c_row, c_col, c_ant, d] = size(csi_tensor);
    % disp(c_row);
    % disp(c_col);
    % disp(c_ant);
    % disp(d);

    % output = kICA(permute(csi_tensor, ), 3);

    % plot(csi_tensor);

    % CSI_TENSOR = reshape(csi_tensor, [], size(csi_tensor, 4)); % Reshape the data into a 2D matrix
    % disp(size(CSI_TENSOR));
    % SUB_TENSOR = CSI_TENSOR(1:1000);

    %plot(SUB_TENSOR);
    
    % plot(SUB_TENSOR);

    % ica_tensor = fastICA(SUB_TENSOR, 200);
    % disp(size(ica_tensor));

    % plot(ica_tensor);

    %CSI_TENSOR = CSI_TENSOR - mean(CSI_TENSOR, 2); % Center the data to zero the mean

    % [icasig, A, W] = fastica(CSI_TENSOR, 'verbose', 'on');
    % icasig = reshape(icasig)
    
    % [M,N,S]: CSI matrix for each instance
    % T: the total number of instances
    [M,N,S,T] = size(csi_tensor);
    Nw = 276; % number of classes

    % disp(kICA(csi_tensor, 3));

    % disp((ica_tensor, 3));
    
    rng(42); % For reproducibility
    n_epoch = 50;
    learn_rate = 0.01;
    l2_factor = 0.025;
    
    % Convolutional Neural Network settings
    layers = [imageInputLayer([M N S]);
              convolution2dLayer(4,4,'Padding',0);
              batchNormalizationLayer();
              reluLayer();
              maxPooling2dLayer(4,'Stride',4); 
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
    t2 = toc; % testing end time
    
    % plot confusion matrix
%     ttest = dummyvar(double(TTest))';
%     tpredict = dummyvar(double(YTest))';
%     [c,cm,ind,per] = confusion(ttest,tpredict);
%     plotconfusion(ttest,tpredict);

    net_info = tr;
    perf = test_accuracy;
end
