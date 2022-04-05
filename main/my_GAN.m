%% Data
clear
filepath_lr = '/Users/rxthinking/Desktop/imgs/';
filepath_gt = '/Users/rxthinking/Desktop/imgs/';
img_name_lr = 'im_3.png';
img_name_gt = 'img_3_gt.png';

% augimds = augmentedImageDatastore([32 32 3], imds);
scale = 0.5;
%% Discriminator
% scale = 0.5;
% layersDiscriminator = [
%     imageInputLayer([32 32 3],"Name","imageinput","Normalization","none")
%     SpectralNormalization(convolution2dLayer([7,7],64,"NumChannels", 3, 'Stride',1,'Padding','same','Name','conv1') , "sn1" )
%     convolution2dLayer([1 1],64,"Name","conv_2","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_1")
%     leakyReluLayer(scale,"Name","relu_1")
%     convolution2dLayer([1 1],64,"Name","conv_3","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_2")
%     leakyReluLayer(scale,"Name","relu_2")
%     convolution2dLayer([1 1],64,"Name","conv_4","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_3")
%     leakyReluLayer(scale,"Name","relu_3")
%     convolution2dLayer([1 1],64,"Name","conv_5","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_4")
%     leakyReluLayer(scale,"Name","relu_4")
%     convolution2dLayer([1 1],64,"Name","conv_6","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_5")
%     leakyReluLayer(scale,"Name","relu_5")
%     convolution2dLayer([1 1],1,"Name","conv_7","Padding","same")
% ];

layersDiscriminator = [
    imageInputLayer([32 32],"Name","imageinput","Normalization","none")
    SpectralNormalization(convolution2dLayer([7,7],64,"NumChannels", 3, 'Stride',1,'Padding','same','Name','conv1') , "sn1" )
    convolution2dLayer([1 1],64,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    convolution2dLayer([1 1],64,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    convolution2dLayer([1 1],64,"Name","conv_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")
    convolution2dLayer([1 1],64,"Name","conv_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_4")
    convolution2dLayer([1 1],64,"Name","conv_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_5")
    convolution2dLayer([1 1],1,"Name","conv_7","Padding","same")
];

lgraphDiscriminator = layerGraph(layersDiscriminator);
dlnetDiscriminator = dlnetwork(lgraphDiscriminator);
%% Generator
stride = 1/scale;
Isize = 64/stride;
layersGenerator = [
imageInputLayer([64 64],"Name","imageinput","Normalization","none")
    convolution2dLayer([7 7],64,"Name","conv_1","Padding","same")
    convolution2dLayer([5 5],64,"Name","conv_2","Padding","same")
    convolution2dLayer([3 3],64,"Name","conv_3","Padding","same")
    convolution2dLayer([1 1],64,"Name","conv_4","Padding","same")
    convolution2dLayer([1 1],1,"Name","conv_5","Padding","same","Stride",[stride stride])
];


% layersGenerator_2 = [
% imageInputLayer([64 64 3],"Name","imageinput","Normalization","none")
%     convolution2dLayer([7 7],64,"Name","conv_1","Padding","same")
%     convolution2dLayer([5 5],64,"Name","conv_2","Padding","same")
%     convolution2dLayer([3 3],64,"Name","conv_3","Padding","same")
%     convolution2dLayer([1 1],64,"Name","conv_4","Padding","same")
%     convolution2dLayer([1 1],3,"Name","conv_5","Padding","same","Stride",[stride stride])
% ];



% layersGenerator_final = [
% imageInputLayer([64 64 3],"Name","imageinput","Normalization","none")
%     convolution2dLayer([7 7],64,"Name","conv_1","Padding","same")
%     convolution2dLayer([5 5],64,"Name","conv_2","Padding","same")
%     convolution2dLayer([3 3],64,"Name","conv_3","Padding","same")
%     convolution2dLayer([1 1],64,"Name","conv_4","Padding","same")
%     convolution2dLayer([1 1],3,"Name","conv_5","Padding","same")
% ];


lgraphGenerator = layerGraph(layersGenerator);


% lgraphGenerator_2 = layerGraph(layersGenerator_2);
% lgraphGenerator_final = layerGraph(layersGenerator_final);

% tempLayers = resize2dLayer("Name","resize-output-size","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[32 32]);
% lgraphGenerator = addLayers(lgraphGenerator,tempLayers);
% clear tempLayers;
% lgraphGenerator = connectLayers(lgraphGenerator,"imageinput","resize-output-size");
% lgraphGenerator = connectLayers(lgraphGenerator,"resize-output-size","crop2d/ref");

% dlnetGenerator_final = dlnetwork(lgraphGenerator_final);

dlnetGenerator = dlnetwork(lgraphGenerator);


% dlnetGenerator_2 = dlnetwork(lgraphGenerator_2);
%% 指定训练选项

numEpochs = 1000;
miniBatchSize = 1;

learnRate = 0.0002;
gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;

flipFactor = 0.3;

validationFrequency = 50;

%% 训练模型

executionEnvironment = "auto";


trailingAvgGenerator = [];
trailingAvgSqGenerator = [];
trailingAvgDiscriminator = [];
trailingAvgSqDiscriminator = [];

numValidationImages = 1;


f = figure;
f.Position(3) = 2*f.Position(3);

imageAxes = subplot(1,2,1);
scoreAxes = subplot(1,2,2);

lineScoreGenerator = animatedline(scoreAxes,'Color',[0 0.447 0.741]);
lineScoreDiscriminator = animatedline(scoreAxes, 'Color', [0.85 0.325 0.098]);
legend('Generator','Discriminator');
ylim([0 1])
xlabel("Iteration")
ylabel("Score")
grid on

iteration = 0;
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    channel = mod(epoch,3)+1;
   if mod(epoch,750) == 0
         learnRate = 0.1*learnRate;
    end
    % Reset and shuffle datastore.
%     shuffle(mbq);
    
    % Loop over mini-batches.
        iteration = iteration + 1;
        
        % Read mini-batch of data.
%         a = next(mbq);
        dlX = im2double(imread([filepath_lr,img_name_lr]));
%         dlX = imresize(dlX,[64 64]);
        dlX = dlX(:,:,channel);
        dlX = imcrop(dlX,[1,1,31,31]);
        dlX = dlarray(dlX,'SSCB'); 
        
        % Generate latent inputs for the generator network. Convert to
        % dlarray and specify the dimension labels 'SSCB' (spatial,
        % spatial, channel, batch). If training on a GPU, then convert
        % latent inputs to gpuArray.
        Z = im2double(imread([filepath_lr,img_name_lr]));
        Z = Z(:,:,channel);
         Z = imcrop(Z,[1 1 63 63]);
        dlZ = dlarray(Z,'SSCB');        
        
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlZ = gpuArray(dlZ);
        end
        
        % Evaluate the model gradients and the generator state using
        % dlfeval and the modelGradients function listed at the end of the
        % example.
        [gradientsGenerator, gradientsDiscriminator, stateGenerator, scoreGenerator, scoreDiscriminator] = ...
            dlfeval(@modelGradients, dlnetGenerator, dlnetDiscriminator, dlX, dlZ, flipFactor);
        dlnetGenerator.State = stateGenerator;
        
        % Update the discriminator network parameters.
        [dlnetDiscriminator,trailingAvgDiscriminator,trailingAvgSqDiscriminator] = ...
            adamupdate(dlnetDiscriminator, gradientsDiscriminator, ...
            trailingAvgDiscriminator, trailingAvgSqDiscriminator, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Update the generator network parameters.
        [dlnetGenerator,trailingAvgGenerator,trailingAvgSqGenerator] = ...
            adamupdate(dlnetGenerator, gradientsGenerator, ...
            trailingAvgGenerator, trailingAvgSqGenerator, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Every validationFrequency iterations, display batch of generated images using the
        % held-out generator input
        if mod(iteration,validationFrequency) == 0 || iteration == 1
%             dlnetGenerator_2.Learnables = dlnetGenerator.Learnables;
            % Generate images using the held-out generator input.

            ZValidation = im2double(imread([filepath_gt,img_name_gt]));
            [m,n,k] = size(ZValidation);
            I = zeros(m/stride,n/stride,3);
            for i = 1:3

                % ZValidation = imcrop(ZValidation,[1 1 64 64]);

                dlZValidation = dlarray(ZValidation(:,:,i),'SSCB');

                if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
                    dlZValidation = gpuArray(dlZValidation);
                end
                dlXGeneratedValidation = predict(dlnetGenerator,dlZValidation);
                I(:,:,i) = extractdata(dlXGeneratedValidation);
            end
            % Tile and rescale the images in the range [0 1].
%             I = imtile(extractdata(dlXGeneratedValidation));
%             I = rescale(I);
            
            % Display the images.
            subplot(1,2,1);
            image(imageAxes,I)
            xticklabels([]);
            yticklabels([]);
            title("Generated Images");
        end
        
        % Update the scores plot
        subplot(1,2,2)
        addpoints(lineScoreGenerator,iteration,...
            double(gather(extractdata(mean(scoreGenerator)))));
        
        addpoints(lineScoreDiscriminator,iteration,...
            double(gather(extractdata(mean(scoreDiscriminator)))));
        
        % Update the title with training progress information.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        title(...
            "Epoch: " + epoch + ", " + ...
            "Iteration: " + iteration + ", " + ...
            "Elapsed: " + string(D))
        
        drawnow
end

%% 生成新图像
ZNew = im2double(imread([filepath_gt,img_name_gt]));
[m,n,k] = size(ZNew);
  I = zeros(m/stride,n/stride,3);
for i = 1:3
dlZNew = dlarray(ZNew(:,:,i),'SSCB');

if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlZNew = gpuArray(dlZNew);
end
% dlnetGenerator_2.Learnables = dlnetGenerator.Learnables;
dlXGeneratedNew = predict(dlnetGenerator,dlZNew);

 I(:,:,i) = extractdata(dlXGeneratedNew);
end
%  I = imresize(I,[64,64]);
%  I = imcrop(I,[16,16,31,31]);
% I = imtile(extractdata(dlXGeneratedNew));
% I = rescale(I);
dlXreal = im2double(imread([filepath_lr,img_name_lr]));
% dlXreal = imresize(dlXreal,[64 64]);
% dlXreal = imcrop(dlXreal,[16,16,31,31]);
down_real = im2double(imread([filepath_gt,img_name_gt]));
figure
subplot(1,3,2),imshow(dlXreal)
title("Real DownScaled Images")
subplot(1,3,1),imshow(down_real)
title("Real Images")
[m,n,c] = size(dlXreal);
I = imresize(I,[m,n]);
subplot(1,3,3),imshow(I)
title("Generated Images")
%% 定义模型梯度、损失函数和分数
function [gradientsGenerator, gradientsDiscriminator, stateGenerator, scoreGenerator, scoreDiscriminator] = ...
    modelGradients(dlnetGenerator, dlnetDiscriminator, dlX, dlZ, flipFactor)

% Calculate the predictions for real data with the discriminator network.
dlYPred = forward(dlnetDiscriminator, dlX);

% Calculate the predictions for generated data with the discriminator network.
[dlXGenerated,stateGenerator] = forward(dlnetGenerator,dlZ);
dlYPredGenerated = forward(dlnetDiscriminator, dlXGenerated);

% Convert the discriminator outputs to probabilities.
probGenerated = sigmoid(dlYPredGenerated);

probReal = sigmoid(dlYPred);

% Calculate the score of the discriminator.
scoreDiscriminator = ((mean(probReal)+mean(1-probGenerated))/2);

% Calculate the score of the generator.
scoreGenerator = mean(probGenerated);
[a1,a2] = find(extractdata(probGenerated)>0.5);
[a,b] = size(a1);
[a3,a4] = find(extractdata(probGenerated)<=0.5);
[c,d] = size(a3);
for i = 1:a
        probGenerated(a1(i,1),a2(i,1),1,1)=1;
end
for i = 1:c
        probGenerated(a3(i,1),a4(i,1),1,1)=0;
end
% % Randomly flip a fraction of the labels of the real images.
% numObservations = size(probReal,4);
% idx = randperm(numObservations,floor(flipFactor * numObservations));
% 
% % Flip the labels
% probReal(:,:,:,idx) = 1-probReal(:,:,:,idx);

% Calculate the GAN loss.
[lossGenerator, lossDiscriminator] = ganLoss(probReal,probGenerated,dlXGenerated,dlX);

% For each network, calculate the gradients with respect to the loss.
gradientsGenerator = dlgradient(lossGenerator, dlnetGenerator.Learnables,'RetainData',true);
gradientsDiscriminator = dlgradient(lossDiscriminator, dlnetDiscriminator.Learnables);

end

function [lossGenerator, lossDiscriminator] = ganLoss(probReal,probGenerated,dlXGenerated,dlX)

% Calculate the loss for the discriminator network.
% lossDiscriminator =  -mean(log(probReal)) -mean(log(1-probGenerated));
lossDiscriminator = mse(probReal,probGenerated);

% Calculate the loss for the generator network.
lossGenerator = mse(dlX,dlXGenerated);


end