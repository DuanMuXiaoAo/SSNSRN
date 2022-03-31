function [I] = f_my_GAN(I_input,numEpochs,scale,im_size)

%% Discriminator

im_input_size = im_size(:,1:2);
layersDiscriminator = [
    imageInputLayer(ceil(im_input_size./scale),"Name","imageinput","Normalization","none")
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
stride = scale;

layersGenerator = [
imageInputLayer(im_input_size,"Name","imageinput","Normalization","none")
    convolution2dLayer([7 7],64,"Name","conv_1","Padding","same")
    convolution2dLayer([5 5],64,"Name","conv_2","Padding","same")
    convolution2dLayer([3 3],64,"Name","conv_3","Padding","same")
    convolution2dLayer([1 1],64,"Name","conv_4","Padding","same")
    convolution2dLayer([1 1],1,"Name","conv_5","Padding","same","Stride",[stride stride])
];


lgraphGenerator = layerGraph(layersGenerator);

dlnetGenerator = dlnetwork(lgraphGenerator);


%% 指定训练选项

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

scoreAxes = subplot(1,1,1);

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
    
    % Loop over mini-batches.
        iteration = iteration + 1;
        
        % Read mini-batch of data.
        dlX = im2double(I_input);
        dlX = dlX(:,:,channel);
        dlX = imcrop(dlX,[1,1,31,31]);
        dlX = dlarray(dlX,'SSCB'); 
        
        % Generate latent inputs for the generator network. Convert to
        % dlarray and specify the dimension labels 'SSCB' (spatial,
        % spatial, channel, batch). If training on a GPU, then convert
        % latent inputs to gpuArray.
        Z = im2double(I_input);
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
        
        
        
        % Update the scores plot
        subplot(1,1,1)
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
 ZNew = im2double(I_input);
 m = ceil(im_input_size./scale);
  I = zeros(m(1,1),m(1,2),3);

for i = 1:3
dlZNew = dlarray(ZNew(:,:,i),'SSCB');

if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlZNew = gpuArray(dlZNew);
end
% dlnetGenerator_2.Learnables = dlnetGenerator.Learnables;
dlXGeneratedNew = predict(dlnetGenerator,dlZNew);

 I(:,:,i) = extractdata(dlXGeneratedNew);
end

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
end
