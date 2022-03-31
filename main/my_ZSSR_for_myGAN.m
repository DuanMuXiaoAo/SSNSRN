clear 
%% 数据输入与初始化
filepath = '/Users/rxthinking/Desktop/imgs/';
img_name = 'im_3.png';


% 输入图片
downscale = 0; %是否需要下采样(为了加快训练速度)
scale_for_kernelGAN = 2;  %下采样倍数
scale_for_accelerate_training = 4;

I = double(imread([filepath,img_name]));

if downscale
    im_input_size = size(I);
    im_input_size = [floor(im_input_size(:,1:2)./scale_for_accelerate_training),im_input_size(:,3)];
    size_for_downscale = im_input_size(:,1:2);
    I_hr = imresize(I,size_for_downscale);
else
    I_hr = I;
    im_input_size = size(I);
    
end

size_for_resize = im_input_size(:,1:2);
I_lr = f_my_GAN(I_hr,1000,scale_for_kernelGAN,im_input_size);  % to use KernelGAN later
I_in = imresize(I_lr,size_for_resize);
imgs_lr = expand_img(I_in);
imgs_hr = expand_img(I_hr);
% 网络结构
layers = [
    imageInputLayer(im_input_size,"Name","imageinput","Normalization","none")
    convolution2dLayer([1 1],64,"Name","conv_1","Padding","same")
    reluLayer("Name","relu_1")
    convolution2dLayer([1 1],64,"Name","conv_2","Padding","same")
    reluLayer("Name","relu_2")
    convolution2dLayer([1 1],64,"Name","conv_3","Padding","same")
    reluLayer("Name","relu_3")
    convolution2dLayer([1 1],64,"Name","conv_4","Padding","same")
    reluLayer("Name","relu_4")
    convolution2dLayer([1 1],64,"Name","conv_5","Padding","same")
    reluLayer("Name","relu_5")
    convolution2dLayer([1 1],64,"Name","conv_6","Padding","same")
    reluLayer("Name","relu_6")
    convolution2dLayer([1 1],64,"Name","conv_7","Padding","same")
    reluLayer("Name","relu_7")
    convolution2dLayer([1 1],3,"Name","conv_8","Padding","same")];

% 初始化绘制框图
f = figure;
f.Position(3) = 2*f.Position(3);

imageAxes = subplot(1,2,1);
scoreAxes = subplot(1,2,2);


lineLoss = animatedline(scoreAxes,'Color',[0 0.447 0.741]);

legend('Training');
xlabel("Iteration")
ylabel("loss")
grid on



dlnet = dlnetwork(layers);
trailingAvg = [];
trailingAvgSq = [];


%一些训练选项
epoch = 3000;
iteration = 0;
learnRate = 0.001;

%% 训练过程
for i = 1:epoch
    img_idx = mod(epoch,5) + 1;

    I_in_dl = dlarray(imgs_lr(:,:,:,img_idx),'SSCB');
    I_hr_dl = dlarray(imgs_hr(:,:,:,img_idx),'SSCB');

    iteration = iteration + 1;


    [grad,loss,I_conv] = dlfeval(@modelGradients,dlnet,I_in_dl,I_hr_dl,im_input_size);

    [dlnet,trailingAvg,trailingAvgSq] = adamupdate(dlnet,grad,...
                                                   trailingAvg,trailingAvgSq,...
                                                   iteration,learnRate);
    subplot(1,2,1)
    I_conv = extractdata(I_conv);
    I_conv = rescale(I_conv);
    image(imageAxes,I_conv)
    xticklabels([]);
    yticklabels([]);
    title("Supervised Images");

    subplot(1,2,2)
    addpoints(lineLoss,iteration,...
        double(gather(extractdata(loss))));

    title("Epoch: " + i,...
          "Loss: " + extractdata(loss))

    drawnow
end

%% 超分辨测试
I_gt = imread('/Users/rxthinking/Desktop/imgs/img_3_gt.png');

% to the size of gt
size_gt = size(I_gt);
I_for_test = imresize(I_hr,size_gt(1,1:2));
%I_for_test = I;

%to the size by scale


I_dl_for_test = dlarray(I_for_test,'SSCB');
%I_dl_for_test = I_hr_dl;
I_output_dl = forward(dlnet,I_dl_for_test);
I_sr_dl = I_dl_for_test + I_output_dl;
I_sr = extractdata(I_sr_dl);

subplot(1,3,1)
imshow(rescale(I));
title('LR')

subplot(1,3,2)
imshow(rescale(I_sr));
title('SR')


subplot(1,3,3)
imshow(rescale(I_gt));
title('HR')
% mse = mse(I_sr_dl,I_hr_dl)

%% some funtions
function [gradients,loss,dlYPred] = modelGradients(dlnet,dlX,Y,im_input_size)

dlYPred = forward(dlnet,dlX);

dlYPred = dlYPred + dlX;

loss = l1loss(dlYPred,Y)/(im_input_size(1,1)*im_input_size(1,2));

gradients = dlgradient(loss,dlnet.Learnables);

end