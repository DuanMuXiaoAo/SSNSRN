%function [kernel] = calcu_k(dlGenerator)
    clear
    load('dlnet.mat')
    delta = double(ones(1,1,1,1));
    value = extractdata(dlnetGenerator.Learnables(1,:).Value{1});
    dl = dlarray(delta, 'SSCB');
    [a,b,c,d] = size(dlnetGenerator.Learnables(1,:).Value{1});
    conv_size =[a,b];
    layer = [imageInputLayer(size(delta),"Name","imageinput","Normalization","none")
        convolution2dLayer([a,b],d,"Name","conv_2","Padding",12,'Weights',value)];
    dlnet = dlnetwork(layer);
    filter = dlnet.forward(dl);
    size(filter)

    for i = 3:2:9
        value = extractdata(dlnetGenerator.Learnables(i,:).Value{1});
        [a1,b1,c1,d1] = size(dlnetGenerator.Learnables(i,:).Value{1});
        [a,b,c,d] = size(filter);
        conv_size = [a,b];
        filter_value = extractdata(filter);
        dl = dlarray(filter_value,"SSCB");
        layer = [imageInputLayer([a,b,c],"Name","in","Normalization","none")
            convolution2dLayer([a1,b1],d1,"Name","conv_2","Padding",0,"Weights",value)];
        dlnet = dlnetwork(layer);
        filter = dlnet.forward(dl);
        size(filter)
    end
    kernel = squeeze(filter);
    