%function [kernel] = calcu_k(dlGenerator)
    load('dlnet.mat')
    delta = double(ones(1,1,1,1));
    value = extractdata(dlnetGenerator.Learnables(1,:).Value{1});
    dl = dlarray(value, 'SSCB');
    [a,b,c,d] = size(dlnetGenerator.Learnables(1,:).Value{1});
    conv_size =[a,b];
    layer = [imageInputLayer(size(delta),"Name","imageinput","Normalization","none")
        convolution2dLayer(conv_size,d,"Name","conv_2","Padding",12,'Weights',extractdata(dlnetGenerator.Learnables(1,:).Value{1}))];
    dlnet = dlnetwork(layer);
    filter = dlnet.forward(dl);
    for i = 3:2:9
        value = extractdata(dlnetGenerator.Learnables(i,:).Value{1});
        dl = dlarray(value,'SSCB');
        [a1,b1,c1,d1] = size(dl);
        [a,b,c,d] = size(filter);
        conv_size = [a,b];
        layer = [imageInputLayer([a1,b1,c1],"Name","in","Normalization","none")
            convolution2dLayer(conv_size,d,"Name","conv_2","Padding","same")];
        dlnet = dlnetwork(layer);
        filter = dlnet.forward(dl);
    end