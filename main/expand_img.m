function [imgs] = expand_img(I)
[m,n,k] = size(I);
imgs = zeros(m,n,k,5);
I_r1 = imrotate(I,90,'crop');
I_r2 = imrotate(I,180,'crop');
I_r3 = imrotate(I,270,'crop');
I_fl = flip(I);
imgs(:,:,:,1) = I;
imgs(:,:,:,2) = I_r1;
imgs(:,:,:,3) = I_r2;
imgs(:,:,:,4) = I_r3;
imgs(:,:,:,5) = I_fl;