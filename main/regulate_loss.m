function [loss_regulation] = regulate_loss(dlgenerate, dlreal, curr_k)
    if canUseGPU
        I_gen = gather(extractdata(dlgenerate));
        I_rea = gather(extractdata(dlreal));
    else
        I_gen = extractdata(dlgenerate);
        I_rea = extractdata(dlreal);
    end
    bicubic_loss = mse(I_gen, imresize(I_rea,[32,32]));
    sum_to_one_loss =
end