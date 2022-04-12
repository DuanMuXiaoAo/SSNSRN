function [loss_regulation] = regulate_loss(curr_k)
    k_value = extractdata(curr_k);
    sum_to_one_loss = abs(1-sum(sum(k_value)));
    [a,b] = size(k_value);
    boundarie_loss = 0;
    for_center_loss_top = 0;
    center_a = a/2;
    center_b = b/2;
    for i = 1:a
        for j = 1:b
            boundarie_loss = boundarie_loss + k_value(i,j)*(sqrt((i-center_a)^2+(j-center_b)^2));
            for_center_loss_top = for_center_loss_top + [i,j].*(k_value(i,j));
        end
    end
    sprase_loss = sum(sum(sqrt(abs(k_value))));
    center_loss = norm(([center_a,center_b]-for_center_loss_top./sum(sum(k_value))),2);
    loss_regulation = 0.5*sum_to_one_loss + 0.5*boundarie_loss + 5*sprase_loss + center_loss;
end