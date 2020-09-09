function [R_pos,R_neg] = predict_behavior(rest_1_mats, PMAT_CR)
   all_mats = rest_1_mats;
   all_behav = PMAT_CR
   thresh = 0.99;
   
   no_sub = size(all_mats,3);
   no_node = size(all_mats,3);
   
   behav_pred_pos = zeros(no_sub,1);
   behav_pred_neg = zeros(no_sub,1);

    for leftout = 1:no_sub
        fprintf('\n Leaving out subj # %6.3f', leftout);
        
        train_mats = all_mats;
        train_mats(:,:,leftout) = [];
        train_vcts = reshape(train_mats,[],size(train_mats,3));
        
        train_behav = all_behav;
        train_behav(leftout) = [];
        
        [r_mat,p_mat] = corr(train_vcts', train_behav);
        
        r_mat = reshape(r_mat,no_node,no_node);
        p_mat = reshape(p_mat,no_node,no_node);
        
        pos_mask = zeros(no_node,no_node);
        neg_mask = zeros(no_node,no_node);
        
        pos_edges = find(r_mat > 0 & p_mat < thresh);
        neg_edges = find(r_mat < 0 & p_mat < thresh);
        
        pos_mask(pos_edges) = 1;
        neg_mask(neg_edges) = 1;
        
        train_sumpos = zeros(no_sub-1,1);
        train_sumneg = zeros(no_sub-1,1);
        
        for ss = 1:size(train_sumpos)
            train_sumpos(ss) = sum(sum(train_mats(:,:,ss).*pos_mask))/2;
            train_sumneg(ss) = sum(sum(train_mats(:,:,ss).*neg_mask))/2;
        end
        
        fit_pos = polyfit(train_sumpos,train_behav,1);
        fit_neg = polyfit(train_sumneg,train_behav,1);
        
        test_mat = all_mats(:,:,leftout);
        test_sumpos = sum(sum(test_mat.*pos_mask))/2;
        test_sumneg = sum(sum(test_mat.*neg_mask))/2;
        
        behav_pred_pos(leftout) = fit_pos(1)*test_sumpos + fit_pos(2);
        behav_pred_neg(leftout) = fit_pos(1)*test_sumneg + fit_neg(2);
    end
    
    [R_pos,P_pos] = corr(behav_pred_pos,all_behav);
    [R_neg,P_neg] = corr(behav_pred_neg,all_behav);
end