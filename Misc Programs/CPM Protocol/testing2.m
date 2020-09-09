testData1 = [1,2,3,9;
         2,1,7,5;
         3,7,2,3;
         9,5,3,7];
testData2 = [4,5,2,3;
         5,4,1,7;
         2,1,9,5;
         3,7,5,1];
testData3 = [2,1,7,6;
         1,3,6,1;
         7,6,7,2;
         6,1,2,9];
testData4 = [9,3,1,8;3,5,4,4;1,4,2,3;8,4,3,8];
testRes = [1;5;3;2]
testDataConcat = cat(3,testData1,testData2,testData3,testData4)

all_mats = testDataConcat;
all_behav = testRes;

   thresh = 0.99;
   
   no_sub = size(all_mats,3);
   no_node = size(all_mats,3);
   
   behav_pred_pos = zeros(no_sub,1);
   behav_pred_neg = zeros(no_sub,1);

    for leftout = 1:no_sub
        
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
        behav_pred_neg(leftout) = fit_neg(1)*test_sumneg + fit_neg(2);
    end
    
    [R_pos,P_pos] = corr(behav_pred_pos,all_behav);
    [R_neg,P_neg] = corr(behav_pred_neg,all_behav);