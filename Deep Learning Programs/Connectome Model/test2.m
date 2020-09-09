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
testRes = [1;5;3;2];
testDataConcat = cat(3,testData1,testData2,testData3,testData4);

all_mats = testDataConcat;
all_behav = testRes;

no_sub = size(all_mats,3);

[true_prediction_r_pos, true_prediction_r_neg] = predict_behavior(all_mats,all_behav);

no_iterations = 1000;
prediction_r = zeros(no_iterations,2);
prediction_r(1,1) = true_prediction_r_pos;
prediction_r(1,2) = true_prediction_r_neg;

for it = 2:no_iterations
    fprintf('\n Performing iteration %d out of %d', it, no_iterations);
    new_behav = all_behav(randperm(no_sub));
    [prediction_r(it,1), prediction_r(it,2)] = predict_behavior(all_mats,new_behav);
end

sorted_prediction_r_pos = sort(prediction_r(:,1), 'descend');
position_pos = find(sorted_prediction_r_pos==true_prediction_r_pos);
pval_pos = position_pos(1)/no_iterations;

sorted_prediction_r_neg = sort(prediction_r(:,2), 'descend');
position_neg = find(sorted_prediction_r_neg==true_prediction_r_neg);
pval_neg = position_neg(1)/no_iterations;