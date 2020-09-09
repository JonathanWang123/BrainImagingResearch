library(ppcor)
library(PEIP)
library("Hmisc")
library("iemisc")
#Need to iron out potential bugs since the results are different from the matlab version
predict_behavior <- function(rest_1_mats, PMAT_CR, corrType = "Default", edgeSel = "Default", evaluation ="Default", thresh = 0.99,verbose = FALSE){
  all_mats <- rest_1_mats
  all_behav <- PMAT_CR
  no_sub <- size(all_mats,3)
  no_node <- size(all_mats,1)
  
  behav_pred_pos <- rep(0,no_sub)
  behav_pred_neg <- rep(0,no_sub)
  
  for (leftout in 1:no_sub){
    if(verbose){
      cat('\n Leaving out subj #',leftout)
    }
    train_mats <- all_mats[,,-leftout]
    train_vcts <- train_mats
    dim(train_vcts) <- c(no_node*no_node,size(train_mats,3))
    
    train_behav <- all_behav[-leftout]
    
    #if(corrType=="Rank" || corrType=="Spearman"){
    #  r_mat <- cor(t(train_vcts),train_behav,method = "spearman")
    #  p_mat <- matrix(0,no_node*no_node)
    #  for(i in 1:(no_node*no_node)){
    #    p_mat[i] <- cor.test(t(train_vcts)[,i],train_behav,method="spearman")$p.value
    #  }
    #    dim(r_mat) <- c(no_node,no_node)
    #    dim(p_mat) <- c(no_node,no_node)
    #} else if(corrType=="Partial"){

     # train_age <- all_age
    #  train_age[leftout] = c()
    #  c(r_mat,p_mat) <- pcorr(t(train_vcts),train_behav,train_age)
    #    dim(r_mat) <- c(no_node,no_node)
    #    dim(p_mat) <- c(no_node,no_node)
      
    #} else if((corrType=="Ridge")){
    #  edge_no <- size(train_vcts,1)
    #  r_mat <- rep(0,edge_no)
    #  p_mat <- rep(0,edge_no)
    #  for(edge_i in 1:edge_no){
    #    c(,stats) = robustfit(t(train_vcts[edge_i,]), train_behav)
    #    cur_t = stats.t[2]
    #    r_mat[edge_i] = sig(cur_t)*sqrt(cur_t^2/(no_sub-1-2+cur_t^2))
    #    P_mat[edge_i] = 2+tcdf(cur_t, no_sub-1-2)
    #  }
    #  dim(r_mat) <- c(no_node,no_node)
    #  dim(p_mat) <- c(no_node,no_node)
    #} else{
      r_mat <- cor(t(train_vcts),train_behav)
      p_mat <- matrix(0,no_node*no_node)
      for(i in 1:(no_node*no_node)){
        p_mat[i] <- cor.test(t(train_vcts)[,i],train_behav)$p.value
      }
      dim(r_mat) <- c(no_node,no_node)
      dim(p_mat) <- c(no_node,no_node)
    #}

    pos_mask <- matrix(0,no_node,no_node)
    neg_mask <- matrix(0,no_node,no_node)
    
    #if(edgeSel=="Sigmoidal"){
    #  pos_edges <- which(r_mat > 0)
    #  neg_edges <- which(r_mat < 0)
    #  T <- tinv(thresh/2,no_sub-1-2)
    #  R <- sqrt(T^2/(no_sub-1-2+T^2))
    #  pos_mask[pos_edges] <- sigmf(r_mat[pos_edges], c(3/R,R/3))
    #  neg_mask[neg_efges] <- sigmf(r_mat[neg_edges], c(-3/R,R/3))
    #} else {
      pos_edges <- which(r_mat > 0 & p_mat < thresh)
      neg_edges <- which(r_mat < 0 & p_mat < thresh)
    #}

    
    pos_mask[pos_edges] <- 1
    neg_mask[neg_edges] <- 1
    
    train_sumpos <- rep(0,no_sub-1)
    train_sumneg <- rep(0,no_sub-1)
    
    for(ss in 1:size(train_sumpos)[2]){
      train_sumpos[ss] <- sum(sum(train_mats[,,ss]*pos_mask))/2
      train_sumneg[ss] <- sum(sum(train_mats[,,ss]*neg_mask))/2
    }  
    
    fit_pos <- coef(lm(train_behav ~ train_sumpos))
    fit_neg <- coef(lm(train_behav ~ train_sumneg))
    
      test_mat <- all_mats[,,leftout]
      test_sumpos <- sum(sum(test_mat*pos_mask))/2
      test_sumneg <- sum(sum(test_mat*neg_mask))/2
      behav_pred_pos[leftout] <- fit_pos[2]*test_sumpos + fit_pos[1]
      behav_pred_neg[leftout] <- fit_neg[2]*test_sumneg + fit_neg[1]

  }
  if((evaluation=="MSE")){
    
    MSE_pos <- sum((behav_pred_pos-all_behav)^2)/(no_sub-length[fit_pos]-1)
    MSE_neg <- sum((behav_pred_neg-all_behav)^2)/(no_sub-length[fit_neg]-1)
    return (c(MSE_pos, MSE_neg))
  } else {
    R_pos <- cor(behav_pred_pos,all_behav)
    P_pos <- cor.test(behav_pred_pos,all_behav)$p.value

    R_neg <- cor(behav_pred_neg,all_behav)
    P_neg <- cor.test(behav_pred_neg,all_behav)$p.value

    return (c(R_pos, R_neg ))
  }
  plot(behav_pred_pos,all_behav)
  plot(behav_pred_neg,all_behav)
}
#Random data used for testing (no actual data just random numbers which I cross 
#checked with the matlab code to make sure it worked)
testA1 <- c(1,2,3,9)
testB1 <- c(2,1,7,5)
testC1 <- c(3,7,2,3)
testD1 <- c(9,5,3,7)
testE1 <- c(testA1,testB1,testC1,testD1)
testA2 <- c(4,5,2,3)
testB2 <- c(5,4,1,7)
testC2 <- c(2,1,9,5)
testD2 <- c(3,7,5,1)
testE2 <- c(testA2,testB2,testC2,testD2)
testA3 <- c(2,1,7,6)
testB3 <- c(1,3,6,1)
testC3 <- c(7,6,7,2)
testD3 <- c(6,1,2,9)
testE3 <- c(testA3,testB3,testC3,testD3)
testA4 <- c(9,3,1,8)
testB4 <- c(3,5,4,4)
testC4 <- c(1,4,2,3)
testD4 <- c(8,4,3,8)
testE4 <- c(testA4,testB4,testC4,testD4)
test3 <- c(testE1,testE2,testE3,testE4)
dim(test3) <- c(4,4,4)
test4 <- c(1,5,3,2)


all_mats <- test3
all_behav <- test4
  
no_sub <- size(all_mats,3)

true_prediction_r_pos <- predict_behavior(all_mats,all_behav)[1]
true_prediction_r_neg <- predict_behavior(all_mats,all_behav)[2]

no_iterations <- 1000
prediction_r <- matrix(0,no_iterations,2)
prediction_r[1,1] <- true_prediction_r_pos
prediction_r[1,2] <- true_prediction_r_neg

for(it in 2:no_iterations){
  #cat('\n Performing iteration ', it,' out of ', no_iterations)
  new_behav <- all_behav[sample(no_sub)]
  
  prediction_r[it,1] <- predict_behavior(all_mats,new_behav)[1]
  prediction_r[it,2] <- predict_behavior(all_mats,new_behav)[2]  
}

sorted_prediction_r_pos <- sort(prediction_r[,1],decreasing = TRUE)
position_pos <- which(sorted_prediction_r_pos==true_prediction_r_pos)
pval_pos <- position_pos[1]/no_iterations

sorted_prediction_r_neg <- sort(prediction_r[,2],decreasing = TRUE)
position_neg <- which(sorted_prediction_r_neg==true_prediction_r_neg)
pval_neg <- position_neg[1]/no_iterations