#library(NetworkToolbox)
#Source Code for relevant NetworkToolBox functions 
convertConnBrainMat <- function (MatlabData, progBar = TRUE)
{
  ###########################
  #### MISSING ARGUMENTS ####
  ###########################
  
  if(missing(MatlabData))
  {mat <- R.matlab::readMat(file.choose())
  }else{mat <- R.matlab::readMat(MatlabData)}
  
  #######################
  #### MAIN FUNCTION ####
  #######################
  
  # If data imported is not in a list, then return
  if(!is.list(mat))
  {return(mat)
  }else{ # If data is in a list
    
    # Determine structure of data
    if("data" %in% names(mat)) # Time series data
    {
      # Grab data
      dat <- unlist(mat$data, recursive = FALSE)
      
      # Grab names
      dat.names <- unlist(mat$names)
      
      # Get names of ROIs
      names(dat) <- dat.names
      
      # Check for CSF, grey and white matter
      if(any(c("Grey Matter", "White Matter", "CSF") %in% names(dat)))
      {dat <- dat[-which(names(dat) == c("Grey Matter", "White Matter", "CSF"))]}
      
      return(dat)
      
    }else if("Z" %in% names(mat)) # Average time series data
    {
      #read in matlab data
      n1<-nrow(mat$Z) #determine number of rows
      n2<-ncol(mat$Z) #determine number of columns
      if(nrow(mat$Z)!=ncol(mat$Z))
      {warning("Row length does not match column length")}
      m<-length(mat$Z)/n1/n2 #determine number of participants
      
      #change row and column names
      coln1<-matrix(0,nrow=n1) #get row names
      for(i in 1:n1)
      {coln1[i,]<-mat$names[[i]][[1]][1,1]}
      
      coln2<-matrix(0,nrow=n2) #get column names
      for(i in 1:n2)
      {coln2[i,]<-mat$names2[[i]][[1]][1,1]}
      
      dat<-mat$Z
      if(progBar)
      {pb <- txtProgressBar(max=m, style = 3)}
      
      for(i in 1:m) #populate array
      {
        dat[,,i]<-psych::fisherz2r(mat$Z[,,i])
        for(j in 1:n1)
          for(k in 1:n2)
            if(is.na(dat[j,k,i]))
            {dat[j,k,i]<-0}
        if(progBar){setTxtProgressBar(pb, i)}
      }
      if(progBar){close(pb)}
      
      colnames(dat)<-coln2
      row.names(dat)<-coln1
      
      return(list(rmat=dat,zmat=mat$Z))
    }
    
  }
}
cpmIV <- function (neuralarray, bstat, covar, thresh = .01,
                   groups = NULL, method = c("mean", "sum"),
                   model = c("linear","quadratic","cubic"),
                   corr = c("pearson","spearman"), nEdges, 
                   standardize = FALSE, cores, progBar = TRUE,
                   plots = TRUE)
{
  ####################################
  #### MISSING ARGUMENTS HANDLING ####
  ####################################
  
  if(missing(method))
  {method<-"mean"
  }else{method<-match.arg(method)}
  
  if(missing(model))
  {model<-"linear"
  }else{model<-match.arg(model)}
  
  if(missing(corr))
  {corr<-"pearson"
  }else{corr<-match.arg(corr)}
  
  if(missing(nEdges))
  {nEdges<-length(bstat)*.10
  }else{nEdges <- nEdges}
  
  if(missing(covar))
  {covar<-NULL
  }else if(!is.list(covar))
  {stop("Covariates vectors must be input as a list: list()")}
  
  ####################################
  #### MISSING ARGUMENTS HANDLING ####
  ####################################
  
  #functions list
  critical.r <- function(iter, a)
  {
    df <- iter - 2
    critical.t <- qt( a/2, df, lower.tail = F )
    cvr <- sqrt( (critical.t^2) / ( (critical.t^2) + df ) )
    return(cvr)
  }
  
  bstat<-as.vector(bstat)
  if(standardize)
  {bstat<-scale(bstat)}
  
  #number of subjects
  no_sub<-dim(neuralarray)[3]
  #number of nodes
  no_node<-ncol(neuralarray)
  
  #initialize positive and negative behavior stats
  behav_pred_pos<-matrix(0,nrow=no_sub,ncol=1)
  behav_pred_neg<-matrix(0,nrow=no_sub,ncol=1)
  
  if(is.list(covar))
  {
    cvars<-do.call(cbind,covar,1)
    cvars<-scale(cvars)
  }
  
  pos_array <- array(0,dim=c(nrow=no_node,ncol=no_node,no_sub))
  neg_array <- array(0,dim=c(nrow=no_node,ncol=no_node,no_sub))
  
  
  #perform leave-out analysis
  if(progBar)
  {pb <- txtProgressBar(max=no_sub, style = 3)}
  
  for(leftout in 1:no_sub)
  {
    train_mats<-neuralarray
    train_mats<-train_mats[,,-leftout]
    ##initialize train vectors
    #vector length
    vctrow<-ncol(neuralarray)^2
    vctcol<-length(train_mats)/nrow(train_mats)/ncol(train_mats)
    train_vcts<-matrix(0,nrow=vctrow,ncol=vctcol)
    for(i in 1:vctcol)
    {train_vcts[,i]<-as.vector(train_mats[,,i])}
    
    #behavior stats
    train_behav<-bstat
    train_behav<-train_behav[-leftout]
    
    #correlate edges with behavior
    if(nrow(train_vcts)!=(no_sub-1))
    {train_vcts<-t(train_vcts)}
    
    rmat<-vector(mode="numeric",length=ncol(train_vcts))
    pmat<-vector(mode="numeric",length=ncol(train_vcts))
    
    if(is.list(covar))
    {
      cl <- parallel::makeCluster(cores)
      doParallel::registerDoParallel(cl)
      
      pcorr<-suppressWarnings(
        foreach::foreach(i=1:ncol(train_vcts))%dopar%
          {
            temp<-cbind(train_vcts[,i],train_behav,cvars[-leftout,])
            ppcor::pcor.test(temp[,1],temp[,2],temp[,c(seq(from=3,to=2+ncol(cvars)))])
          }
      )
      parallel::stopCluster(cl)
      
      for(i in 1:length(pcorr))
      {
        rmat[i]<-pcorr[[i]]$estimate
        pmat[i]<-pcorr[[i]]$p.value
      }
      rmat<-ifelse(is.na(rmat),0,rmat)
      pmat<-ifelse(is.na(pmat),0,pmat)
    }else{rmat<-suppressWarnings(cor(train_vcts,train_behav,method=corr))}
    
    r_mat<-matrix(rmat,nrow=no_node,ncol=no_node)
    
    #set threshold and define masks
    pos_mask<-matrix(0,nrow=no_node,ncol=no_node)
    neg_mask<-matrix(0,nrow=no_node,ncol=no_node)
    
    if(!is.list(covar))
    {
      #critical r-value
      cvr<-critical.r((no_sub-1),thresh)
      pos_edges<-which(r_mat>=cvr)
      neg_edges<-which(r_mat<=(-cvr))
    }else
    {
      p_mat<-matrix(pmat,nrow=no_node,ncol=no_node)
      sig<-ifelse(p_mat<=thresh,r_mat,0)
      pos_edges<-which(r_mat>0&sig!=0)
      neg_edges<-which(r_mat<0&sig!=0)
    }
    
    
    pos_mask[pos_edges]<-1
    neg_mask[neg_edges]<-1
    
    pos_array[,,leftout] <- pos_mask
    neg_array[,,leftout] <- neg_mask
    
    #get sum of all edges in TRAIN subs (divide, if symmetric matrices)
    train_sumpos<-matrix(0,nrow=(no_sub-1),ncol=1)
    train_sumneg<-matrix(0,nrow=(no_sub-1),ncol=1)
    
    for(ss in 1:nrow(train_sumpos))
    {
      if(method=="sum")
      {
        train_sumpos[ss]<-sum(train_mats[,,ss]*pos_mask)/2
        train_sumneg[ss]<-sum(train_mats[,,ss]*neg_mask)/2
      }else if(method=="mean")
      {
        train_sumpos[ss]<-mean(train_mats[,,ss]*pos_mask)/2
        train_sumneg[ss]<-mean(train_mats[,,ss]*neg_mask)/2
      }
    }
    
    #generate regression formula with covariates
    #if(is.list(covar))
    #{cvar<-cvars[-leftout,]}
    
    #regressions----
    
    #build model on TRAIN subs
    if(model=="linear")
    {
      fit_pos<-coef(lm(train_behav~train_sumpos))
      fit_neg<-coef(lm(train_behav~train_sumneg))
    }else if(model=="quadratic")
    {
      quad_pos<-train_sumpos^2
      quad_neg<-train_sumneg^2
      
      fit_pos<-coef(lm(train_behav~train_sumpos+quad_pos))
      fit_neg<-coef(lm(train_behav~train_sumneg+quad_neg))
    }else if(model=="cubic")
    {
      cube_pos<-train_sumpos^3
      cube_neg<-train_sumneg^3
      
      quad_pos<-train_sumpos^2
      quad_neg<-train_sumneg^2
      
      fit_pos<-coef(lm(train_behav~train_sumpos+quad_pos+cube_pos))
      fit_neg<-coef(lm(train_behav~train_sumneg+quad_neg+cube_neg))
    }
    
    #run model on TEST sub
    test_mat<-neuralarray[,,leftout]
    if(method=="sum")
    {
      test_sumpos<-sum(test_mat*pos_mask)/2
      test_sumneg<-sum(test_mat*neg_mask)/2
    }else if(method=="mean")
    {
      test_sumpos<-mean(test_mat*pos_mask)/2
      test_sumneg<-mean(test_mat*neg_mask)/2
    }
    
    if(model=="linear")
    {
      behav_pred_pos[leftout]<-fit_pos[2]*test_sumpos+fit_pos[1]
      behav_pred_neg[leftout]<-fit_neg[2]*test_sumneg+fit_neg[1]
    }else if(model=="quadratic")
    {
      quad_post<-test_sumpos^2
      quad_negt<-test_sumneg^2
      
      behav_pred_pos[leftout]<-fit_pos[3]*quad_post+fit_pos[2]*test_sumpos+fit_pos[1]
      behav_pred_neg[leftout]<-fit_neg[3]*quad_negt+fit_neg[2]*test_sumneg+fit_neg[1]
    }else if(model=="cubic")
    {
      cube_post<-test_sumpos^3
      cube_negt<-test_sumneg^3
      
      quad_post<-test_sumpos^2
      quad_negt<-test_sumneg^2
      
      behav_pred_pos[leftout]<-fit_pos[4]*cube_post+fit_pos[3]*quad_post+fit_pos[2]*test_sumpos+fit_pos[1]
      behav_pred_neg[leftout]<-fit_neg[4]*cube_negt+fit_neg[3]*quad_negt+fit_neg[2]*test_sumneg+fit_neg[1]
    }
    
    if(progBar)
    {setTxtProgressBar(pb, leftout)}
  }
  if(progBar)
  {close(pb)}
  
  pos_mat <- matrix(0, nrow = no_node, ncol = no_node)
  neg_mat <- matrix(0, nrow = no_node, ncol = no_node)
  
  for(i in 1:no_node)
    for(j in 1:no_node)
    {
      pos_mat[i,j] <- sum(pos_array[i,j,])
      neg_mat[i,j] <- sum(neg_array[i,j,])
    }
  
  posmask <- ifelse(pos_mat>=nEdges,1,0)
  negmask <- ifelse(neg_mat>=nEdges,1,0)
  
  if(!is.null(colnames(neuralarray)))
  {
    colnames(posmask) <- colnames(neuralarray)
    row.names(posmask) <- colnames(posmask)
    colnames(negmask) <- colnames(neuralarray)
    row.names(negmask) <- colnames(negmask)
  }
  
  R_pos<-cor(behav_pred_pos,bstat,use="pairwise.complete.obs")
  P_pos<-cor.test(behav_pred_pos,bstat)$p.value
  R_neg<-cor(behav_pred_neg,bstat,use="pairwise.complete.obs")
  P_neg<-cor.test(behav_pred_neg,bstat)$p.value
  
  P_pos<-ifelse(round(P_pos,3)!=0,round(P_pos,3),noquote("< .001"))
  P_neg<-ifelse(round(P_neg,3)!=0,round(P_neg,3),noquote("< .001"))
  
  bstat<-as.vector(bstat)
  behav_pred_pos<-as.vector(behav_pred_pos)
  behav_pred_neg<-as.vector(behav_pred_neg)
  perror <- vector(mode="numeric",length = length(bstat))
  nerror <- vector(mode="numeric",length = length(bstat))
  
  for(i in 1:length(bstat))
  {
    perror[i] <- behav_pred_pos[i]-bstat[i]
    nerror[i] <- behav_pred_neg[i]-bstat[i]
    
    #mae
    mae_pos<-mean(abs(perror))
    mae_neg<-mean(abs(nerror))
    
    #rmse
    pos_rmse<-sqrt(mean(perror^2))
    neg_rmse<-sqrt(mean(nerror^2))
  }
  
  results<-matrix(0,nrow=2,ncol=4)
  
  results[1,1]<-round(R_pos,3)
  results[1,2]<-P_pos
  results[1,3]<-round(mae_pos,3)
  results[1,4]<-round(pos_rmse,3)
  results[2,1]<-round(R_neg,3)
  results[2,2]<-P_neg
  results[2,3]<-round(mae_neg,3)
  results[2,4]<-round(neg_rmse,3)
  
  colnames(results)<-c("r","p","mae","rmse")
  row.names(results)<-c("positive","negative")
  
  #Results list
  res <- list()
  res$results <- results
  res$posMask <- posmask
  res$negMask <- negmask
  res$posArray <- pos_array * neuralarray
  res$negArray <- neg_array * neuralarray
  res$behav <- bstat
  res$posPred <- behav_pred_pos
  res$negPred <- behav_pred_neg
  res$groups <- groups
  
  class(res) <- "cpm"
  
  if(plots)
  {plot(res)}
  
  return(res)
}
