concatenateMatrix <- function(x){ #Takes matrix of strings and combines them into one string separated by |
  stringFinal <- ""
  for(str in x){
    stringFinal <- paste(stringFinal,str, sep="|")
  }
  stringFinal <- substring(stringFinal,2,nchar(stringFinal))
}
trimByCol <- function(x,y){ #Takes data matrix and string values separated by | and trims out all columns not listed in the string
  if(length(y)>1){
    trimBy <- concatenateMatrix(y)
  } else {
    trimBy <- y
  }
  trimmedMatrix <- x[, grep(trimBy ,colnames(x))]
}
crossValidate<- function(x,trainPercentage=0.8,by = "row", seed=1){ #Randomely split matrix into train and test data
  set.seed(seed)
  if(by=="row"){
    sample_size <- floor(trainPercentage*nrow(x))
    picked <- sample(seq_len(nrow(x)),size = sample_size)
    
    train <- x[picked,]
    test <- x[-picked,]
    returnVal <- list(train,test)
    return(returnVal)
  } else if(by=="col"){
    sample_size <- floor(trainPercentage*ncol(x))
    picked <- sample(seq_len(ncol(x)),size = sample_size)
    
    train <- x[,picked]
    test <- x[,-picked,]
    returnVal <- list(train,test)
    return(returnVal)
  } else {
    return(NULL)
  }
}
normalize <- function(x,y=x) { #Normalize list x in respect to list y
  num <- x - min(y)
  denom <- max(y) - min(y)
  
  return (num/denom)
  
}
standardize <- function(x,y=x){ #Standardize list x in respect to list y
  num <- x - mean(y)
  denom <- sd(y)
  return (num/denom)
  
}
library(lime)      
library(vip)      
library(pdp)      
library(ggplot2)   
library(xgboost)


#Set random seed for semi-reproducibility (TensorFlow on the backend has its own randomization so setting a seed doesnt effect it)
seed = 12345
set.seed(seed)

#Import Database
simu.f <- readRDS("D:/Research/Yihong Zhao/Brain Data Sets/Simulated Data 2/simu.f.2.rds")

weight.f <- readRDS("D:/Research/Yihong Zhao/Brain Data Sets/Simulated Data 2/weight.f.updated.rds")

#Separate out variables used to predict and variable to be predicted
simu.f.age <- cbind(simu.f[,1:220],simu.f[,221])
colnames(simu.f.age)[221] <- "age"

#Split into train and test data
simu.f.age.split <- crossValidate(simu.f.age,seed = seed, trainPercentage = 0.8)
names(simu.f.age.split) <- c("train","test")

simu.f.age.train <- simu.f.age.split$train[,1:ncol(simu.f.age.split$train)]
simu.f.age.test <- simu.f.age.split$test[,1:ncol(simu.f.age.split$test)]

#Normalize Train Data
simu.f.age.trainData <- as.data.frame(apply(simu.f.age.train[,1:220],2,normalize))

#Normalize test data with respect to train data
simu.f.age.testData <- c()
for (x in 1:220){
  simu.f.age.testData <- cbind(simu.f.age.testData,normalize(simu.f.age.test[,x],y=simu.f.age.train[,x]))
}
colnames(simu.f.age.testData) <- colnames(simu.f.age.trainData)
simu.f.age.testData <- as.data.frame(simu.f.age.testData)


#Format data to be used for training
simu.f.age.trainTarget <- (simu.f.age.train[,221])
simu.f.age.testTarget <- (simu.f.age.test[,221])

simu.f.age.trainData <- data.matrix(simu.f.age.trainData)
simu.f.age.trainLabels <- data.matrix(simu.f.age.trainTarget)
simu.f.age.testData <- data.matrix(simu.f.age.testData)
simu.f.age.testLabels <- data.matrix(simu.f.age.testTarget)


xgboost_model <- xgboost(
    data = simu.f.age.trainData, 
    label = simu.f.age.trainTarget, 
    max.depth = 2, 
    eta = 1, 
    nthread = 10, 
    nrounds = 100, 
    objective = "reg:squarederror")

pred_xgb <- predict(xgboost_model, simu.f.age.testData)

sum <- 0
for(x in 1:nrow(simu.f.age.testData)){
  sum <- sum + abs(simu.f.age.testTarget[x] - pred_xgb[x])
}
mae <- sum/nrow(simu.f.age.testData)