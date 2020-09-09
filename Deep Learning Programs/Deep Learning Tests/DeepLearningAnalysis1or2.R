#This code is not the final version, use with discretion and experiment with values as you please
concatenateMatrix <- function(x){
  stringFinal <- ""
  for(str in x){
    stringFinal <- paste(stringFinal,str, sep="|")
  }
  stringFinal <- substring(stringFinal,2,nchar(stringFinal))
}
trimByCol <- function(x,y){
  if(length(y)>1){
    trimBy <- concatenateMatrix(y)
  } else {
    trimBy <- y
  }
  trimmedMatrix <- x[, grep(trimBy ,colnames(x))]
}
crossValidate<- function(x,trainPercentage=0.8,by = "row", seed=1){
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
normalize <- function(x,y=x) {
  num <- x - min(y)
  denom <- max(y) - min(y)
  return (num/denom)
}
standardize <- function(x,y=x){
  num <- x - mean(y)
  denom <- sd(y)
  return (num/denom)
}
randomTest <- function(x,percentage=0.2, seed = 1){
  set.seed(seed)
  sample_size <- floor(percentage*nrow(x))
  picked <- sample(seq_len(nrow(x)),size = sample_size)
  
  test <- x[picked,]
  return(test)
}
seed = 1
set.seed(seed)
#Import Data Sets (change filepath to your computer)
freesurfer <- read.csv("D:/Research/Yihong Zhao/Brain Data Sets/freesurfer.csv")
hcpSud <- read.csv("D:/Research/Yihong Zhao/Brain Data Sets/hcp.sud.csv")

#trimVars is a matrix of all the variable substrings you want to include
trimVars<-c("Subject","_Thck$","_Area$","_GrayVol$",
            "FS_L_Hippo_Vol","FS_L_Amygdala_Vol","FS_L_AccumbensArea_Vol", "FS_L_VentDC_Vol",
            "FS_L_ThalamusProper_Vol","FS_L_Caudate_Vol","FS_L_Putamen_Vol","FS_L_Pallidum_Vol",
            "FS_R_Hippo_Vol","FS_R_Amygdala_Vol","FS_R_AccumbensArea_Vol", "FS_R_VentDC_Vol",
            "FS_R_ThalamusProper_Vol","FS_R_Caudate_Vol","FS_R_Putamen_Vol","FS_R_Pallidum_Vol")

#Split the dataset by column, merge by Subject, then separate subjects into training and testing data
trimmedFreesurfer <- trimByCol(freesurfer,trimVars)
trimmedFreesurfer <- merge(hcpSud,trimmedFreesurfer, by.x = "Subject",by.y = "Subject")

#Remove 1 values for sudcat
sudcatData_1v2 <- trimmedFreesurfer[trimmedFreesurfer$sud_cat!='0',]
sudcatData_1v2[which(sudcatData_1v2[,3]=='1'),3] <- '0'
sudcatData_1v2[which(sudcatData_1v2[,3]=='2'),3] <- '1'

splitData_1v2 <- crossValidate(sudcatData_1v2,seed = seed, trainPercentage = 0.95)
names(splitData_1v2) <- c("train","test")

#Import Libraries and Properly format data
library(keras)
library(tensorflow)

kerasTrainData_1v2 <- splitData_1v2$train[,4:ncol(splitData_1v2$train)]
kerasTrainData_1v2 <- cbind(kerasTrainData_1v2,splitData_1v2$train[,3])
colnames(kerasTrainData_1v2)[221] <- "sud_cat"

kerasTestData_1v2 <- splitData_1v2$test[,4:ncol(splitData_1v2$test)]
kerasTestData_1v2 <- cbind(kerasTestData_1v2,splitData_1v2$test[,3])
colnames(kerasTestData_1v2)[221] <- "sud_cat"

#Split Data Variables
keras_training_1v2 <- kerasTrainData_1v2[,1:220]
keras_test_1v2 <- kerasTestData_1v2[,1:220]

#Split results (sud_cat)
keras_trainingTarget_1v2 <- kerasTrainData_1v2[,221]
keras_testTarget_1v2 <- kerasTestData_1v2[,221]

keras_trainLabels_1v2 <- to_categorical(keras_trainingTarget_1v2)
keras_testLabels_1v2 <- to_categorical(keras_testTarget_1v2)

#Do the same but for normalized data
kerasTrainDataNormalized_1v2 <- as.data.frame(lapply(kerasTrainData_1v2[1:220],standardize))
kerasTrainDataNormalized_1v2 <- cbind(kerasTrainDataNormalized_1v2,splitData_1v2$train[,3])
colnames(kerasTrainDataNormalized_1v2)[221] <- "sud_cat"

kerasTestDataNormalized_1v2 <- c()
for (x in 1:ncol(kerasTrainData_1v2[1:220])){
  kerasTestDataNormalized_1v2 <- cbind(kerasTestDataNormalized_1v2,standardize(kerasTestData_1v2[,x],y=kerasTrainData_1v2[,x]))
}
kerasTestDataNormalized_1v2 <- as.data.frame(kerasTestDataNormalized_1v2)
kerasTestDataNormalized_1v2 <- cbind(kerasTestDataNormalized_1v2,splitData_1v2$test[,3])
colnames(kerasTestDataNormalized_1v2)[221] <- "sud_cat"

#Split Data Variables
keras_training_normal_1v2 <- kerasTrainDataNormalized_1v2[,1:220]
keras_test_normal_1v2 <- kerasTestDataNormalized_1v2[,1:220]

#Split results (sud_cat)
keras_trainingTarget_normal_1v2 <- kerasTrainDataNormalized_1v2[,221]
keras_testTarget_normal_1v2 <- kerasTestDataNormalized_1v2[,221]

keras_trainLabels_normal_1v2 <- to_categorical(keras_trainingTarget_normal_1v2)
keras_testLabels_normal_1v2 <- to_categorical(keras_testTarget_normal_1v2)

keras_training_1v2 <- data.matrix(keras_training_1v2)
keras_test_1v2 <- data.matrix(keras_test_1v2)
keras_training_normal_1v2 <- data.matrix(keras_training_normal_1v2)
keras_test_normal_1v2 <- data.matrix(keras_test_normal_1v2)

#Multi-Layer Perceptron using keras and tensorflow on normalized Data
modelMLPN_1v2<- keras_model_sequential()

modelMLPN_1v2%>% 
  layer_dense(units = 220, activation = 'relu',kernel_regularizer = regularizer_l1_l2(l1 = 0.001, l2 = 0.001), input_shape = c(220)) %>% 
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 2, activation = 'sigmoid')

modelMLPN_1v2%>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'nadam',
  metrics = 'accuracy'
)


modelMLPN_1v2%>% fit(
  keras_training_normal_1v2,
  keras_trainLabels_normal_1v2,
  epochs = 100, 
  batch_size = 200, 
  validation_split = 0.3
)


# Predict the classes for the test data
predictionsMLPN_1v2 <- modelMLPN_1v2%>% predict_classes(keras_test_normal_1v2, batch_size = 54)
# Confusion matrix
confusionMatrixMLPN_1v2 <- table(keras_testTarget_normal_1v2, predictionsMLPN_1v2)

correctMLPN_1v2 <- 0
incorrectMLPN_1v2 <- 0
for(row in 1:nrow(confusionMatrixMLPN_1v2)) {
  for(col in 1:ncol(confusionMatrixMLPN_1v2)) {
    if(rownames(confusionMatrixMLPN_1v2)[row] == colnames(confusionMatrixMLPN_1v2)[col]){
      correctMLPN_1v2 <- sum(correctMLPN_1v2,confusionMatrixMLPN_1v2[row,col])
    } else {
      incorrectMLPN_1v2 <- sum(incorrectMLPN_1v2,confusionMatrixMLPN_1v2[row,col])
    }
  }
}
cat("The accuracy of the multilayer perceptron is", 100*(correctMLPN_1v2/(correctMLPN_1v2+incorrectMLPN_1v2)),"%")
modelMLPN_1v2 %>% save_model_hdf5("model1v2.h5")
test <- load_model_hdf5("model1v2.h5")
summary(modelMLPN_1v2)
weightArray <- get_weights(modelMLPN_1v2)
weightsLayer1 <- weightArray[[1]]
biasLayer1 <- weightArray[[2]]
weightsLayerNormalization <- weightArray[[3]]
biasLayerNormalization <- weightArray[[4]]
weightsLayerDropout <- weightArray[[5]]
biasLayerDropout <- weightArray[[6]]
weightsLayer2 <- weightArray[[7]]
biasLayer2 <- weightArray[[8]]

