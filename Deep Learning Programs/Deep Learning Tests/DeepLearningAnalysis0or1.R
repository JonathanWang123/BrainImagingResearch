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
seed = 11
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

#Remove 2 values for sudcat
sudcatData_0v1 <- trimmedFreesurfer[trimmedFreesurfer$sud_cat!='2',]

splitData_0v1 <- crossValidate(sudcatData_0v1,seed = seed, trainPercentage = 0.9)
names(splitData_0v1) <- c("train","test")

#Import Libraries and Properly format data
library(keras)
library(tensorflow)

kerasTrainData_0v1 <- splitData_0v1$train[,4:ncol(splitData_0v1$train)]
kerasTrainData_0v1 <- cbind(kerasTrainData_0v1,splitData_0v1$train[,3])
colnames(kerasTrainData_0v1)[221] <- "sud_cat"

kerasTestData_0v1 <- splitData_0v1$test[,4:ncol(splitData_0v1$test)]
kerasTestData_0v1 <- cbind(kerasTestData_0v1,splitData_0v1$test[,3])
colnames(kerasTestData_0v1)[221] <- "sud_cat"

#Split Data Variables
keras_training_0v1 <- kerasTrainData_0v1[,1:220]
keras_test_0v1 <- kerasTestData_0v1[,1:220]

#Split results (sud_cat)
keras_trainingTarget_0v1 <- kerasTrainData_0v1[,221]
keras_testTarget_0v1 <- kerasTestData_0v1[,221]

keras_trainLabels_0v1 <- to_categorical(keras_trainingTarget_0v1)
keras_testLabels_0v1 <- to_categorical(keras_testTarget_0v1)

#Do the same but for normalized data
kerasTrainDataNormalized_0v1 <- as.data.frame(lapply(kerasTrainData_0v1[1:220],standardize))
kerasTrainDataNormalized_0v1 <- cbind(kerasTrainDataNormalized_0v1,splitData_0v1$train[,3])
colnames(kerasTrainDataNormalized_0v1)[221] <- "sud_cat"

kerasTestDataNormalized_0v1 <- c()
for (x in 1:ncol(kerasTrainData_0v1[1:220])){
  kerasTestDataNormalized_0v1 <- cbind(kerasTestDataNormalized_0v1,standardize(kerasTestData_0v1[,x],y=kerasTrainData_0v1[,x]))
}
kerasTestDataNormalized_0v1 <- as.data.frame(kerasTestDataNormalized_0v1)
kerasTestDataNormalized_0v1 <- cbind(kerasTestDataNormalized_0v1,splitData_0v1$test[,3])
colnames(kerasTestDataNormalized_0v1)[221] <- "sud_cat"

#Split Data Variables
keras_training_normal_0v1 <- kerasTrainDataNormalized_0v1[,1:220]
keras_test_normal_0v1 <- kerasTestDataNormalized_0v1[,1:220]

#Split results (sud_cat)
keras_trainingTarget_normal_0v1 <- kerasTrainDataNormalized_0v1[,221]
keras_testTarget_normal_0v1 <- kerasTestDataNormalized_0v1[,221]

keras_trainLabels_normal_0v1 <- to_categorical(keras_trainingTarget_normal_0v1)
keras_testLabels_normal_0v1 <- to_categorical(keras_testTarget_normal_0v1)

keras_training_0v1 <- data.matrix(keras_training_0v1)
keras_test_0v1 <- data.matrix(keras_test_0v1)
keras_training_normal_0v1 <- data.matrix(keras_training_normal_0v1)
keras_test_normal_0v1 <- data.matrix(keras_test_normal_0v1)

#Multi-Layer Perceptron using keras and tensorflow on normalized Data
modelMLPN_0v1<- keras_model_sequential()

modelMLPN_0v1%>% 
  layer_dense(units = 220, activation = 'relu',kernel_regularizer = regularizer_l1_l2(l1 = 0.001, l2 = 0.001), input_shape = c(220)) %>% 
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 2, activation = 'sigmoid')

modelMLPN_0v1%>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'nadam',
  metrics = 'accuracy'
)


modelMLPN_0v1%>% fit(
  keras_training_normal_0v1,
  keras_trainLabels_normal_0v1,
  epochs = 50, 
  batch_size = 100, 
  validation_split = 0.25
)


# Predict the classes for the test data
predictionsMLPN_0v1 <- modelMLPN_0v1%>% predict_classes(keras_test_normal_0v1, batch_size = 48)
# Confusion matrix
confusionMatrixMLPN_0v1 <- table(keras_testTarget_normal_0v1, predictionsMLPN_0v1)

correctMLPN_0v1 <- 0
incorrectMLPN_0v1 <- 0
for(row in 1:nrow(confusionMatrixMLPN_0v1)) {
  for(col in 1:ncol(confusionMatrixMLPN_0v1)) {
    if(rownames(confusionMatrixMLPN_0v1)[row] == colnames(confusionMatrixMLPN_0v1)[col]){
      correctMLPN_0v1 <- sum(correctMLPN_0v1,confusionMatrixMLPN_0v1[row,col])
    } else {
      incorrectMLPN_0v1 <- sum(incorrectMLPN_0v1,confusionMatrixMLPN_0v1[row,col])
    }
  }
}
cat("The accuracy of the multilayer perceptron is", 100*(correctMLPN_0v1/(correctMLPN_0v1+incorrectMLPN_0v1)),"%")
modelMLPN_0v1 %>% save_model_hdf5("model0v1.h5")
test <- load_model_hdf5("model0v1.h5")
summary(modelMLPN_0v1)
weightArray <- get_weights(modelMLPN_0v1)
weightsLayer1 <- weightArray[[1]]
biasLayer1 <- weightArray[[2]]
weightsLayerNormalization <- weightArray[[3]]
biasLayerNormalization <- weightArray[[4]]
weightsLayerDropout <- weightArray[[5]]
biasLayerDropout <- weightArray[[6]]
weightsLayer2 <- weightArray[[7]]
biasLayer2 <- weightArray[[8]]

