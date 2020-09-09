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
seed = 1
set.seed(1)
#Import Data Sets
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

splitTrimmedFreesurfer <- crossValidate(trimmedFreesurfer, seed = seed, trainPercentage = 0.8)
names(splitTrimmedFreesurfer) <- c("train","test")
#Train Algorithms

#Import Libraries and Properly format data
library(keras)
library(tensorflow)
kerasTrainData <- splitTrimmedFreesurfer$train[,4:ncol(splitTrimmedFreesurfer$train)]
kerasTrainData <- cbind(kerasTrainData,splitTrimmedFreesurfer$train[,3])
colnames(kerasTrainData)[221] <- "sud_cat"

kerasTestData <- splitTrimmedFreesurfer$test[,4:ncol(splitTrimmedFreesurfer$test)]
kerasTestData <- cbind(kerasTestData,splitTrimmedFreesurfer$test[,3])
colnames(kerasTestData)[221] <- "sud_cat"

#Split Data Variables
keras_training <- kerasTrainData[,1:220]
keras_test <- kerasTestData[,1:220]

#Split results (sud_cat)
keras_trainingTarget <- kerasTrainData[,221]
keras_testTarget <- kerasTestData[,221]

keras_trainLabels <- to_categorical(keras_trainingTarget)
keras_testLabels <- to_categorical(keras_testTarget)

#Do the same but for normalized data
kerasTrainDataNormalized <- as.data.frame(lapply(kerasTrainData[1:220],standardize))
kerasTrainDataNormalized <- cbind(kerasTrainDataNormalized,splitTrimmedFreesurfer$train[,3])
colnames(kerasTrainDataNormalized)[221] <- "sud_cat"

kerasTestDataNormalized <- c()
for (x in 1:ncol(kerasTrainData[1:220])){
  kerasTestDataNormalized <- cbind(kerasTestDataNormalized,standardize(kerasTestData[,x],y=kerasTrainData[,x]))
}
kerasTestDataNormalized <- as.data.frame(kerasTestDataNormalized)
kerasTestDataNormalized <- cbind(kerasTestDataNormalized,splitTrimmedFreesurfer$test[,3])
colnames(kerasTestDataNormalized)[221] <- "sud_cat"

#Split Data Variables
keras_training_normal <- kerasTrainDataNormalized[,1:220]
keras_test_normal <- kerasTestDataNormalized[,1:220]

#Split results (sud_cat)
keras_trainingTarget_normal <- kerasTrainDataNormalized[,221]
keras_testTarget_normal <- kerasTestDataNormalized[,221]

keras_trainLabels_normal <- to_categorical(keras_trainingTarget_normal)
keras_testLabels_normal <- to_categorical(keras_testTarget_normal)

keras_training <- data.matrix(keras_training)
keras_test <- data.matrix(keras_test)
keras_training_normal <- data.matrix(keras_training_normal)
keras_test_normal <- data.matrix(keras_test_normal)

#Multi-Layer Perceptron using keras and tensorflow on normalized Data
modelMLPN<- keras_model_sequential()

modelMLPN%>% 
  layer_dense(units = 220, activation = 'relu',kernel_regularizer = regularizer_l1_l2(l1 = 0.001, l2 = 0.001), input_shape = c(220)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_batch_normalization() %>%
  layer_dense(units = 110, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_batch_normalization() %>%
  layer_dense(units = 3, activation = 'sigmoid')


modelMLPN%>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'nadam',
  metrics = 'accuracy'
)


modelMLPN%>% fit(
  keras_training_normal,
  keras_trainLabels_normal,
  epochs = 100, 
  batch_size = 557, 
  validation_split = 0.2
)


# Predict the classes for the test data
predictionsMLPN <- modelMLPN%>% predict_classes(keras_test_normal, batch_size = 140)
# Confusion matrix
confusionMatrixMLPN <- table(keras_testTarget_normal, predictionsMLPN)

correctMLPN <- 0
incorrectMLPN <- 0
for(row in 1:nrow(confusionMatrixMLPN)) {
  for(col in 1:ncol(confusionMatrixMLPN)) {
    if(row == col){
      correctMLPN <- sum(correctMLPN,confusionMatrixMLPN[row,col])
    } else {
      incorrectMLPN <- sum(incorrectMLPN,confusionMatrixMLPN[row,col])
    }
  }
}
cat("The accuracy of the multilayer perceptron is", 100*(correctMLPN/(correctMLPN+incorrectMLPN)),"%")





