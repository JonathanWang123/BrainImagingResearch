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
test <- function(x,y=x){
  return(x)
}
kerasAuc <- function(y_true,y_pred){
  true= k_flatten(y_true)
  pred = k_flatten(y_pred)
  
  #total number of elements in this batch
  totalCount = k_shape(true)[1]
  
  
  #sorting the prediction values in descending order
  values = tensorflow::tf$nn$top_k(pred,k=totalCount)
  indices<-values[[1]]
  values<-values[[0]]
  
  #sorting the ground truth values based on the predictions above         
  sortedTrue = k_gather(true, indices)
  
  #getting the ground negative elements (already sorted above)
  negatives = 1 - sortedTrue
  
  #the true positive count per threshold
  TPCurve = k_cumsum(sortedTrue)
  
  #area under the curve
  auc = k_sum(TPCurve * negatives)
  
  #normalizing the result between 0 and 1
  totalCount = k_cast(totalCount, k_floatx())
  positiveCount = k_sum(true)
  negativeCount = totalCount - positiveCount
  totalArea = positiveCount * negativeCount
  return  (auc / totalArea)
}
keras_model_to_estimator <- function(
  keras_model = NULL, keras_model_path = NULL, custom_objects = NULL,
  model_dir = NULL, config = NULL) {
  
  if (is.null(keras_model) && is.null(keras_model_path))
    stop("Either keras_model or keras_model_path needs to be provided.")
  
  if (!is.null(keras_model_path)) {
    if (!is.null(keras_model))
      stop("Please specity either keras_model or keras_model_path but not both.")
    if (grepl("^(gs://|storage\\.googleapis\\.com)", keras_model_path))
      stop("'keras_model_path' is not a local path. Please copy the model locally first.")
    keras_model <- tf$keras$models$load_model(keras_model_path)
  }
  
  tryCatch(reticulate::py_get_attr(keras_model, "optimizer"),
           error = function(e) stop(
             "Given keras model has not been compiled yet. Please compile first 
             before creating the estimator.")
  )
  
  args <- as.list(environment(), all = TRUE)
  
  estimator <- py_suppress_warnings(
    tf$keras$estimator$model_to_estimator(
      keras_model = keras_model,
      keras_model_path = keras_model_path,
      custom_objects = custom_objects,
      model_dir = model_dir,
      config = config
    ))
  
  new_tf_keras_estimator(estimator, args = args)
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
sudcatData_0v2 <- trimmedFreesurfer[trimmedFreesurfer$sud_cat!='1',]
sudcatData_0v2[which(sudcatData_0v2[,3]=='2'),3] <- '1'

splitData_0v2 <- crossValidate(sudcatData_0v2,seed = seed, trainPercentage = 0.9)
names(splitData_0v2) <- c("train","test")

#Import Libraries and Properly format data
library(keras)
library(tensorflow)

kerasTrainData_0v2 <- splitData_0v2$train[,4:ncol(splitData_0v2$train)]
kerasTrainData_0v2 <- cbind(kerasTrainData_0v2,splitData_0v2$train[,3])
colnames(kerasTrainData_0v2)[221] <- "sud_cat"

kerasTestData_0v2 <- splitData_0v2$test[,4:ncol(splitData_0v2$test)]
kerasTestData_0v2 <- cbind(kerasTestData_0v2,splitData_0v2$test[,3])
colnames(kerasTestData_0v2)[221] <- "sud_cat"

#Split Data Variables
keras_training_0v2 <- kerasTrainData_0v2[,1:220]
keras_test_0v2 <- kerasTestData_0v2[,1:220]

#Split results (sud_cat)
keras_trainingTarget_0v2 <- kerasTrainData_0v2[,221]
keras_testTarget_0v2 <- kerasTestData_0v2[,221]

keras_trainLabels_0v2 <- to_categorical(keras_trainingTarget_0v2)
keras_testLabels_0v2 <- to_categorical(keras_testTarget_0v2)

#Do the same but for normalized data
kerasTrainDataNormalized_0v2 <- as.data.frame(lapply(kerasTrainData_0v2[,1:220],normalize))
kerasTrainDataNormalized_0v2 <- cbind(kerasTrainDataNormalized_0v2,splitData_0v2$train[,3])
colnames(kerasTrainDataNormalized_0v2)[221] <- "sud_cat"

kerasTestDataNormalized_0v2 <- c()
for (x in 1:ncol(kerasTrainData_0v2[1:220])){
  kerasTestDataNormalized_0v2 <- cbind(kerasTestDataNormalized_0v2,standardize(kerasTestData_0v2[,x],y=kerasTrainData_0v2[,x]))
}
kerasTestDataNormalized_0v2 <- as.data.frame(kerasTestDataNormalized_0v2)
kerasTestDataNormalized_0v2 <- cbind(kerasTestDataNormalized_0v2,splitData_0v2$test[,3])
colnames(kerasTestDataNormalized_0v2)[221] <- "sud_cat"

#Split Data Variables
keras_training_normal_0v2 <- kerasTrainDataNormalized_0v2[,1:220]
keras_test_normal_0v2 <- kerasTestDataNormalized_0v2[,1:220]

#Split results (sud_cat)
keras_trainingTarget_normal_0v2 <- kerasTrainDataNormalized_0v2[,221]
keras_testTarget_normal_0v2 <- kerasTestDataNormalized_0v2[,221]

keras_trainLabels_normal_0v2 <- to_categorical(keras_trainingTarget_normal_0v2)
keras_testLabels_normal_0v2 <- to_categorical(keras_testTarget_normal_0v2)

keras_training_0v2 <- data.matrix(keras_training_0v2)
keras_test_0v2 <- data.matrix(keras_test_0v2)
keras_training_normal_0v2 <- data.matrix(keras_training_normal_0v2)
keras_test_normal_0v2 <- data.matrix(keras_test_normal_0v2)

#Multi-Layer Perceptron using keras and tensorflow on normalized Data
rm(modelMLPN_0v2)
modelMLPN_0v2<- keras_model_sequential()

modelMLPN_0v2%>% 
  layer_dense(units = 220, activation = 'relu',kernel_regularizer = regularizer_l1_l2(l1 = 0.001, l2 = 0.001), input_shape = c(220)) %>% 
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 2, activation = 'sigmoid')

modelMLPN_0v2%>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'nadam',
  metrics = 'accuracy'
)


modelMLPN_0v2%>% fit(
  keras_training_normal_0v2,
  keras_trainLabels_normal_0v2,
  epochs = 100, 
  batch_size = 200, 
  validation_split = 0.5,
)



# Predict the classes for the test data
predictionsMLPN_0v2 <- modelMLPN_0v2%>% predict_classes(keras_test_normal_0v2, batch_size = 48)
# Confusion matrix
confusionMatrixMLPN_0v2 <- table(keras_testTarget_normal_0v2, predictionsMLPN_0v2)

correctMLPN_0v2 <- 0
incorrectMLPN_0v2 <- 0
for(row in 1:nrow(confusionMatrixMLPN_0v2)) {
  for(col in 1:ncol(confusionMatrixMLPN_0v2)) {
    if(rownames(confusionMatrixMLPN_0v2)[row] == colnames(confusionMatrixMLPN_0v2)[col]){
      correctMLPN_0v2 <- sum(correctMLPN_0v2,confusionMatrixMLPN_0v2[row,col])
    } else {
      incorrectMLPN_0v2 <- sum(incorrectMLPN_0v2,confusionMatrixMLPN_0v2[row,col])
    }
  }
}
cat("The accuracy of the multilayer perceptron is", 100*(correctMLPN_0v2/(correctMLPN_0v2+incorrectMLPN_0v2)),"%")
modelMLPN_0v2 %>% save_model_hdf5("model0v2.h5")
test <- load_model_hdf5("model0v2.h5")
summary(modelMLPN_0v2)
weightArray <- get_weights(modelMLPN_0v2)
weightsLayer1 <- weightArray[[1]]
biasLayer1 <- weightArray[[2]]
weightsLayerNormalization <- weightArray[[3]]
biasLayerNormalization <- weightArray[[4]]
weightsLayerDropout <- weightArray[[5]]
biasLayerDropout <- weightArray[[6]]
weightsLayer2 <- weightArray[[7]]
biasLayer2 <- weightArray[[8]]


#ALL CODE BENEATH THIS POINT IS FOR MY EXPERIMENTATION, DISREGARD WHEN RUNNING THE PROGRAM
library(lime)
library(corrplot)
class(modelMLPN_0v2)

correlationPlot <- cor(sudcatData_0v2[,4:223])
corrplot(correlationPlot, type = 'upper', title = 'Correlation Matrix for Variables')

model_type.keras.models.Sequential <- function(x, ...) {
  "classification"}
predict_model.keras.engine.sequential.Sequential <- function (x, newdata, type, ...) {
  pred <- predict_proba (object = x, x = as.matrix(newdata))
  data.frame (Positive = pred, Negative = 1 - pred) }
train_data_frame <- as.data.frame(keras_training_normal_0v2)
test_data_frame <- as.data.frame(keras_test_normal_0v2)
predict_model (x       = modelMLPN_0v2, 
               newdata = train_data_frame, 
               type    = 'raw') %>%
  tibble::as_tibble()

explainer <- lime::lime (
  x              = train_data_frame, 
  model          = modelMLPN_0v2, 
  bin_continuous = FALSE)

explanation <- lime::explain (
  train_data_frame[1:10, ], 
  explainer    = explainer, 
  n_labels     = 2, 
  n_features   = 220, 
  kernel_width = 0.5)
plot_features(explanation[1:5,]) 
plot_explanations (explanation[1:5,]) 