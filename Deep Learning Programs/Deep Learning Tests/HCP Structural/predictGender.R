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
library(keras)
library(tensorflow)
library(lime)
library(corrplot)
library(ggplot2)
seed = 123
set.seed(seed)

#Import Data Sets (change filepath to your computer)
freesurfer <- read.csv("D:/Research/Yihong Zhao/Brain Data Sets/freesurfer.csv")
behavioral <- read.csv("D:/Research/Yihong Zhao/Brain Data Sets/behavioraldata.csv")

#trimVars is a matrix of all the variable substrings you want to include
trimVars<-c("Subject","_Thck$","_Area$","_GrayVol$",
            "FS_L_Hippo_Vol","FS_L_Amygdala_Vol","FS_L_AccumbensArea_Vol", "FS_L_VentDC_Vol",
            "FS_L_ThalamusProper_Vol","FS_L_Caudate_Vol","FS_L_Putamen_Vol","FS_L_Pallidum_Vol",
            "FS_R_Hippo_Vol","FS_R_Amygdala_Vol","FS_R_AccumbensArea_Vol", "FS_R_VentDC_Vol",
            "FS_R_ThalamusProper_Vol","FS_R_Caudate_Vol","FS_R_Putamen_Vol","FS_R_Pallidum_Vol")

#Split the dataset by column, merge by Subject, then separate subjects into training and testing data
trimmedFreesurfer <- trimByCol(freesurfer,trimVars)
trimmedBehavioral <- trimByCol(behavioral,c("Subject","Gender"))
trimmedFreesurfer <- merge(trimmedBehavioral,trimmedFreesurfer, by.x = "Subject",by.y = "Subject")
trimmedFreesurfer <- na.omit(trimmedFreesurfer)
gender <- c()
for(x in 1:nrow(trimmedFreesurfer)){
  if((trimmedFreesurfer[x,2]=="M")){
    gender <- rbind(gender, c(0))
  } else {
    gender <- rbind(gender, c(1))
  }
}
trimmedFreesurfer[,2] <- gender
hcp.gender.split <- crossValidate(trimmedFreesurfer,seed = seed, trainPercentage = 0.8)
names(hcp.gender.split) <- c("train","test")

hcp.gender.train <- hcp.gender.split$train[,2:ncol(hcp.gender.split$train)]
hcp.gender.test <- hcp.gender.split$test[,2:ncol(hcp.gender.split$test)]

#Normalize Train Data
hcp.gender.trainData <- as.data.frame(apply(hcp.gender.train[,2:221],2,normalize))

#Normalize test data with respect to train data
hcp.gender.testData <- c()
for (x in 2:221){
  hcp.gender.testData <- cbind(hcp.gender.testData,normalize(hcp.gender.test[,x],y=hcp.gender.train[,x]))
}
colnames(hcp.gender.testData) <- colnames(hcp.gender.trainData)
hcp.gender.testData <- as.data.frame(hcp.gender.testData)

#Format data to be used for training M = 0, F = 1
hcp.gender.trainTarget <- to_categorical(hcp.gender.train[,1])
hcp.gender.testTarget <- to_categorical(hcp.gender.test[,1])

hcp.gender.trainData <- data.matrix(hcp.gender.trainData)
hcp.gender.trainLabels <- data.matrix(hcp.gender.trainTarget)
hcp.gender.testData <- data.matrix(hcp.gender.testData)
hcp.gender.testLabels <- data.matrix(hcp.gender.testTarget)

#Stop the training when val_loss doesn't change for 20 epochs
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

#Create Model
model.hcp.gender<- keras_model_sequential()

model.hcp.gender%>% #Architechture can be tweaked to change model
  layer_dense(units = 220, activation = 'relu', kernel_initializer='he_normal',bias_initializer ='he_normal', input_shape = c(220)) %>% 
  layer_dense(units = 2, activation = 'sigmoid')

model.hcp.gender%>% compile(
  loss = 'binary_crossentropy', #Can be tweaked to Change model
  optimizer = 'nadam', #Can be tweaked to change model 
  metrics = 'accuracy' #Has no effect on training, just for visualization 
)

#Train Model
model.hcp.gender%>% fit(
  hcp.gender.trainData,
  hcp.gender.trainLabels,
  epochs = 1000,
  validation_split = 0.5, #can be tweaked to change model
  callbacks = list(early_stop),
  verbose = 0
)
# Predict the classes for the test data
hcp.gender.predictions <- model.hcp.gender%>% predict_classes(hcp.gender.testData, batch_size = 48)
# Confusion matrix
hcp.gender.confusion <- table(hcp.gender.test[,1], hcp.gender.predictions)

hcp.gender.correct <- 0
hcp.gender.incorrect <- 0
for(row in 1:nrow(hcp.gender.confusion)) {
  for(col in 1:ncol(hcp.gender.confusion)) {
    if(rownames(hcp.gender.confusion)[row] == colnames(hcp.gender.confusion)[col]){
      hcp.gender.correct <- sum(hcp.gender.correct,hcp.gender.confusion[row,col])
    } else {
      hcp.gender.incorrect <- sum(hcp.gender.incorrect,hcp.gender.confusion[row,col])
    }
  }
}
cat("The accuracy of the multilayer perceptron is", 100*(hcp.gender.correct/(hcp.gender.correct+hcp.gender.incorrect)),"%")
#Next Section is for Variable Importance

model_type.keras.models.Sequential <- function(x, ...) {
  "classification"}
predict_model.keras.engine.sequential.Sequential <- function (x, newdata, type, ...) {
  pred <- predict_proba (object = x, x = as.matrix(newdata))
  data.frame (Positive = pred, Negative = 1 - pred) }
train_data_frame <- as.data.frame(hcp.gender.trainData)
test_data_frame <- as.data.frame(hcp.gender.testData)

explainer <- lime::lime (
  x              = train_data_frame, 
  model          = model.hcp.gender, 
  bin_continuous = FALSE)

hcp.gender.explanation <- lime::explain (
  train_data_frame[1:10, ], 
  explainer    = explainer, 
  n_labels     = 2, 
  n_features   = 220, 
  kernel_width = 0.5)
plot_features(hcp.gender.explanation[1:5,]) 
plot_explanations (hcp.gender.explanation[1:5,]) 

#Get Weights
weights.hcp.gender <- get_weights(model.hcp.gender)


