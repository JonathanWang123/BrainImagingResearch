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

dt2 <- readRDS("D:/Research/Yihong Zhao/Brain Data Sets/dt2.subid.rds")

#trimVars is a matrix of all the variable substrings you want to include
trimVars<-c("Subject","_Thck$","_Area$","_GrayVol$",
            "FS_L_Hippo_Vol","FS_L_Amygdala_Vol","FS_L_AccumbensArea_Vol", "FS_L_VentDC_Vol",
            "FS_L_ThalamusProper_Vol","FS_L_Caudate_Vol","FS_L_Putamen_Vol","FS_L_Pallidum_Vol",
            "FS_R_Hippo_Vol","FS_R_Amygdala_Vol","FS_R_AccumbensArea_Vol", "FS_R_VentDC_Vol",
            "FS_R_ThalamusProper_Vol","FS_R_Caudate_Vol","FS_R_Putamen_Vol","FS_R_Pallidum_Vol",
            "FS_L_Cerebellum_Cort_Vol","FS_R_Cerebellum_Cort_Vol","FS_BrainStem_Vol")

#Split the dataset by column, merge by Subject, then separate subjects into training and testing data

dt2su18 <- cbind(dt2[,1],dt2[,7])
colnames(dt2su18) <- c("Subject","su.17")

trimmedFreesurfer <- trimByCol(freesurfer,trimVars)
trimmedFreesurfer <- merge(dt2su18,trimmedFreesurfer, by.x = "Subject",by.y = "Subject")
trimmedFreesurfer <- na.omit(trimmedFreesurfer)

hcp.su18.dt2.split <- crossValidate(trimmedFreesurfer,seed = seed, trainPercentage = 1)
names(hcp.su18.dt2.split) <- c("train","test")

hcp.su18.dt2.train <- hcp.su18.dt2.split$train[,2:ncol(hcp.su18.dt2.split$train)]
hcp.su18.dt2.test <- hcp.su18.dt2.split$test[,2:ncol(hcp.su18.dt2.split$test)]

#Normalize Train Data
hcp.su18.dt2.trainData <- as.data.frame(apply(hcp.su18.dt2.train[,2:224],2,normalize))

#Normalize test data with respect to train data
hcp.su18.dt2.testData <- c()
for (x in 2:224){
  hcp.su18.dt2.testData <- cbind(hcp.su18.dt2.testData,normalize(hcp.su18.dt2.test[,x],y=hcp.su18.dt2.train[,x]))
}
colnames(hcp.su18.dt2.testData) <- colnames(hcp.su18.dt2.trainData)
hcp.su18.dt2.testData <- as.data.frame(hcp.su18.dt2.testData)

#Format data to be used for training
hcp.su18.dt2.trainTarget <- to_categorical(hcp.su18.dt2.train[,1])
hcp.su18.dt2.testTarget <- to_categorical(hcp.su18.dt2.test[,1])

hcp.su18.dt2.trainData <- data.matrix(hcp.su18.dt2.trainData)
hcp.su18.dt2.trainLabels <- data.matrix(hcp.su18.dt2.trainTarget)
hcp.su18.dt2.testData <- data.matrix(hcp.su18.dt2.testData)
hcp.su18.dt2.testLabels <- data.matrix(hcp.su18.dt2.testTarget)
#Stop the training when val_loss doesn't change for 20 epochs
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 250)

#Create Model
model.hcp.su18.dt2<- keras_model_sequential()

model.hcp.su18.dt2%>% #Architechture can be tweaked to change model
  layer_dense(units = 223, activation = 'relu', kernel_initializer='he_normal',bias_initializer ='he_normal', input_shape = c(223)) %>% 
  layer_dense(units = 2, activation = 'sigmoid')

model.hcp.su18.dt2%>% compile(
  loss = 'binary_crossentropy', #Can be tweaked to Change model
  optimizer = 'adam', #Can be tweaked to change model 
  metrics = 'accuracy' #Has no effect on training, just for visualization 
)

#Train Model
model.hcp.su18.dt2%>% fit(
  hcp.su18.dt2.trainData,
  hcp.su18.dt2.trainLabels,
  epochs = 250,
  validation_split = 0.5, #can be tweaked to change model
  batch_size = 10,
  callbacks = list(early_stop),
  verbose = FALSE
)
# Predict the classes for the test data
hcp.su18.dt2.predictions <- model.hcp.su18.dt2%>% predict_classes(hcp.su18.dt2.trainData, batch_size = 48)
# Confusion matrix
hcp.su18.dt2.confusion <- table(hcp.su18.dt2.train[,1], hcp.su18.dt2.predictions)

hcp.su18.dt2.correct <- 0
hcp.su18.dt2.incorrect <- 0
for(row in 1:nrow(hcp.su18.dt2.confusion)) {
  for(col in 1:ncol(hcp.su18.dt2.confusion)) {
    if(rownames(hcp.su18.dt2.confusion)[row] == colnames(hcp.su18.dt2.confusion)[col]){
      hcp.su18.dt2.correct <- sum(hcp.su18.dt2.correct,hcp.su18.dt2.confusion[row,col])
    } else {
      hcp.su18.dt2.incorrect <- sum(hcp.su18.dt2.incorrect,hcp.su18.dt2.confusion[row,col])
    }
  }
}
cat("The accuracy of the multilayer perceptron is", 100*(hcp.su18.dt2.correct/(hcp.su18.dt2.correct+hcp.su18.dt2.incorrect)),"%")
#Next Section is for Variable Importance

model_type.keras.models.Sequential <- function(x, ...) {
  "classification"}
predict_model.keras.engine.sequential.Sequential <- function (x, newdata, type, ...) {
  pred <- predict_proba (object = x, x = as.matrix(newdata))
  data.frame (Positive = pred, Negative = 1 - pred) }
train_data_frame <- as.data.frame(hcp.su18.dt2.trainData)
test_data_frame <- as.data.frame(hcp.su18.dt2.testData)
predict_model (x       = model.hcp.su18.dt2, 
               newdata = train_data_frame, 
               type    = 'raw') %>%
  tibble::as_tibble()

explainer <- lime::lime (
  x              = train_data_frame, 
  model          = model.hcp.su18.dt2, 
  bin_continuous = FALSE)

hcp.su18.dt2.explanation <- lime::explain (
  train_data_frame[1:10, ], 
  explainer    = explainer, 
  n_labels     = 2, 
  n_features   = 223, 
  kernel_width = 0.5)
plot_features(hcp.su18.dt2.explanation[1:50,]) 
plot_explanations (hcp.su18.dt2.explanation[1:50,]) 

#Get Weights
weights.hcp.su18.dt2 <- get_weights(model.hcp.su18.dt2)


