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
library(vip)
library(pdp)
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
trimmedBehavioral <- trimByCol(behavioral,c("Subject","AngHostil_Unadj"))
trimmedFreesurfer <- merge(trimmedBehavioral,trimmedFreesurfer, by.x = "Subject",by.y = "Subject")
trimmedFreesurfer <- na.omit(trimmedFreesurfer)


NIH.AngHostility.split <- crossValidate(trimmedFreesurfer,seed = seed, trainPercentage = 0.8)
names(NIH.AngHostility.split) <- c("train","test")

NIH.AngHostility.train <- NIH.AngHostility.split$train[,2:ncol(NIH.AngHostility.split$train)]
NIH.AngHostility.test <- NIH.AngHostility.split$test[,2:ncol(NIH.AngHostility.split$test)]

#Normalize Train Data
NIH.AngHostility.trainData <- as.data.frame(apply(NIH.AngHostility.train[,2:221],2,normalize))

#Normalize test data with respect to train data
NIH.AngHostility.testData <- c()
for (x in 2:221){
  NIH.AngHostility.testData <- cbind(NIH.AngHostility.testData,normalize(NIH.AngHostility.test[,x],y=NIH.AngHostility.train[,x]))
}
colnames(NIH.AngHostility.testData) <- colnames(NIH.AngHostility.trainData)
NIH.AngHostility.testData <- as.data.frame(NIH.AngHostility.testData)

#Format data to be used for training
NIH.AngHostility.trainTarget <- (NIH.AngHostility.train[,1])
NIH.AngHostility.testTarget <- (NIH.AngHostility.test[,1])

NIH.AngHostility.trainData <- data.matrix(NIH.AngHostility.trainData)
NIH.AngHostility.trainLabels <- data.matrix(NIH.AngHostility.trainTarget)
NIH.AngHostility.testData <- data.matrix(NIH.AngHostility.testData)
NIH.AngHostility.testLabels <- data.matrix(NIH.AngHostility.testTarget)

#Stop the training when val_loss doesn't change for 20 epochs
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

#Create Model
model.NIH.AngHostility<- keras_model_sequential()

model.NIH.AngHostility%>% #Architechture can be tweaked to change model
  layer_dense(units = 220, activation = 'relu', kernel_initializer='he_normal',bias_initializer ='he_normal', input_shape = c(220)) %>% 
  layer_dense(units = 1, activation = 'linear')

model.NIH.AngHostility%>% compile(
  loss = 'mean_squared_error', #Can be tweaked to Change model
  optimizer = 'adam', #Can be tweaked to change model 
  metrics = list("mean_absolute_error") #Has no effect on training, just for visualization 
)

#Train Model
model.NIH.AngHostility%>% fit(
  NIH.AngHostility.trainData,
  NIH.AngHostility.trainLabels,
  epochs = 500,
  validation_split = 0.5, #can be tweaked to change model
  callbacks = list(early_stop),
  verbose = 0
)
c(loss.NIH.AngHostility, mae.NIH.AngHostility) %<-% (model.NIH.AngHostility %>% evaluate(NIH.AngHostility.testData, NIH.AngHostility.testLabels, verbose = 0))

paste0("Mean absolute error on test set: ", sprintf("%.2f", mae.NIH.AngHostility))

predictions.NIH.AngHostility <- model.NIH.AngHostility%>% predict(NIH.AngHostility.testData, batch_size = 2374)
#Next Section is for Variable Importance
#Resource used https://bgreenwell.github.io/pdp/articles/pdp-example-tensorflow.html

#Prediction Function
pred_wrapper <- function(object, newdata) {
  predict(object, x = as.matrix(newdata)) %>%
    as.vector()
}

#Create and Print Variable importance plot
NIH.AngHostility.variableImportancePlot <- vip(
  object = model.NIH.AngHostility,          # fitted model
  method = "permute",                 # permutation-based VI scores,
  num_features = 10,       # top X features you want to keep track of (but ncol(NIH.AngHostility.trainData) for all variables)
  pred_wrapper = pred_wrapper,            # user-defined prediction function
  train = as.data.frame(NIH.AngHostility.trainData) ,    # training data
  target = NIH.AngHostility.trainLabels,                   # response values used for training
  metric = "rsquared",                # evaluation metric
  progress = "text"                 # request a text-based progress bar
)
print(NIH.AngHostility.variableImportancePlot)
print(NIH.AngHostility.variableImportancePlot[1])

#Get weights 
weights.NIH.AngHostility <- get_weights(model.NIH.AngHostility)


