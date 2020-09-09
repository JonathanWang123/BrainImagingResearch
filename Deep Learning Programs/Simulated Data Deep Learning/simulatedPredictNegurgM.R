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
#Import Libraries
library(keras)
library(tensorflow)
library(vip)
library(pdp)
library(ggplot2)

#Set random seed for semi-reproducibility (TensorFlow on the backend has its own randomization so setting a seed doesnt effect it)
seed = 12345
set.seed(seed)

#Import Database
simu.m <- readRDS("D:/Research/Yihong Zhao/Brain Data Sets/simu.m.rds")

#Separate out variables used to predict and variable to be predicted
simu.m.negurg <- cbind(simu.m[,1:220],simu.m[,223])
colnames(simu.m.negurg)[221] <- "negurg"

#Split into train and test data
simu.m.negurg.split <- crossValidate(simu.m.negurg,seed = seed, trainPercentage = 0.8)
names(simu.m.negurg.split) <- c("train","test")

simu.m.negurg.train <- simu.m.negurg.split$train[,1:ncol(simu.m.negurg.split$train)]
simu.m.negurg.test <- simu.m.negurg.split$test[,1:ncol(simu.m.negurg.split$test)]

#Normalize Train Data
simu.m.negurg.trainData <- as.data.frame(apply(simu.m.negurg.train[,1:220],2,normalize))

#Normalize test data with respect to train data
simu.m.negurg.testData <- c()
for (x in 1:220){
  simu.m.negurg.testData <- cbind(simu.m.negurg.testData,normalize(simu.m.negurg.test[,x],y=simu.m.negurg.train[,x]))
}
colnames(simu.m.negurg.testData) <- colnames(simu.m.negurg.trainData)
simu.m.negurg.testData <- as.data.frame(simu.m.negurg.testData)


#Format data to be used for training
simu.m.negurg.trainTarget <- (simu.m.negurg.train[,221])
simu.m.negurg.testTarget <- (simu.m.negurg.test[,221])

simu.m.negurg.trainData <- data.matrix(simu.m.negurg.trainData)
simu.m.negurg.trainLabels <- data.matrix(simu.m.negurg.trainTarget)
simu.m.negurg.testData <- data.matrix(simu.m.negurg.testData)
simu.m.negurg.testLabels <- data.matrix(simu.m.negurg.testTarget)

#Stop the training when val_loss doesn't change for 20 epochs
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

#Create Model
model.simu.m.negurg<- keras_model_sequential()

model.simu.m.negurg%>% #Architechture can be tweaked to change model
  layer_dense(units = 220, activation = 'relu', kernel_initializer='he_normal',bias_initializer ='he_normal', input_shape = c(220)) %>% 
  layer_dense(units = 1, activation = 'linear')

model.simu.m.negurg%>% compile(
  loss = 'mean_squared_error', #Can be tweaked to Change model
  optimizer = 'adam', #Can be tweaked to change model 
  metrics = list("mean_absolute_error") #Has no effect on training, just for visualization 
)

#Train Model
model.simu.m.negurg%>% fit(
  simu.m.negurg.trainData,
  simu.m.negurg.trainLabels,
  epochs = 2000,
  validation_split = 0.25, #can be tweaked to change model
  callbacks = list(early_stop),
  verbose = 0
)

#Get Loss and Mean Absolute Error of Model on Testing Data
c(loss.m.negurg, mae.m.negurg) %<-% (model.simu.m.negurg %>% evaluate(simu.m.negurg.testData, simu.m.negurg.testLabels, verbose = 0))

paste0("Mean absolute error on test set: ", sprintf("%.2f", mae.m.negurg))

# Store predictions in a new dataset
predictions.simu.m.negurg <- model.simu.m.negurg%>% predict(simu.m.negurg.testData, batch_size = 2374)

#Next Section is for Variable Importance
#Resource used https://bgreenwell.github.io/pdp/articles/pdp-example-tensorflow.html

#Prediction Function
pred_wrapper <- function(object, newdata) {
  predict(object, x = as.matrix(newdata)) %>%
    as.vector()
}

#Create and Print Variable importance plot
simu.m.negurg.variableImportancePlot <- vip(
  object = model.simu.m.negurg,          # fitted model
  method = "permute",                 # permutation-based VI scores,
  num_features = 10,       # top X features you want to keep track of (but ncol(simu.m.negurg.trainData) for all variables)
  pred_wrapper = pred_wrapper,            # user-defined prediction function
  train = as.data.frame(simu.m.negurg.trainData) ,    # training data
  target = simu.m.negurg.trainLabels,                   # response values used for training
  metric = "rsquared",                # evaluation metric
  progress = "text"                 # request a text-based progress bar
)
print(simu.m.negurg.variableImportancePlot)
print(simu.m.negurg.variableImportancePlot[1])

#Get weights
weights.m.negurg <- get_weights(model.simu.m.negurg)

