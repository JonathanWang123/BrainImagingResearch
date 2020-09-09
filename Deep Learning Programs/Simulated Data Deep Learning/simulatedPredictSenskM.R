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
simu.m.sensk <- cbind(simu.m[,1:220],simu.m[,224])
colnames(simu.m.sensk)[221] <- "sensk"

#Split into train and test data
simu.m.sensk.split <- crossValidate(simu.m.sensk,seed = seed, trainPercentage = 0.8)
names(simu.m.sensk.split) <- c("train","test")

simu.m.sensk.train <- simu.m.sensk.split$train[,1:ncol(simu.m.sensk.split$train)]
simu.m.sensk.test <- simu.m.sensk.split$test[,1:ncol(simu.m.sensk.split$test)]

#Normalize Train Data
simu.m.sensk.trainData <- as.data.frame(apply(simu.m.sensk.train[,1:220],2,normalize))

#Normalize test data with respect to train data
simu.m.sensk.testData <- c()
for (x in 1:220){
  simu.m.sensk.testData <- cbind(simu.m.sensk.testData,normalize(simu.m.sensk.test[,x],y=simu.m.sensk.train[,x]))
}
colnames(simu.m.sensk.testData) <- colnames(simu.m.sensk.trainData)
simu.m.sensk.testData <- as.data.frame(simu.m.sensk.testData)


#Format data to be used for training
simu.m.sensk.trainTarget <- (simu.m.sensk.train[,221])
simu.m.sensk.testTarget <- (simu.m.sensk.test[,221])

simu.m.sensk.trainData <- data.matrix(simu.m.sensk.trainData)
simu.m.sensk.trainLabels <- data.matrix(simu.m.sensk.trainTarget)
simu.m.sensk.testData <- data.matrix(simu.m.sensk.testData)
simu.m.sensk.testLabels <- data.matrix(simu.m.sensk.testTarget)

#Stop the training when val_loss doesn't change for 20 epochs
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

#Create Model
model.simu.m.sensk<- keras_model_sequential()

model.simu.m.sensk%>% #Architechture can be tweaked to change model
  layer_dense(units = 220, activation = 'relu', kernel_initializer='he_normal',bias_initializer ='he_normal', input_shape = c(220)) %>% 
  layer_dense(units = 1, activation = 'linear')

model.simu.m.sensk%>% compile(
  loss = 'mean_squared_error', #Can be tweaked to Change model
  optimizer = 'adam', #Can be tweaked to change model 
  metrics = list("mean_absolute_error") #Has no effect on training, just for visualization 
)

#Train Model
model.simu.m.sensk%>% fit(
  simu.m.sensk.trainData,
  simu.m.sensk.trainLabels,
  epochs = 2000,
  validation_split = 0.25, #can be tweaked to change model
  callbacks = list(early_stop),
  verbose = 0
)

#Get Loss and Mean Absolute Error of Model on Testing Data
c(loss.m.sensk, mae.m.sensk) %<-% (model.simu.m.sensk %>% evaluate(simu.m.sensk.testData, simu.m.sensk.testLabels, verbose = 0))

paste0("Mean absolute error on test set: ", sprintf("%.2f", mae.m.sensk))

# Store predictions in a new dataset
predictions.simu.m.sensk <- model.simu.m.sensk%>% predict(simu.m.sensk.testData, batch_size = 2374)

#Next Section is for Variable Importance
#Resource used https://bgreenwell.github.io/pdp/articles/pdp-example-tensorflow.html

#Prediction Function
pred_wrapper <- function(object, newdata) {
  predict(object, x = as.matrix(newdata)) %>%
    as.vector()
}

#Create and Print Variable importance plot
simu.m.sensk.variableImportancePlot <- vip(
  object = model.simu.m.sensk,          # fitted model
  method = "permute",                 # permutation-based VI scores,
  num_features = 10,       # top X features you want to keep track of (but ncol(simu.m.sensk.trainData) for all variables)
  pred_wrapper = pred_wrapper,            # user-defined prediction function
  train = as.data.frame(simu.m.sensk.trainData) ,    # training data
  target = simu.m.sensk.trainLabels,                   # response values used for training
  metric = "rsquared",                # evaluation metric
  progress = "text"                 # request a text-based progress bar
)
print(simu.m.sensk.variableImportancePlot)
print(simu.m.sensk.variableImportancePlot[1])

#Get weights
weights.m.sensk <- get_weights(model.simu.m.sensk)

