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
simu.f <- readRDS("D:/Research/Yihong Zhao/Brain Data Sets/simu.f.rds")

#Separate out variables used to predict and variable to be predicted
simu.hyperParam <- cbind(simu.f[,1:220],simu.f[,221])
colnames(simu.hyperParam)[221] <- "age"

#Split into train and test data
simu.hyperParam.split <- crossValidate(simu.hyperParam,seed = seed, trainPercentage = 0.8)
names(simu.hyperParam.split) <- c("train","test")

simu.hyperParam.train <- simu.hyperParam.split$train[,1:ncol(simu.hyperParam.split$train)]
simu.hyperParam.test <- simu.hyperParam.split$test[,1:ncol(simu.hyperParam.split$test)]

#Normalize Train Data
simu.hyperParam.trainData <- as.data.frame(apply(simu.hyperParam.train[,1:220],2,normalize))

#Normalize test data with respect to train data
simu.hyperParam.testData <- c()
for (x in 1:220){
  simu.hyperParam.testData <- cbind(simu.hyperParam.testData,normalize(simu.hyperParam.test[,x],y=simu.hyperParam.train[,x]))
}
colnames(simu.hyperParam.testData) <- colnames(simu.hyperParam.trainData)
simu.hyperParam.testData <- as.data.frame(simu.hyperParam.testData)


#Format data to be used for training
simu.hyperParam.trainTarget <- (simu.hyperParam.train[,221])
simu.hyperParam.testTarget <- (simu.hyperParam.test[,221])

simu.hyperParam.trainData <- data.matrix(simu.hyperParam.trainData)
simu.hyperParam.trainLabels <- data.matrix(simu.hyperParam.trainTarget)
simu.hyperParam.testData <- data.matrix(simu.hyperParam.testData)
simu.hyperParam.testLabels <- data.matrix(simu.hyperParam.testTarget)

#Stop the training when val_loss doesn't change for 20 epochs
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

#Create Model

#numLayers <- c(1,3,5)
#numUnits <- c(220, 440)
#valSplit <- c(0.25,0.5,0.75)
activationFun <- c('relu','sigmoid')
initializer <- c('he_normal', 'zeros','glorot_normal')
optimizer <- c('adam','nadam','sgd')
lossFun <- c('mean_squared_error','mean_absolute_error','mean_squared_logarithmic_error')
tuningPermutations <- expand.grid(activationFun,initializer,optimizer,lossFun)

results <- c()
for(x in 1:nrow(tuningPermutations)){
  model.simu.hyperParam<- keras_model_sequential()
  model.simu.hyperParam%>% #Architechture can be tweaked to change model
    layer_dense(units = 220, activation = tuningPermutations[x,1], kernel_initializer=tuningPermutations[x,2],bias_initializer =tuningPermutations[x,2], input_shape = c(220)) %>% 
    layer_dense(units = 1, activation = 'linear')
  
  model.simu.hyperParam%>% compile(
    loss = tuningPermutations[x,4], #Can be tweaked to Change model
    optimizer = tuningPermutations[x,3], #Can be tweaked to change model 
    metrics = list("mean_absolute_error") #Has no effect on training, just for visualization 
  )
  
  #Train Model
  model.simu.hyperParam%>% fit(
    simu.hyperParam.trainData,
    simu.hyperParam.trainLabels,
    epochs = 2000,
    validation_split = 0.75, #can be tweaked to change model
    callbacks = list(early_stop),
    verbose = 0
  )
  c(loss.hyperParam, mae.hyperParam) %<-% (model.simu.hyperParam %>% evaluate(simu.hyperParam.testData, simu.hyperParam.testLabels, verbose = 0))
  results <- rbind(results,c(mae.hyperParam))
}

# Store predictions in a new dataset
predictions.simu.hyperParam <- model.simu.hyperParam%>% predict(simu.hyperParam.testData, batch_size = 2374)

#Next Section is for Variable Importance
#Resource used https://bgreenwell.github.io/pdp/articles/pdp-example-tensorflow.html

#Prediction Function
pred_wrapper <- function(object, newdata) {
  predict(object, x = as.matrix(newdata)) %>%
    as.vector()
}

#Create and Print Variable importance plot
simu.hyperParam.variableImportancePlot <- vip(
  object = model.simu.hyperParam,          # fitted model
  method = "permute",                 # permutation-based VI scores,
  num_features = 10,       # top X features you want to keep track of (but ncol(simu.hyperParam.trainData) for all variables)
  pred_wrapper = pred_wrapper,            # user-defined prediction function
  train = as.data.frame(simu.hyperParam.trainData) ,    # training data
  target = simu.hyperParam.trainLabels,                   # response values used for training
  metric = "rsquared",                # evaluation metric
  progress = "text"                 # request a text-based progress bar
)
print(simu.hyperParam.variableImportancePlot)
print(simu.hyperParam.variableImportancePlot[1])

#Get weights 
weights.hyperParam <- get_weights(model.simu.hyperParam)


