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
simu.f.sensk <- cbind(simu.f[,1:220],simu.f[,224])
colnames(simu.f.sensk)[221] <- "sensk"

#Split into train and test data
simu.f.sensk.split <- crossValidate(simu.f.sensk,seed = seed, trainPercentage = 0.8)
names(simu.f.sensk.split) <- c("train","test")

simu.f.sensk.train <- simu.f.sensk.split$train[,1:ncol(simu.f.sensk.split$train)]
simu.f.sensk.test <- simu.f.sensk.split$test[,1:ncol(simu.f.sensk.split$test)]

#Normalize Train Data
simu.f.sensk.trainData <- as.data.frame(apply(simu.f.sensk.train[,1:220],2,normalize))

#Normalize test data with respect to train data
simu.f.sensk.testData <- c()
for (x in 1:220){
  simu.f.sensk.testData <- cbind(simu.f.sensk.testData,normalize(simu.f.sensk.test[,x],y=simu.f.sensk.train[,x]))
}
colnames(simu.f.sensk.testData) <- colnames(simu.f.sensk.trainData)
simu.f.sensk.testData <- as.data.frame(simu.f.sensk.testData)


#Format data to be used for training
simu.f.sensk.trainTarget <- (simu.f.sensk.train[,221])
simu.f.sensk.testTarget <- (simu.f.sensk.test[,221])

simu.f.sensk.trainData <- data.matrix(simu.f.sensk.trainData)
simu.f.sensk.trainLabels <- data.matrix(simu.f.sensk.trainTarget)
simu.f.sensk.testData <- data.matrix(simu.f.sensk.testData)
simu.f.sensk.testLabels <- data.matrix(simu.f.sensk.testTarget)

#Stop the training when val_loss doesn't change for 20 epochs
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

#Create Model
model.simu.f.sensk<- keras_model_sequential()

model.simu.f.sensk%>% #Architechture can be tweaked to change model
  layer_dense(units = 220, activation = 'relu', kernel_initializer='he_normal',bias_initializer ='he_normal', input_shape = c(220)) %>% 
  layer_dense(units = 1, activation = 'linear')

model.simu.f.sensk%>% compile(
  loss = 'mean_squared_error', #Can be tweaked to Change model
  optimizer = 'adam', #Can be tweaked to change model 
  metrics = list("mean_absolute_error") #Has no effect on training, just for visualization 
)

#Train Model
model.simu.f.sensk%>% fit(
  simu.f.sensk.trainData,
  simu.f.sensk.trainLabels,
  epochs = 2000,
  validation_split = 0.25, #can be tweaked to change model
  callbacks = list(early_stop),
  verbose = 0
)

#Get Loss and Mean Absolute Error of Model on Testing Data
c(loss.f.sensk, mae.f.sensk) %<-% (model.simu.f.sensk %>% evaluate(simu.f.sensk.testData, simu.f.sensk.testLabels, verbose = 0))

paste0("Mean absolute error on test set: ", sprintf("%.2f", mae.f.sensk))

# Store predictions in a new dataset
predictions.simu.f.sensk <- model.simu.f.sensk%>% predict(simu.f.sensk.testData, batch_size = 2374)

#Next Section is for Variable Importance
#Resource used https://bgreenwell.github.io/pdp/articles/pdp-example-tensorflow.html

#Prediction Function
pred_wrapper <- function(object, newdata) {
  predict(object, x = as.matrix(newdata)) %>%
    as.vector()
}

#Create and Print Variable importance plot
simu.f.sensk.variableImportancePlot <- vip(
  object = model.simu.f.sensk,          # fitted model
  method = "permute",                 # permutation-based VI scores,
  num_features = 10,       # top X features you want to keep track of (but ncol(simu.f.sensk.trainData) for all variables)
  pred_wrapper = pred_wrapper,            # user-defined prediction function
  train = as.data.frame(simu.f.sensk.trainData) ,    # training data
  target = simu.f.sensk.trainLabels,                   # response values used for training
  metric = "rsquared",                # evaluation metric
  progress = "text"                 # request a text-based progress bar
)
print(simu.f.sensk.variableImportancePlot)
print(simu.f.sensk.variableImportancePlot[1])

#Get weights
weights.f.sensk <- get_weights(model.simu.f.sensk)

