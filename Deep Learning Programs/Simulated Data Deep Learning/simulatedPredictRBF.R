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
simu.f.rb <- cbind(simu.f[,1:220],simu.f[,222])
colnames(simu.f.rb)[221] <- "rb"

#Split into train and test data
simu.f.rb.split <- crossValidate(simu.f.rb,seed = seed, trainPercentage = 0.8)
names(simu.f.rb.split) <- c("train","test")

simu.f.rb.train <- simu.f.rb.split$train[,1:ncol(simu.f.rb.split$train)]
simu.f.rb.test <- simu.f.rb.split$test[,1:ncol(simu.f.rb.split$test)]

#Normalize Train Data
simu.f.rb.trainData <- as.data.frame(apply(simu.f.rb.train[,1:220],2,normalize))

#Normalize test data with respect to train data
simu.f.rb.testData <- c()
for (x in 1:220){
  simu.f.rb.testData <- cbind(simu.f.rb.testData,normalize(simu.f.rb.test[,x],y=simu.f.rb.train[,x]))
}
colnames(simu.f.rb.testData) <- colnames(simu.f.rb.trainData)
simu.f.rb.testData <- as.data.frame(simu.f.rb.testData)


#Format data to be used for training
simu.f.rb.trainTarget <- (simu.f.rb.train[,221])
simu.f.rb.testTarget <- (simu.f.rb.test[,221])

simu.f.rb.trainData <- data.matrix(simu.f.rb.trainData)
simu.f.rb.trainLabels <- data.matrix(simu.f.rb.trainTarget)
simu.f.rb.testData <- data.matrix(simu.f.rb.testData)
simu.f.rb.testLabels <- data.matrix(simu.f.rb.testTarget)

#Stop the training when val_loss doesn't change for 20 epochs
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

#Create Model
model.simu.f.rb<- keras_model_sequential()

model.simu.f.rb%>% #Architechture can be tweaked to change model
  layer_dense(units = 220, activation = 'relu', kernel_initializer='he_normal',bias_initializer ='he_normal', input_shape = c(220)) %>% 
  layer_dense(units = 1, activation = 'linear')

model.simu.f.rb%>% compile(
  loss = 'mean_squared_error', #Can be tweaked to Change model
  optimizer = 'adam', #Can be tweaked to change model 
  metrics = list("mean_absolute_error") #Has no effect on training, just for visualization 
)

#Train Model
model.simu.f.rb%>% fit(
  simu.f.rb.trainData,
  simu.f.rb.trainLabels,
  epochs = 2000,
  validation_split = 0.25, #can be tweaked to change model
  callbacks = list(early_stop),
  verbose = 0
)

#Get Loss and Mean Absolute Error of Model on Testing Data
c(loss.f.rb, mae.f.rb) %<-% (model.simu.f.rb %>% evaluate(simu.f.rb.testData, simu.f.rb.testLabels, verbose = 0))

paste0("Mean absolute error on test set: ", sprintf("%.2f", mae.f.rb))

# Store predictions in a new dataset
predictions.simu.f.rb <- model.simu.f.rb%>% predict(simu.f.rb.testData, batch_size = 2374)

#Next Section is for Variable Importance
#Resource used https://bgreenwell.github.io/pdp/articles/pdp-example-tensorflow.html

#Prediction Function
pred_wrapper <- function(object, newdata) {
  predict(object, x = as.matrix(newdata)) %>%
    as.vector()
}

#Create and Print Variable importance plot
simu.f.rb.variableImportancePlot <- vip(
  object = model.simu.f.rb,          # fitted model
  method = "permute",                 # permutation-based VI scores,
  num_features = 10,       # top X features you want to keep track of (but ncol(simu.f.rb.trainData) for all variables)
  pred_wrapper = pred_wrapper,            # user-defined prediction function
  train = as.data.frame(simu.f.rb.trainData) ,    # training data
  target = simu.f.rb.trainLabels,                   # response values used for training
  metric = "rsquared",                # evaluation metric
  progress = "text"                 # request a text-based progress bar
)
print(simu.f.rb.variableImportancePlot)
print(simu.f.rb.variableImportancePlot[1])

#Get weights
weights.f.rb <- get_weights(model.simu.f.rb)

