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
crossValidate<- function(x,trainPercentnegurg=0.8,by = "row", seed=1){ #Randomely split matrix into train and test data
  set.seed(seed)
  if(by=="row"){
    sample_size <- floor(trainPercentnegurg*nrow(x))
    picked <- sample(seq_len(nrow(x)),size = sample_size)
    
    train <- x[picked,]
    test <- x[-picked,]
    returnVal <- list(train,test)
    return(returnVal)
  } else if(by=="col"){
    sample_size <- floor(trainPercentnegurg*ncol(x))
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
simu.f.2 <- readRDS("D:/Research/Yihong Zhao/Brain Data Sets/Simulated Data 2/simu.f.2.rds")

#Separate out variables used to predict and variable to be predicted
simu.f.2.negurg <- cbind(simu.f.2[,1:220],simu.f.2[,223])
colnames(simu.f.2.negurg)[221] <- "negurg"

#Split into train and test data
simu.f.2.negurg.split <- crossValidate(simu.f.2.negurg,seed = seed, trainPercentnegurg = 0.8)
names(simu.f.2.negurg.split) <- c("train","test")

simu.f.2.negurg.train <- simu.f.2.negurg.split$train[,1:ncol(simu.f.2.negurg.split$train)]
simu.f.2.negurg.test <- simu.f.2.negurg.split$test[,1:ncol(simu.f.2.negurg.split$test)]

#Normalize Train Data
simu.f.2.negurg.trainData <- as.data.frame(apply(simu.f.2.negurg.train[,1:220],2,normalize))

#Normalize test data with respect to train data
simu.f.2.negurg.testData <- c()
for (x in 1:220){
  simu.f.2.negurg.testData <- cbind(simu.f.2.negurg.testData,normalize(simu.f.2.negurg.test[,x],y=simu.f.2.negurg.train[,x]))
}
colnames(simu.f.2.negurg.testData) <- colnames(simu.f.2.negurg.trainData)
simu.f.2.negurg.testData <- as.data.frame(simu.f.2.negurg.testData)


#Format data to be used for training
simu.f.2.negurg.trainTarget <- (simu.f.2.negurg.train[,221])
simu.f.2.negurg.testTarget <- (simu.f.2.negurg.test[,221])

simu.f.2.negurg.trainData <- data.matrix(simu.f.2.negurg.trainData)
simu.f.2.negurg.trainLabels <- data.matrix(simu.f.2.negurg.trainTarget)
simu.f.2.negurg.testData <- data.matrix(simu.f.2.negurg.testData)
simu.f.2.negurg.testLabels <- data.matrix(simu.f.2.negurg.testTarget)

#Stop the training when val_loss doesn't change for 20 epochs
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

#Create Model
model.simu.f.2.negurg<- keras_model_sequential()

model.simu.f.2.negurg%>% #Architechture can be tweaked to change model
  layer_dense(units = 220, activation = 'relu', kernel_initializer='he_normal',bias_initializer ='he_normal', input_shape = c(220)) %>%
  layer_dense(units = 9, activation = 'relu', kernel_initializer='he_normal',bias_initializer ='he_normal') %>%
  layer_dense(units = 1, activation = 'linear')

model.simu.f.2.negurg%>% compile(
  loss = 'mean_squared_error', #Can be tweaked to Change model
  optimizer = 'adam', #Can be tweaked to change model 
  metrics = list("mean_absolute_error") #Has no effect on training, just for visualization 
)

#Train Model
model.simu.f.2.negurg%>% fit(
  simu.f.2.negurg.trainData,
  simu.f.2.negurg.trainLabels,
  epochs = 2000,
  validation_split = 0.75, #can be tweaked to change model
  callbacks = list(early_stop),
  verbose = 0
)

#Get Loss and Mean Absolute Error of Model on Testing Data
c(loss.f.2.negurg, mae.f.2.negurg) %<-% (model.simu.f.2.negurg %>% evaluate(simu.f.2.negurg.testData, simu.f.2.negurg.testLabels, verbose = 0))

paste0("Mean absolute error on test set: ", sprintf("%.2f", mae.f.2.negurg))

# Store predictions in a new dataset
predictions.simu.f.2.negurg <- model.simu.f.2.negurg%>% predict(simu.f.2.negurg.testData, batch_size = 2374)

#Next Section is for Variable Importance
#Resource used https://bgreenwell.github.io/pdp/articles/pdp-example-tensorflow.html

#Prediction Function
pred_wrapper <- function(object, newdata) {
  predict(object, x = as.matrix(newdata)) %>%
    as.vector()
}

#Create and Print Variable importance plot
simu.f.2.negurg.variableImportancePlot <- vip(
  object = model.simu.f.2.negurg,          # fitted model
  method = "permute",                 # permutation-based VI scores,
  num_features = 10,       # top X features you want to keep track of (but ncol(simu.f.2.negurg.trainData) for all variables)
  pred_wrapper = pred_wrapper,            # user-defined prediction function
  train = as.data.frame(simu.f.2.negurg.trainData) ,    # training data
  target = simu.f.2.negurg.trainLabels,                   # response values used for training
  metric = "rsquared",                # evaluation metric
  progress = "text"                 # request a text-based progress bar
)
print(simu.f.2.negurg.variableImportancePlot)
print(simu.f.2.negurg.variableImportancePlot[1])

#Get weights 
weights.f.2.negurg <- get_weights(model.simu.f.2.negurg)


