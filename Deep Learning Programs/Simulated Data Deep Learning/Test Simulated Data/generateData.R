normalize <- function(x,y=x) { #Normalize list x in respect to list y
  num <- x - min(y)
  denom <- max(y) - min(y)
  
  return (num/denom)
  
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
library(keras)
library(tensorflow)
library(vip)
library(pdp)
library(ggplot2)
library(MASS)
library(RMThreshold)
#Generate Data
#Redo with 68 variables
data <- data.frame(0,0,0,0,0,0)
names(data) <- c("x1","x2","x3","x4","x5","y")
w <- c(-5,3,-3,0,7)
x <-mvrnorm(n=10000,c(0,3,2,-5,9), Sigma = diag(5))
xMeans <- apply(x,2,sd)
y <- x%*%w + rnorm(10000,0,0.1)

data <- cbind(x[,1:5],y[,1])
#Formatting Data
data.split <- crossValidate(data,seed = 1, trainPercentage = 0.8)
names(data.split) <- c("train","test")

data.train <- data.split$train[,1:ncol(data.split$train)]
data.test <- data.split$test[,1:ncol(data.split$test)]

#Normalize Train Data
data.trainData <- as.data.frame(apply(data.train[,1:5],2,normalize))

#Normalize test data with respect to train data
data.testData <- c()
for (i in 1:5){
  data.testData <- cbind(data.testData,normalize(data.test[,i],y=data.train[,i]))
}
colnames(data.testData) <- colnames(data.trainData)
data.testData <- as.data.frame(data.testData)



#Format data to be used for training
data.trainTarget <- (data.train[,6])
data.testTarget <- (data.test[,6])

data.trainData <- data.matrix(data.trainData)
data.trainTarget <- data.matrix(data.trainTarget)
data.testData <- data.matrix(data.testData)
data.testTarget <- data.matrix(data.testTarget)


#Stop the training when val_loss doesn't change for 20 epochs
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

#Create Model
testModel<- keras_model_sequential()

testModel%>% #Architechture can be tweaked to change model
  layer_dense(units = 5, activation = 'relu', kernel_initializer='he_normal',bias_initializer ='he_normal', input_shape = c(5)) %>% 
  layer_dense(units = 1, activation = 'linear')

testModel%>% compile(
  loss = 'mean_squared_error', #Can be tweaked to Change model
  optimizer = 'adam', #Can be tweaked to change model 
  metrics = list("mean_absolute_error") #Has no effect on training, just for visualization 
)

#Train Model
testModel%>% fit(
  data.trainData,
  data.trainTarget,
  epochs = 2000,
  validation_split = 0.75, #can be tweaked to change model
  callbacks = list(early_stop),
  verbose = 0
)

#Get Loss and Mean Absolute Error of Model on Testing Data
c(loss, mae) %<-% (testModel %>% evaluate(data.testData, data.testTarget, verbose = 0))

paste0("Mean absolute error on test set: ", sprintf("%.2f", mae))

# Store predictions in a new dataset
predictions <- testModel%>% predict(data.testData, batch_size = 2000)
predictions <- round(predictions)


#Prediction Function
pred_wrapper <- function(object, newdata) {
  predict(object, x = as.matrix(newdata)) %>%
    as.vector()
}

#Create and Print Variable importance plot
data.variableImportancePlot <- vip(
  object = testModel,          # fitted model
  method = "permute",                 # permutation-based VI scores,
  num_features = 10,       # top X features you want to keep track of (but ncol(simu.f.age.trainData) for all variables)
  pred_wrapper = pred_wrapper,            # user-defined prediction function
  train = as.data.frame(data.trainData) ,    # training data
  target = data.trainTarget,                   # response values used for training
  metric = "rsquared",                # evaluation metric
  progress = "text"                 # request a text-based progress bar
)
#Get Variable Importance 
print(data.variableImportancePlot)
print(data.variableImportancePlot[1])

#Get weights (Odd number elements of weights list are weights, even number elements are biases)
weights <- get_weights(testModel)

weightLayer1 <- weights[[3]]