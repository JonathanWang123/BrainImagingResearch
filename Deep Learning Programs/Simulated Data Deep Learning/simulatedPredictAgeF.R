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
#Import Libraries
library(keras)
library(tensorflow)
library(vip)
library(pdp)
library(ggplot2)
library(lime)
library(corrplot)

#Set random seed for semi-reproducibility (TensorFlow on the backend has its own randomization so setting a seed doesnt effect it)
seed = 12345
set.seed(seed)

#Import Database
simu.f <- readRDS("D:/Research/Yihong Zhao/Brain Data Sets/Simulated Data 2/simu.f.2.rds")

weight.f <- readRDS("D:/Research/Yihong Zhao/Brain Data Sets/Simulated Data 2/weight.f.updated.rds")

#Separate out variables used to predict and variable to be predicted
simu.f.age <- cbind(simu.f[,1:220],simu.f[,221])
colnames(simu.f.age)[221] <- "age"

#Split into train and test data
simu.f.age.split <- crossValidate(simu.f.age,seed = seed, trainPercentage = 0.8)
names(simu.f.age.split) <- c("train","test")

simu.f.age.train <- simu.f.age.split$train[,1:ncol(simu.f.age.split$train)]
simu.f.age.test <- simu.f.age.split$test[,1:ncol(simu.f.age.split$test)]

#Normalize Train Data
simu.f.age.trainData <- as.data.frame(apply(simu.f.age.train[,1:220],2,normalize))

#Normalize test data with respect to train data
simu.f.age.testData <- c()
for (x in 1:220){
  simu.f.age.testData <- cbind(simu.f.age.testData,normalize(simu.f.age.test[,x],y=simu.f.age.train[,x]))
}
colnames(simu.f.age.testData) <- colnames(simu.f.age.trainData)
simu.f.age.testData <- as.data.frame(simu.f.age.testData)


#Format data to be used for training
simu.f.age.trainTarget <- (simu.f.age.train[,221])
simu.f.age.testTarget <- (simu.f.age.test[,221])

simu.f.age.trainData <- data.matrix(simu.f.age.trainData)
simu.f.age.trainLabels <- data.matrix(simu.f.age.trainTarget)
simu.f.age.testData <- data.matrix(simu.f.age.testData)
simu.f.age.testLabels <- data.matrix(simu.f.age.testTarget)

#Stop the training when val_loss doesn't change for 20 epochs
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

#Create Model
inputs <- layer_input(shape = c(220))
predictions <- inputs %>%
  layer_dense(units = 220, activation = 'linear', kernel_initializer='Identity',bias_initializer ='RandomNormal') %>% 
  layer_dense(units = 9, activation = 'linear') %>% 
  layer_dense(units = 1, activation = 'linear')
model.simu.f.age <- keras_model(inputs = inputs, outputs = predictions)
model.simu.f.age %>% compile(
  loss = 'mean_squared_error', #Can be tweaked to Change model
  optimizer = 'adam', #Can be tweaked to change model 
  metrics = list("mean_absolute_error") #Has no effect on training, just for visualization 
)

model.simu.f.age%>% compile(
  loss = 'mean_squared_error', #Can be tweaked to Change model
  optimizer = 'adam', #Can be tweaked to change model 
  metrics = list("mean_absolute_error") #Has no effect on training, just for visualization 
)

#Train Model
model.simu.f.age%>% fit(
  simu.f.age.trainData,
  simu.f.age.trainLabels,
  epochs = 5000,
  validation_split = 0.5, #can be tweaked to change model
  callbacks = list(early_stop),
  verbose =FALSE
)

#Get Loss and Mean Absolute Error of Model on Testing Data
c(loss, mae) %<-% (model.simu.f.age %>% evaluate(simu.f.age.testData, simu.f.age.testLabels, verbose = 0))

paste0("Mean absolute error on test set: ", sprintf("%.2f", mae))

weights <- get_weights(model.simu.f.age)

sort(abs(cor(weights[[1]], weight.f[,2])),decreasing = T)

weight1<-data.frame(colnames(simu.f.age.trainData),weights[[1]])
colnames(weight1)<-c("ROI",paste0("wt",1:dim(weight1)[1]))

tt<-merge(weight1, weight.f, by="ROI")

round(sort(abs(cor(tt[1:16,2:221],tt$joint1[1:16])),decreasing = T),2)
round(sort(abs(cor(tt[17:84,2:221],tt$joint1[17:84])),decreasing = T),2)
round(sort(abs(cor(tt[85:152,2:221],tt$joint1[85:152])),decreasing = T),2)
round(sort(abs(cor(tt[153:220,2:221],tt$joint1[153:220])),decreasing = T),2)

round(sort(abs(cor(tt[,2:221],tt$subcort1)),decreasing = T),2)
round(sort(abs(cor(tt[,2:221],tt$gvol1)),decreasing = T),2)
round(sort(abs(cor(tt[,2:221],tt$area1)),decreasing = T),2)
round(sort(abs(cor(tt[,2:221],tt$thick1)),decreasing = T),2)
round(sort(abs(cor(tt[,2:221],tt$thick2)),decreasing = T),2)
round(sort(abs(cor(tt[,2:221],tt$thick3)),decreasing = T),2)
round(sort(abs(cor(tt[,2:221],tt$thick4)),decreasing = T),2)

round(sort(abs(cor(tt[1:16,2:221],tt$subcort1[1:16])),decreasing = T),2)
round(sort(abs(cor(tt[17:84,2:221],tt$gvol1[17:84])),decreasing = T),2)
round(sort(abs(cor(tt[85:152,2:221],tt$area1[85:152])),decreasing = T),2)
round(sort(abs(cor(tt[153:220,2:221],tt$thick1[153:220])),decreasing = T),2)
round(sort(abs(cor(tt[153:220,2:221],tt$thick2[153:220])),decreasing = T),2)
round(sort(abs(cor(tt[153:220,2:221],tt$thick3[153:220])),decreasing = T),2)
round(sort(abs(cor(tt[153:220,2:221],tt$thick4[153:220])),decreasing = T),2)


s.cor<-sort(abs(cor(tt[,2:221], tt$joint1)),decreasing = T, index.return=T)



cor(tt[,s.cor$ix[1:3]+1],tt$joint1)

plot(1:220,tt$wt210, ylim=c(-0.5,0.2))
points(1:220, tt$wt211, col=2)
points(1:220, tt$wt218, col=3)
points(1:220, tt$joint1, col=4)

cor(tt[,s.cor$ix[1:3]+1],tt$joint1)


cor(tt[1:16,],tt$joint1[1:16])


s.cor$ix[4:6]

plot(1:220,tt$wt219, ylim=c(-0.5,0.2))
plot(1:220,tt$wt219)
points(1:220, tt$wt217, col=2)
points(1:220, tt$wt206, col=3)
points(1:220, tt$joint1, col=4)

# Store predictions in a new dataset
predictions.simu.f.age <- model.simu.f.age%>% predict(simu.f.age.testData, batch_size = 2374)

predict(model.simu.f.age,simu.f.age.testData)
#Next Section is for Variable Importance
#Resource used https://bgreenwell.github.io/pdp/articles/pdp-example-tensorflow.html

#Prediction Function
pred_wrapper <- function(object, newdata) {
  predict(object, x = as.matrix(newdata)) %>%
    as.vector()
}

#Create and Print Variable importance plot
simu.f.age.variableImportancePlot <- vip(
  object = model.simu.f.age,          # fitted model
  method = "permute",                 # permutation-based VI scores,
  num_features = 10,       # top X features you want to keep track of (but ncol(simu.f.age.trainData) for all variables)
  pred_wrapper = pred_wrapper,            # user-defined prediction function
  train = as.data.frame(simu.f.age.trainData) ,    # training data
  target = simu.f.age.trainLabels,                   # response values used for training
  metric = "rsquared",                # evaluation metric
  progress = "text"                 # request a text-based progress bar
)
print(simu.f.age.variableImportancePlot)
print(simu.f.age.variableImportancePlot[1])

#Get weights 
model.simu.f.age %>% save_model_weights_tf("testing1")
#
model_type.keras.engine.training.Model <- function(x, ...) {
  if (!requireNamespace('keras', quietly = TRUE)) {
    stop('The keras package is required for predicting keras models')
  }
  num_layers <- length(x$layers)
  if (keras::get_config(keras::get_layer(x, index = num_layers))$activation == 'linear') {
    'regression'
  } else {
    'classification'
  }
}
predict_model.keras.engine.training.Model <- function(x, newdata, type, ...) {
  if (!requireNamespace('keras', quietly = TRUE)) {
    stop('The keras package is required for predicting keras models')
  }
  res <- predict(x, as.matrix(newdata))
  if (type == 'raw') {
    data.frame(Response = res[, 1])
    cat("hello")
  } else {
    if (ncol(res) == 1) {
      res <- cbind(1 - res, res)
    }
    colnames(res) <- as.character(seq_len(ncol(res)))
    as.data.frame(res, check.names = FALSE)
  }
}

train_data_frame <- as.data.frame(simu.f.age.trainData)
test_data_frame <- as.data.frame(simu.f.age.testData)
predict_model(model.simu.f.age, train_data_frame, type='raw')


explainer <- lime::lime (
  x              = (train_data_frame), 
  model          =(model.simu.f.age))

simu.f.explanation <- lime::explain (
  train_data_frame[1:10,], 
  explainer    = explainer, 
  n_features   = 220)
plot_features(simu.f.explanation[1:50,]) 
plot_explanations (simu.f.explanation[1:50,]) 