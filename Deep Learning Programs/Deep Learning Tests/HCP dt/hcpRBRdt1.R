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
keras_model_to_estimator <- function(
  keras_model = NULL, keras_model_path = NULL, custom_objects = NULL,
  model_dir = NULL, config = NULL) {
  
  if (is.null(keras_model) && is.null(keras_model_path))
    stop("Either keras_model or keras_model_path needs to be provided.")
  
  if (!is.null(keras_model_path)) {
    if (!is.null(keras_model))
      stop("Please specity either keras_model or keras_model_path but not both.")
    if (grepl("^(gs://|storage\\.googleapis\\.com)", keras_model_path))
      stop("'keras_model_path' is not a local path. Please copy the model locally first.")
    keras_model <- tf$keras$models$load_model(keras_model_path)
  }
  
  tryCatch(reticulate::py_get_attr(keras_model, "optimizer"),
           error = function(e) stop(
             "Given keras model has not been compiled yet. Please compile first 
             before creating the estimator.")
  )
  
  args <- as.list(environment(), all = TRUE)
  
  estimator <- py_suppress_warnings(
    tf$keras$estimator$model_to_estimator(
      keras_model = keras_model,
      keras_model_path = keras_model_path,
      custom_objects = custom_objects,
      model_dir = model_dir,
      config = config
    ))
  
  new_tf_keras_estimator(estimator, args = args)
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
dt1 <- readRDS("D:/Research/Yihong Zhao/Brain Data Sets/dt1.subid.rds")
dt2 <- readRDS("D:/Research/Yihong Zhao/Brain Data Sets/dt2.subid.rds")
dt4 <- readRDS("D:/Research/Yihong Zhao/Brain Data Sets/dt4grp.subid.rds")

#trimVars is a matrix of all the variable substrings you want to include
trimVars<-c("Subject","_Thck$","_Area$","_GrayVol$",
            "FS_L_Hippo_Vol","FS_L_Amygdala_Vol","FS_L_AccumbensArea_Vol", "FS_L_VentDC_Vol",
            "FS_L_ThalamusProper_Vol","FS_L_Caudate_Vol","FS_L_Putamen_Vol","FS_L_Pallidum_Vol",
            "FS_R_Hippo_Vol","FS_R_Amygdala_Vol","FS_R_AccumbensArea_Vol", "FS_R_VentDC_Vol",
            "FS_R_ThalamusProper_Vol","FS_R_Caudate_Vol","FS_R_Putamen_Vol","FS_R_Pallidum_Vol",
            "FS_L_Cerebellum_Cort_Vol","FS_R_Cerebellum_Cort_Vol","FS_BrainStem_Vol")

#Split the dataset by column, merge by Subject, then separate subjects into training and testing data
dt1rbr <- cbind(dt1[,1],dt1[,4])
colnames(dt1rbr) <- c("Subject","rbr")

trimmedFreesurfer <- trimByCol(freesurfer,trimVars)
trimmedFreesurfer <- merge(dt1rbr,trimmedFreesurfer, by.x = "Subject",by.y = "Subject")
trimmedFreesurfer <- na.omit(trimmedFreesurfer)

hcp.rbr.dt1.split <- crossValidate(trimmedFreesurfer,seed = seed, trainPercentage = 0.8)
names(hcp.rbr.dt1.split) <- c("train","test")

hcp.rbr.dt1.train <- hcp.rbr.dt1.split$train[,2:ncol(hcp.rbr.dt1.split$train)]
hcp.rbr.dt1.test <- hcp.rbr.dt1.split$test[,2:ncol(hcp.rbr.dt1.split$test)]

#Normalize Train Data
hcp.rbr.dt1.trainData <- as.data.frame(apply(hcp.rbr.dt1.train[,2:224],2,normalize))

#Normalize test data with respect to train data
hcp.rbr.dt1.testData <- c()
for (x in 2:224){
  hcp.rbr.dt1.testData <- cbind(hcp.rbr.dt1.testData,normalize(hcp.rbr.dt1.test[,x],y=hcp.rbr.dt1.train[,x]))
}
colnames(hcp.rbr.dt1.testData) <- colnames(hcp.rbr.dt1.trainData)
hcp.rbr.dt1.testData <- as.data.frame(hcp.rbr.dt1.testData)

#Format data to be used for training
hcp.rbr.dt1.trainTarget <- (hcp.rbr.dt1.train[,1])
hcp.rbr.dt1.testTarget <- (hcp.rbr.dt1.test[,1])

hcp.rbr.dt1.trainData <- data.matrix(hcp.rbr.dt1.trainData)
hcp.rbr.dt1.trainLabels <- data.matrix(hcp.rbr.dt1.trainTarget)
hcp.rbr.dt1.testData <- data.matrix(hcp.rbr.dt1.testData)
hcp.rbr.dt1.testLabels <- data.matrix(hcp.rbr.dt1.testTarget)

#Stop the training when val_loss doesn't change for 20 epochs
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

#Create Model
model.hcp.rbr.dt1<- keras_model_sequential()

model.hcp.rbr.dt1%>% #Architechture can be tweaked to change model
  layer_dense(units = 223, activation = 'relu', kernel_initializer='he_normal',bias_initializer ='he_normal', input_shape = c(223)) %>% 
  layer_dense(units = 1, activation = 'linear')

model.hcp.rbr.dt1%>% compile(
  loss = 'mean_squared_error', #Can be tweaked to Change model
  optimizer = 'adam', #Can be tweaked to change model 
  metrics = list("mean_absolute_error") #Has no effect on training, just for visualization 
)

#Train Model
model.hcp.rbr.dt1%>% fit(
  hcp.rbr.dt1.trainData,
  hcp.rbr.dt1.trainLabels,
  epochs = 500,
  validation_split = 0.5, #can be tweaked to change model
  batch_size = 10,
  callbacks = list(early_stop)
)


#Get Loss and Mean Absolute Error of Model on Testing Data
c(loss.dt1.rbr, mae.dt1.rbr) %<-% (model.hcp.rbr.dt1 %>% evaluate(hcp.rbr.dt1.testData, hcp.rbr.dt1.testLabels, verbose = 0))

paste0("Mean absolute error on test set: ", sprintf("%.2f", mae.dt1.rbr))

# Store predictions in a new dataset
predictions.hcp.rbr.dt1 <- model.hcp.rbr.dt1%>% predict(hcp.rbr.dt1.testData, batch_size = 2374)

#Next Section is for Variable Importance
#Resource used https://bgreenwell.github.io/pdp/articles/pdp-example-tensorflow.html

#Prediction Function
pred_wrapper <- function(object, newdata) {
  predict(object, x = as.matrix(newdata)) %>%
    as.vector()
}

#Create and Print Variable importance plot
hcp.rbr.dt1.variableImportancePlot <- vip(
  object = model.hcp.rbr.dt1,          # fitted model
  method = "permute",                 # permutation-based VI scores,
  num_features = 10,       # top X features you want to keep track of (but ncol(hcp.rbr.dt1.trainData) for all variables)
  pred_wrapper = pred_wrapper,            # user-defined prediction function
  train = as.data.frame(hcp.rbr.dt1.trainData) ,    # training data
  target = hcp.rbr.dt1.trainLabels,                   # response values used for training
  metric = "rsquared",                # evaluation metric
  progress = "text"                 # request a text-based progress bar
)
print(hcp.rbr.dt1.variableImportancePlot)
print(hcp.rbr.dt1.variableImportancePlot[1])

#Get weights 
weights.rbr.dt1 <- get_weights(model.hcp.rbr.dt1)





