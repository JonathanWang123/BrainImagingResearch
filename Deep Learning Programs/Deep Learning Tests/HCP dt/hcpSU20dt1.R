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
library(lime)
library(corrplot)
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
dt1su20 <- cbind(dt1[,1],dt1[,7])
colnames(dt1su20) <- c("Subject","su.20")

trimmedFreesurfer <- trimByCol(freesurfer,trimVars)
trimmedFreesurfer <- merge(dt1su20,trimmedFreesurfer, by.x = "Subject",by.y = "Subject")
trimmedFreesurfer <- na.omit(trimmedFreesurfer)

hcp.su20.dt1.split <- crossValidate(trimmedFreesurfer,seed = seed, trainPercentage = 0.8)
names(hcp.su20.dt1.split) <- c("train","test")

hcp.su20.dt1.train <- hcp.su20.dt1.split$train[,2:ncol(hcp.su20.dt1.split$train)]
hcp.su20.dt1.test <- hcp.su20.dt1.split$test[,2:ncol(hcp.su20.dt1.split$test)]

#Normalize Train Data
hcp.su20.dt1.trainData <- as.data.frame(apply(hcp.su20.dt1.train[,2:224],2,normalize))

#Normalize test data with respect to train data
hcp.su20.dt1.testData <- c()
for (x in 2:224){
  hcp.su20.dt1.testData <- cbind(hcp.su20.dt1.testData,normalize(hcp.su20.dt1.test[,x],y=hcp.su20.dt1.train[,x]))
}
colnames(hcp.su20.dt1.testData) <- colnames(hcp.su20.dt1.trainData)
hcp.su20.dt1.testData <- as.data.frame(hcp.su20.dt1.testData)

#Format data to be used for training
hcp.su20.dt1.trainTarget <- to_categorical(hcp.su20.dt1.train[,1])
hcp.su20.dt1.testTarget <- to_categorical(hcp.su20.dt1.test[,1])

hcp.su20.dt1.trainData <- data.matrix(hcp.su20.dt1.trainData)
hcp.su20.dt1.trainLabels <- data.matrix(hcp.su20.dt1.trainTarget)
hcp.su20.dt1.testData <- data.matrix(hcp.su20.dt1.testData)
hcp.su20.dt1.testLabels <- data.matrix(hcp.su20.dt1.testTarget)

#Stop the training when val_loss doesn't change for 20 epochs
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

#Create Model
model.hcp.su20.dt1<- keras_model_sequential()

model.hcp.su20.dt1%>% #Architechture can be tweaked to change model
  layer_dense(units = 223, activation = 'relu', kernel_initializer='he_normal',bias_initializer ='he_normal', input_shape = c(223)) %>% 
  layer_dense(units = 2, activation = 'sigmoid')

model.hcp.su20.dt1%>% compile(
  loss = 'binary_crossentropy', #Can be tweaked to Change model
  optimizer = 'adam', #Can be tweaked to change model 
  metrics = 'accuracy' #Has no effect on training, just for visualization 
)

#Train Model
model.hcp.su20.dt1%>% fit(
  hcp.su20.dt1.trainData,
  hcp.su20.dt1.trainLabels,
  epochs = 500,
  validation_split = 0.5, #can be tweaked to change model
  batch_size = 10,
  callbacks = list(early_stop)
)
# Predict the classes for the test data
hcp.su20.dt1.predictions <- model.hcp.su20.dt1%>% predict_classes(hcp.su20.dt1.testData, batch_size = 48)
# Confusion matrix
hcp.su20.dt1.confusion <- table(hcp.su20.dt1.test[,1], hcp.su20.dt1.predictions)

hcp.su20.dt1.correct <- 0
hcp.su20.dt1.incorrect <- 0
for(row in 1:nrow(hcp.su20.dt1.confusion)) {
  for(col in 1:ncol(hcp.su20.dt1.confusion)) {
    if(rownames(hcp.su20.dt1.confusion)[row] == colnames(hcp.su20.dt1.confusion)[col]){
      hcp.su20.dt1.correct <- sum(hcp.su20.dt1.correct,hcp.su20.dt1.confusion[row,col])
    } else {
      hcp.su20.dt1.incorrect <- sum(hcp.su20.dt1.incorrect,hcp.su20.dt1.confusion[row,col])
    }
  }
}
cat("The accuracy of the multilayer perceptron is", 100*(hcp.su20.dt1.correct/(hcp.su20.dt1.correct+hcp.su20.dt1.incorrect)),"%")
#Next Section is for Variable Importance

model_type.keras.models.Sequential <- function(x, ...) {
  "classification"}
predict_model.keras.engine.sequential.Sequential <- function (x, newdata, type, ...) {
  pred <- predict_proba (object = x, x = as.matrix(newdata))
  data.frame (Positive = pred, Negative = 1 - pred) }
train_data_frame <- as.data.frame(hcp.su20.dt1.trainData)
test_data_frame <- as.data.frame(hcp.su20.dt1.testData)
predict_model (x       = model.hcp.su20.dt1, 
               newdata = train_data_frame, 
               type    = 'raw') %>%
  tibble::as_tibble()

explainer <- lime::lime (
  x              = train_data_frame, 
  model          = model.hcp.su20.dt1, 
  bin_continuous = FALSE)

hcp.su20.dt1.explanation <- lime::explain (
  train_data_frame[1:10, ], 
  explainer    = explainer, 
  n_labels     = 2, 
  n_features   = 220, 
  kernel_width = 0.5)
plot_features(hcp.su20.dt1.explanation[1:5,]) 
plot_explanations (hcp.su20.dt1.explanation[1:5,]) 

#Get Weights
weights.hcp.su20.dt1 <- get_weights(model.hcp.su20.dt1)


