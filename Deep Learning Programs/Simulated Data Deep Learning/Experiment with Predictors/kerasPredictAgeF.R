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
library(pdp)
library(vip)
library(ggplot2)
library(lime)
library(corrplot)

#Set random seed for semi-reproducibility (TensorFlow on the backend has its own randomization so setting a seed doesnt effect it)
seed = 12345
set.seed(seed)

#Import Data Sets (change filepath to your computer)
freesurfer <- read.csv("D:/Research/Yihong Zhao/Brain Data Sets/freesurfer.csv")
behavioral <- read.csv("D:/Research/Yihong Zhao/Brain Data Sets/behavioraldata.csv")

#trimVars is a matrix of all the variable substrings you want to include
trimVars<-c("Subject","_Thck$","_Area$","_GrayVol$",
            "FS_L_Hippo_Vol","FS_L_Amygdala_Vol","FS_L_AccumbensArea_Vol", "FS_L_VentDC_Vol",
            "FS_L_ThalamusProper_Vol","FS_L_Caudate_Vol","FS_L_Putamen_Vol","FS_L_Pallidum_Vol",
            "FS_R_Hippo_Vol","FS_R_Amygdala_Vol","FS_R_AccumbensArea_Vol", "FS_R_VentDC_Vol",
            "FS_R_ThalamusProper_Vol","FS_R_Caudate_Vol","FS_R_Putamen_Vol","FS_R_Pallidum_Vol")

#Split the dataset by column, merge by Subject, then separate subjects into training and testing data
trimmedFreesurfer <- trimByCol(freesurfer,trimVars)
trimmedBehavioral <- trimByCol(behavioral,c("Subject","Gender"))
trimmedFreesurfer <- merge(trimmedBehavioral,trimmedFreesurfer, by.x = "Subject",by.y = "Subject")
trimmedFreesurfer <- na.omit(trimmedFreesurfer)
gender <- c()
for(x in 1:nrow(trimmedFreesurfer)){
  if((trimmedFreesurfer[x,2]=="M")){
    gender <- rbind(gender, c(0))
  } else {
    gender <- rbind(gender, c(1))
  }
}
trimmedFreesurfer[,2] <- gender
hcp.gender.split <- crossValidate(trimmedFreesurfer,seed = seed, trainPercentage = 0.8)
names(hcp.gender.split) <- c("train","test")

hcp.gender.train <- hcp.gender.split$train[,2:ncol(hcp.gender.split$train)]
hcp.gender.test <- hcp.gender.split$test[,2:ncol(hcp.gender.split$test)]

#Normalize Train Data
hcp.gender.trainData <- as.data.frame(apply(hcp.gender.train[,2:221],2,normalize))

#Normalize test data with respect to train data
hcp.gender.testData <- c()
for (x in 2:221){
  hcp.gender.testData <- cbind(hcp.gender.testData,normalize(hcp.gender.test[,x],y=hcp.gender.train[,x]))
}
colnames(hcp.gender.testData) <- colnames(hcp.gender.trainData)
hcp.gender.testData <- as.data.frame(hcp.gender.testData)

#Format data to be used for training M = 0, F = 1
hcp.gender.trainTarget <- to_categorical(hcp.gender.train[,1])
hcp.gender.testTarget <- to_categorical(hcp.gender.test[,1])

simu.f.age.trainData <- data.matrix(hcp.gender.trainData)
simu.f.age.trainLabels <- data.matrix(hcp.gender.trainTarget)
simu.f.age.testData <- data.matrix(hcp.gender.testData)
simu.f.age.testLabels <- data.matrix(hcp.gender.testTarget)

#Stop the training when val_loss doesn't change for 20 epochs
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

#Create Model
inputShape <- c(105, 105, 1)
leftInput <- layer_input(inputShape)
rightInput <- layer_input(inputShape)

model<- keras_model_sequential()

model %>%
  layer_conv_2d(filter=64,
                kernel_size=c(10,10),
                activation = "relu",
                input_shape=inputShape,
                kernel_initializer = initializer_random_normal(0, 1e-2),
                kernel_regularizer = regularizer_l2(2e-4)) %>%
  layer_max_pooling_2d() %>%
  
  layer_conv_2d(filter=128,
                kernel_size=c(7,7),
                activation = "relu",
                kernel_initializer = initializer_random_normal(0, 1e-2),
                kernel_regularizer = regularizer_l2(2e-4),
                bias_initializer = initializer_random_normal(0.5, 1e-2)) %>%
  layer_max_pooling_2d() %>%
  
  layer_conv_2d(filter=128,
                kernel_size=c(4,4),
                activation = "relu",
                kernel_initializer = initializer_random_normal(0, 1e-2),
                kernel_regularizer = regularizer_l2(2e-4),
                bias_initializer = initializer_random_normal(0.5, 1e-2)) %>%
  layer_max_pooling_2d() %>%
  
  layer_conv_2d(filter=256,
                kernel_size=c(4,4),
                activation = "relu",
                kernel_initializer = initializer_random_normal(0, 1e-2),
                kernel_regularizer = regularizer_l2(2e-4),
                bias_initializer = initializer_random_normal(0.5, 1e-2)) %>%
  
  layer_flatten() %>%
  layer_dense(4096, 
              activation = "sigmoid",
              kernel_initializer = initializer_random_normal(0, 1e-2),
              kernel_regularizer = regularizer_l2(1e-3),
              bias_initializer = initializer_random_normal(0.5, 1e-2)) 

encoded_left <- leftInput %>% model
encoded_right <- rightInput %>% model

#Train Model
model.simu.f.age%>% fit(
  simu.f.age.trainData,
  simu.f.age.trainLabels,
  epochs = 5000,
  validation_split = 0.5, #can be tweaked to change model
  callbacks = list(early_stop),
  verbose =FALSE
)

