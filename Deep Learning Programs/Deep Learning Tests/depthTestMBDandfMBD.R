#combinat
combinat <- function(n,p){
  if (n<p){combinat=0}
  else {combinat=exp(lfactorial(n)-(lfactorial(p)+lfactorial(n-p)))}
}

#fMBD
fMBD <- function(data){
  p=dim(data)[1]
  n=dim(data)[2]
  rmat=apply(data,1,rank)
  down=rmat-1
  up=n-rmat
  (rowSums(up*down)/p+n-1)/combinat(n,2)
}
#MBD
MBD <- function(x, xRef=NULL)
{
  n <- nrow(x); d <- ncol(x) # n: number of observations (samples);  d: dimension of the data
  x <- as.matrix(x)
  
  if (length(xRef)==0) {  ## MBD with respect to the same sample
    
    ## depth computation
    if (ncol(x) == 1) {x <- t(x)}
    depth <- matrix(0,1,n)
    ordered.matrix <- x
    if (n>1) {
      for (columns in 1:d) {
        ordered.matrix[,columns] <- sort(x[,columns])
        for (element in 1:n) {
          index.1 <- length(which(ordered.matrix[,columns] < x[element,columns]))
          index.2 <- length(which(ordered.matrix[,columns] <= x[element,columns]))
          multiplicity <- index.2 - index.1
          depth[element] <- depth[element] + index.1 * (n - (index.2)) + multiplicity * (n - index.2 + index.1) + choose(multiplicity,2)
        }   ### end FOR element
      }  ### end FOR columns
      depth <- depth / (d * choose(n,2) )
    } ## end IF
    if (n==1) {deepest <- x; depth <- 0}
    ordering<-order(depth,decreasing=TRUE)
    
  } ## end IF no reference sample
  
  else {
    xRef <- as.matrix(xRef)
    if (ncol(xRef)!=d) {stop("Dimensions of x and xRef do not match")}
    n0 <- nrow(xRef)
    
    ## depth computations
    if (ncol(x) == 1) {x <- t(x)}
    depth <- matrix(0,1,n)
    ordered.matrix <- xRef
    if (n0>1) {
      for (columns in 1:d) {
        ordered.matrix[,columns] <- sort(xRef[,columns])
        for (element in 1:n) {
          index.1 <- length(which(ordered.matrix[,columns] < x[element,columns]))
          index.2 <- length(which(ordered.matrix[,columns] <= x[element,columns]))
          multiplicity <- index.2 - index.1
          depth[element] <- depth[element] + (index.1 + multiplicity ) * (n0 - index.1 - multiplicity) + multiplicity * ( index.1 + (multiplicity-1)/2)
        }   ### end FOR element
      }   ### end FOR columns
      depth <- depth / (d * choose(n0,2) )
    } ## end IF
    if (n==1) {deepest <- x; depth <- 0}
    ordering<-order(depth,decreasing=TRUE)
    
  }  ## end ELSE
  return(list(ordering=ordering,MBD=depth))
}

library(depthTests)

freesurfer <- read.csv("D:/Research/Yihong Zhao/Brain Data Sets/freesurfer.csv")
behavioral <- read.csv("D:/Research/Yihong Zhao/Brain Data Sets/behavioraldata.csv")

freesurferGendered <- split(freesurfer,freesurfer$Gender)

thicknessM <- freesurferGendered$M[, grep("_Thck$" , colnames(freesurfer))]
areaM <- freesurferGendered$M[, grep("_Area", colnames(freesurfer))]
grayVolM <- freesurferGendered$M[, grep("_GrayVol", colnames(freesurfer))]
foldingIndexM <- freesurferGendered$M[, grep("_FoldInd" , colnames(freesurfer))]
curveIndexM <- freesurferGendered$M[, grep("_CurvInd" , colnames(freesurfer))]
gausCurvM <- freesurferGendered$M[, grep("_GausCurv" , colnames(freesurfer))]
meanCurvM <- freesurferGendered$M[, grep("_MeanCurv" , colnames(freesurfer))]

thicknessF <- freesurferGendered$F[, grep("_Thck$" , colnames(freesurfer))]
areaF <- freesurferGendered$F[, grep("_Area", colnames(freesurfer))]
grayVolF <- freesurferGendered$F[, grep("_GrayVol", colnames(freesurfer))]
foldingIndexF <- freesurferGendered$F[, grep("_FoldInd" , colnames(freesurfer))]
curveIndexF <- freesurferGendered$F[, grep("_CurvInd" , colnames(freesurfer))]
gausCurvF <- freesurferGendered$F[, grep("_GausCurv" , colnames(freesurfer))]
meanCurvF <- freesurferGendered$F[, grep("_MeanCurv" , colnames(freesurfer))]

thickness <- freesurfer[, grep("_Thck$" , colnames(freesurfer))]
area <- freesurfer[, grep("_Area", colnames(freesurfer))]
grayVol <- freesurfer[, grep("_GrayVol", colnames(freesurfer))]
foldingIndex <- freesurfer[, grep("_FoldInd" , colnames(freesurfer))]
curveIndex <- freesurfer[, grep("_CurvInd" , colnames(freesurfer))]
gausCurv <- freesurfer[, grep("_GausCurv" , colnames(freesurfer))]
meanCurv <- freesurfer[, grep("_MeanCurv" , colnames(freesurfer))]

#MBD
MBD_thicknessM_ordering <- MBD(thicknessM)
MBD_thicknessF_ordering <- MBD(thicknessF)
MBD_thickness_ordering <- MBD(thickness)

MBD_areaM_ordering <- MBD(areaM)
MBD_areaF_ordering <- MBD(areaF)
MBD_area_ordering <- MBD(area)

MBD_grayVolM_ordering <- MBD(grayVolM)
MBD_grayVolF_ordering <- MBD(grayVolF)
MBD_grayVol_ordering <- MBD(grayVol)

MBD_foldingIndexM_ordering <- MBD(foldingIndexM)
MBD_foldingIndexF_ordering <- MBD(foldingIndexF)
MBD_foldingIndex_ordering <- MBD(foldingIndex)

MBD_curveIndexM_ordering <- MBD(curveIndexM)
MBD_curveIndexF_ordering <- MBD(curveIndexF)
MBD_curveIndex_ordering <- MBD(curveIndex)

MBD_gausCurvM_ordering <- MBD(gausCurvM)
MBD_gausCurvF_ordering <- MBD(gausCurvF)
MBD_gausCurv_ordering <- MBD(gausCurv)

MBD_meanCurvM_ordering <- MBD(meanCurvM)
MBD_meanCurvF_ordering <- MBD(meanCurvF)
MBD_meanCurv_ordering <- MBD(meanCurv)

male.order <-cbind(MBD_thicknessM_ordering$ordering,MBD_grayVolM_ordering$ordering,MBD_areaM_ordering$ordering,
                   MBD_foldingIndexM_ordering$ordering,MBD_curveIndexM_ordering$ordering,MBD_gausCurvM_ordering$ordering,
                   MBD_meanCurvM_ordering$ordering)
male.MBD <-rbind(MBD_thicknessM_ordering$MBD,MBD_grayVolM_ordering$MBD,MBD_areaM_ordering$MBD,MBD_foldingIndexM_ordering$MBD,
                 MBD_curveIndexM_ordering$MBD,MBD_gausCurvM_ordering$MBD,MBD_meanCurvM_ordering$MBD)
male.MBD <- t(male.MBD)

female.order <-cbind(MBD_thicknessF_ordering$ordering,MBD_grayVolF_ordering$ordering,MBD_areaF_ordering$ordering,
                   MBD_foldingIndexF_ordering$ordering,MBD_curveIndexF_ordering$ordering,MBD_gausCurvF_ordering$ordering,
                   MBD_meanCurvF_ordering$ordering)
female.MBD <-rbind(MBD_thicknessF_ordering$MBD,MBD_grayVolF_ordering$MBD,MBD_areaF_ordering$MBD,MBD_foldingIndexF_ordering$MBD,
                 MBD_curveIndexF_ordering$MBD,MBD_gausCurvF_ordering$MBD,MBD_meanCurvF_ordering$MBD)
female.MBD <- t(female.MBD)

full.order <-cbind(MBD_thickness_ordering$ordering,MBD_grayVol_ordering$ordering,MBD_area_ordering$ordering,
                   MBD_foldingIndex_ordering$ordering,MBD_curveIndex_ordering$ordering,MBD_gausCurv_ordering$ordering,
                   MBD_meanCurv_ordering$ordering)
full.MBD <-rbind(MBD_thickness_ordering$MBD,MBD_grayVol_ordering$MBD,MBD_area_ordering$MBD,MBD_foldingIndex_ordering$MBD,
                 MBD_curveIndex_ordering$MBD,MBD_gausCurv_ordering$MBD,MBD_meanCurv_ordering$MBD)
full.MBD <- t(full.MBD)


#fMBD
fMBD_thicknessM_ordering <- fMBD(thicknessM)
fMBD_thicknessF_ordering <- fMBD(thicknessF)
fMBD_thickness_ordering <- fMBD(thickness)

fMBD_areaM_ordering <- fMBD(areaM)
fMBD_areaF_ordering <- fMBD(areaF)
fMBD_area_ordering <- fMBD(area)

fMBD_grayVolM_ordering <- fMBD(grayVolM)
fMBD_grayVolF_ordering <- fMBD(grayVolF)
fMBD_grayVol_ordering <- fMBD(grayVol)

fMBD_foldingIndexM_ordering <- fMBD(foldingIndexM)
fMBD_foldingIndexF_ordering <- fMBD(foldingIndexF)
fMBD_foldingIndex_ordering <- fMBD(foldingIndex)

fMBD_curveIndexM_ordering <- fMBD(curveIndexM)
fMBD_curveIndexF_ordering <- fMBD(curveIndexF)
fMBD_curveIndex_ordering <- fMBD(curveIndex)

fMBD_gausCurvM_ordering <- fMBD(gausCurvM)
fMBD_gausCurvF_ordering <- fMBD(gausCurvF)
fMBD_gausCurv_ordering <- fMBD(gausCurv)

fMBD_meanCurvM_ordering <- fMBD(meanCurvM)
fMBD_meanCurvF_ordering <- fMBD(meanCurvF)
fMBD_meanCurv_ordering <- fMBD(meanCurv)

male.fMBD <-cbind(fMBD_grayVolM_ordering,fMBD_thicknessM_ordering,fMBD_areaM_ordering,fMBD_foldingIndexM_ordering,
                 fMBD_curveIndexM_ordering,fMBD_gausCurvM_ordering,fMBD_meanCurvM_ordering)

female.fMBD <-cbind(fMBD_thicknessF_ordering,fMBD_grayVolF_ordering,fMBD_areaF_ordering,fMBD_foldingIndexF_ordering,
                   fMBD_curveIndexF_ordering,fMBD_gausCurvF_ordering,fMBD_meanCurvF_ordering)

full.fMBD <-cbind(fMBD_thickness_ordering,fMBD_grayVol_ordering,fMBD_area_ordering,fMBD_foldingIndex_ordering,
                   fMBD_curveIndex_ordering,fMBD_gausCurv_ordering,fMBD_meanCurv_ordering)

