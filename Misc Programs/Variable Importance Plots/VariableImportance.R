axisFontRatio = 0.4
legendFontRatio = 0.5
sizeOfData = 83
numberOfCategories = 6


dataMale<-RFmale[order(RFmale$MeanDecreaseGini),]

yMale<-c(1:sizeOfData)
xMale<-dataMale$MeanDecreaseGini

par(mfrow=c(1,2),las=1)

plot(xMale,yMale,main="RF Male",xlab = "Mean Decrease Gini", ylab = "", yaxt='n',cex=0.75,cex.axis=1,col=dataMale$Category,pch=16)
axis(2, at=1:sizeOfData, labels=dataMale$Variable,cex.axis=axisFontRatio)
legend (x = 2.6,y = 24.2, legend = levels(dataMale$Category), col = c(1:numberOfCategories), pch = 16,cex=legendFontRatio)

dataFemale<-RFfemale[order(RFfemale$MeanDecreaseGini),]

yFemale<-c(1:sizeOfData)
xFemale<-dataFemale$MeanDecreaseGini

plot(xFemale,yFemale,main="RF Female",xlab = "Mean Decrease Gini", ylab = "", yaxt='n',cex=0.75,cex.axis=1,col=dataFemale$Category,pch=16)
axis(2, at=1:sizeOfData, labels=dataFemale$Variable,cex.axis=axisFontRatio)
legend (x = 4.6,y = 24.2, legend = levels(dataFemale$Category), col = c(1:numberOfCategories), pch = 16,cex=legendFontRatio)
