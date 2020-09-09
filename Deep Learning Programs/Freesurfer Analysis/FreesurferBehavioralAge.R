library(depthTools)

mergeData <- merge(freesurfer, behavioral, by = "Subject")
ageCol <- grep("Age$", names(mergeData))
#move age column to col 3
mergeData <- mergeData[, c(c(1:2), ageCol, (1:ncol(mergeData))[-ageCol])]

behavioralAge <- split(mergeData, mergeData$Age)

thickness22 <- behavioralAge$`22-25`[, grep("_Thck$" , colnames(freesurfer))+3]
area22 <- behavioralAge$`22-25`[, grep("_Area" , colnames(freesurfer))+3]
grayVol22 <- behavioralAge$`22-25`[, grep("_GrayVol" , colnames(freesurfer))+3]

thickness26 <- behavioralAge$`26-30`[, grep("_Thck$" , colnames(freesurfer))+3]
area26 <- behavioralAge$`26-30`[, grep("_Area" , colnames(freesurfer))+3]
grayVol26 <- behavioralAge$`26-30`[, grep("_GrayVol" , colnames(freesurfer))+3]

thickness31 <- behavioralAge$`31-35`[, grep("_Thck$" , colnames(freesurfer))+3]
area31 <- behavioralAge$`31-35`[, grep("_Area" , colnames(freesurfer))+3]
grayVol31 <- behavioralAge$`31-35`[, grep("_GrayVol" , colnames(freesurfer))+3]

thickness36 <- behavioralAge$`36+`[, grep("_Thck$" , colnames(freesurfer))+3]
area36 <- behavioralAge$`36+`[, grep("_Area" , colnames(freesurfer))+3]
grayVol36 <- behavioralAge$`36+`[, grep("_GrayVol" , colnames(freesurfer))+3]

#Ctrl Shift C to uncomment chunk
# thickness22_ordering <- MBD(thickness22, plotting = FALSE)
# thickness26_ordering <- MBD(thickness26, plotting = FALSE)
# thickness31_ordering <- MBD(thickness31, plotting = FALSE)
# thickness36_ordering <- MBD(thickness36, plotting = FALSE)
# 
# area22_ordering <- MBD(area22, plotting = FALSE)
# area26_ordering <- MBD(area26, plotting = FALSE)
# area31_ordering <- MBD(area31, plotting = FALSE)
# area36_ordering <- MBD(area36, plotting = FALSE)
# 
# grayVol22_ordering <- MBD(grayVol22, plotting = FALSE)
# grayVol26_ordering <- MBD(grayVol26, plotting = FALSE)
# grayVol31_ordering <- MBD(grayVol31, plotting = FALSE)
# grayVol36_ordering <- MBD(grayVol36, plotting = FALSE)
# 
# cat("The most representative 22-25 year old for thickness is subject number",behavioralAge$`22-25`$Subject[thickness22_ordering$ordering[1]],"\n")
# cat("The most representative 26-30 year old for thickness is subject number",behavioralAge$`26-30`$Subject[thickness26_ordering$ordering[1]],"\n")
# cat("The most representative 31-35 year old for thickness is subject number",behavioralAge$`31-35`$Subject[thickness31_ordering$ordering[1]],"\n")
# cat("The most representative 36+ year old for thickness is subject number",behavioralAge$`36+`$Subject[thickness36_ordering$ordering[1]],"\n")
# 
# cat("The most representative 22-25 year old for area is subject number",behavioralAge$`22-25`$Subject[area22_ordering$ordering[1]],"\n")
# cat("The most representative 26-30 year old for area is subject number",behavioralAge$`26-30`$Subject[area26_ordering$ordering[1]],"\n")
# cat("The most representative 31-35 year old for area is subject number",behavioralAge$`31-35`$Subject[area31_ordering$ordering[1]],"\n")
# cat("The most representative 36+ year old for area is subject number",behavioralAge$`36+`$Subject[area36_ordering$ordering[1]],"\n")
# 
# cat("The most representative 22-25 year old for gray matter volume is subject number",behavioralAge$`22-25`$Subject[grayVol22_ordering$ordering[1]],"\n")
# cat("The most representative 26-30 year old for gray matter volume is subject number",behavioralAge$`26-30`$Subject[grayVol26_ordering$ordering[1]],"\n")
# cat("The most representative 31-35 year old for gray matter volume is subject number",behavioralAge$`31-35`$Subject[grayVol31_ordering$ordering[1]],"\n")
# cat("The most representative 36+ year old for gray matter volume is subject number",behavioralAge$`36+`$Subject[grayVol36_ordering$ordering[1]],"\n")

library(depthTests)

thicknessAll <- freesurfer[, grep("_Thck$", colnames(freesurfer))]
areaAll <- freesurfer[, grep("_Area", colnames(freesurfer))]
grayVolAll <- freesurfer[, grep("_GrayVol$", colnames(freesurfer))]

thicknessAll <- t(thicknessAll)
thickness22 <- t(thickness22)
thickness26 <- t(thickness26)
thickness31 <- t(thickness31)
thickness36 <- t(thickness36)

areaAll <- t(areaAll)
area22 <- t(area22)
area26 <- t(area26)
area31 <- t(area31)
area36 <- t(area36)

grayVolAll <- t(grayVolAll)
grayVol22 <- t(grayVol22)
grayVol26 <- t(grayVol26)
grayVol31 <- t(grayVol31)
grayVol36 <- t(grayVol36)

numOfPermutations <- 10 #increase for more accuracy

permutationTest(thicknessAll, thickness22, numOfPermutations)
permutationTest(thicknessAll, thickness26, numOfPermutations)
permutationTest(thicknessAll, thickness31, numOfPermutations)
permutationTest(thicknessAll, thickness36, numOfPermutations)

permutationTest(areaAll, area22, numOfPermutations)
permutationTest(areaAll, area26, numOfPermutations)
permutationTest(areaAll, area31, numOfPermutations)
permutationTest(areaAll, area36, numOfPermutations)

permutationTest(grayVolAll, grayVol22, numOfPermutations)
permutationTest(grayVolAll, grayVol26, numOfPermutations)
permutationTest(grayVolAll, grayVol31, numOfPermutations)
permutationTest(grayVolAll, grayVol36, numOfPermutations)

