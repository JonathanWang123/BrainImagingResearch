library(depthTools)

freesurferGendered <- split(freesurfer,freesurfer$Gender)
thicknessM <- freesurferGendered$M[, grep("_Thck$" , colnames(freesurfer))]
areaM <- freesurferGendered$M[, grep("_Area", colnames(freesurfer))]
grayVolM <- freesurferGendered$M[, grep("_GrayVol", colnames(freesurfer))]

thicknessF <- freesurferGendered$F[, grep("_Thck$" , colnames(freesurfer))]
areaF <- freesurferGendered$F[, grep("_Area", colnames(freesurfer))]
grayVolF <- freesurferGendered$F[, grep("_GrayVol", colnames(freesurfer))]

#Ctrl Shift C to uncomment chunk
# thicknessM_ordering <- MBD(thicknessM, plotting = FALSE)
# thicknessF_ordering <- MBD(thicknessF, plotting = FALSE)
# 
# areaM_ordering <- MBD(areaM, plotting = FALSE)
# areaF_ordering <- MBD(areaF, plotting = FALSE)
# 
# grayVolM_ordering <- MBD(grayVolM, plotting = FALSE)
# grayVolF_ordering <- MBD(grayVolF, plotting = FALSE)
# 
# cat("The most representative male for thickness is subject number",freesurferGendered$M$Subject[thicknessM_ordering$ordering[1]],"\n")
# cat("The most representative female for thickness is subject number",freesurferGendered$F$Subject[thicknessF_ordering$ordering[1]],"\n")
# cat("The most representative male for area is subject number",freesurferGendered$M$Subject[areaM_ordering$ordering[1]],"\n")
# cat("The most representative female for area is subject number",freesurferGendered$F$Subject[areaF_ordering$ordering[1]],"\n")
# cat("The most representative male for gray matter volume is subject number",freesurferGendered$M$Subject[grayVolM_ordering$ordering[1]],"\n")
# cat("The most representative female for gray matter volume is subject number",freesurferGendered$F$Subject[grayVolF_ordering$ordering[1]],"\n")

# Permutation
install.packages("devtools")
devtools::install_github("julia-wrobel/depthTests")
library(depthTests)
thicknessAll <- freesurfer[, grep("_Thck$", colnames(freesurfer))]
areaAll <- freesurfer[, grep("_Area", colnames(freesurfer))]
grayVolAll <- freesurfer[, grep("_GrayVol$", colnames(freesurfer))]

thicknessAll <- t(thicknessAll)
thicknessM <- t(thicknessM)
thicknessF <- t(thicknessF)

areaAll <- t(areaAll)
areaM <- t(areaM)
areaF <- t(areaF)

grayVolAll <- t(grayVolAll)
grayVolM <- t(grayVolM)
grayVolF <- t(grayVolF)

numOfPermutations <- 100 #increase for more accuracy

permutationTest(thicknessAll, thicknessM, numOfPermutations)
permutationTest(thicknessAll, thicknessF, numOfPermutations)

permutationTest(areaAll, areaM, numOfPermutations)
permutationTest(areaAll, areaF, numOfPermutations)

permutationTest(grayVolAll, grayVolM, numOfPermutations)
permutationTest(grayVolAll, grayVolF, numOfPermutations)




