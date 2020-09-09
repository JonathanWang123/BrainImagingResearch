install.packages("neurohcp")
library(neurohcp)

#Run in Amarel
#Make sure you create the Research and HCP folders in your scratch directory before running
#Directory should look like /scratch/[netid]/Research/HCP/
#Otherwise you can modify the paths in the code in lines 21-25, 36 and 55
#Input your netid, access key and secret key as strings below

#Download the subjIDs.rds file and copy it into amarel by typing the following into command prompt 
#scp Path/To/The/File/subjIDs.rds [netid]@amarel.rutgers.edu:/scratch/[netid]/Research/subjIDs.rds (Make sure you have a Research folder)
#modify Path/To/The/File and replace [netid] with your netid (ex: jc2549)

NetID <- ""
access_key <- ""
secret_key <- ""

eval(parse(text = paste("subjIDs <- readRDS(file = \"/scratch/",NetID,"/Research/subjIDs.rds\")",sep = ""))) 

for(i in 1:length(subjIDs)){
  eval(parse(text = paste("dir.create(\"/scratch/",NetID,"/Research/HCP/",subjIDs[i],"\")", sep = "")))
  eval(parse(text = paste("dir.create(\"/scratch/",NetID,"/Research/HCP/",subjIDs[i],"/MNINonLinear\")", sep = "")))
  eval(parse(text = paste("dir.create(\"/scratch/",NetID,"/Research/HCP/",subjIDs[i],"/MNINonLinear/Results\")", sep = "")))
  eval(parse(text = paste("dir.create(\"/scratch/",NetID,"/Research/HCP/",subjIDs[i],"/MNINonLinear/Results/rfMRI_REST1_LR\")", sep = "")))
  eval(parse(text = paste("dir.create(\"/scratch/",NetID,"/Research/HCP/",subjIDs[i],"/MNINonLinear/Results/rfMRI_REST1_RL\")", sep = "")))
}

set_aws_api_key(access_key = access_key, secret_key = secret_key, default_region = "us-east-1", error = TRUE)

LRskipped <- RLskipped <- NULL
for(j in 81:length(subjIDs)){
  if (have_aws_key()){
    cat("Subj: ",subjIDs[j]," | ",j,"/",length(subjIDs),"\n",sep = "")
    #LR
    eval(parse(text = paste("path_to_file <- \"HCP/",subjIDs[j],"/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii\"", sep = "")))
    eval(parse(text = paste("destfile <- \"/scratch/",NetID,"/Research/HCP/",subjIDs[j],"/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii\"", sep = "")))
    tryCatch(download_hcp_file(path_to_file = path_to_file, destfile = destfile, verbose = FALSE, error = TRUE),
             error = function(e){
               message("subject ID ",subjIDs[j], " HCP failed, trying HCP_900")
               eval(parse(text = paste("path_to_file <- \"HCP_900/",subjIDs[j],"/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii\"", sep = "")))
               tryCatch(download_hcp_file(path_to_file = path_to_file, destfile = destfile, verbose = FALSE, error = TRUE),
                        error = function(e){
                          message("subject ID ",subjIDs[j], " HCP 900 failed, trying HCP_1200")
                          eval(parse(text = paste("path_to_file <- \"HCP_1200/",subjIDs[j],"/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii\"", sep = "")))
                          tryCatch(download_hcp_file(path_to_file = path_to_file, destfile = destfile, verbose = FALSE, error = TRUE),
                                   error = function(e){
                                     message("subject ID ",subjIDs[j], " LR HCP 1200 failed, skipping subject")
                                     LRskipped <- c(LRskipped, subjIDs[j])
                                   })
                        })
             })
    
    #RL
    eval(parse(text = paste("path_to_file <- \"HCP/",subjIDs[j],"/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_hp2000_clean.dtseries.nii\"", sep = "")))
    eval(parse(text = paste("destfile <- \"/scratch/",NetID,"/Research/HCP/",subjIDs[j],"/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_hp2000_clean.dtseries.nii\"", sep = "")))
    tryCatch(download_hcp_file(path_to_file = path_to_file, destfile = destfile, verbose = FALSE, error = TRUE),
             error = function(e){
               eval(parse(text = paste("path_to_file <- \"HCP_900/",subjIDs[j],"/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_hp2000_clean.dtseries.nii\"", sep = "")))
               tryCatch(download_hcp_file(path_to_file = path_to_file, destfile = destfile, verbose = FALSE, error = TRUE),
                        error = function(e){
                          eval(parse(text = paste("path_to_file <- \"HCP_1200/",subjIDs[j],"/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_hp2000_clean.dtseries.nii\"", sep = "")))
                          tryCatch(download_hcp_file(path_to_file = path_to_file, destfile = destfile, verbose = FALSE, error = TRUE),
                                   error = function(e){
                                     message("subject ID ",subjIDs[j], " RL HCP 1200 failed, skipping subject")
                                     RLskipped <- c(RLskipped, subjIDs[j]) 
                                   })
                        })
             })
  }
}