###########################################################
### Train a classification model with training features ###
###########################################################
train_gbm <- function(feature_df){
  ### Train a classfication model using processed features from training images
  
  ### Input:
  ### - a data frame containing features and labels
  ### Output: trained model
  
  ### load libraries
  library(gbm)
  
  ### set seed
  set.seed(5)
  seed <- .Random.seed
  
  ### Train with gbm
  
  model <- gbm(emotion_idx~., data = feature_df, 
               distribution = "multinomial",
               n.trees = 300,
               shrinkage = 0.15, 
               interaction.depth = 2,
               n.minobsinnode = 10)
  
  return(model)
}

