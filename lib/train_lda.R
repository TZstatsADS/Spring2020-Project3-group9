###########################################################
### Train a classification model with training features ###
###########################################################
train_lda <- function(feature_df = pairwise_data, par = NULL){
  ### Train an SVM model using processed features from training images
  
  ### Input:
  ### - a data frame containing features and labels
  ### - a parameter list
  ### Output: trained model
  
  ### load libraries
  library(MASS)
  
  ### set seed
  set.seed(5)
  seed <- .Random.seed
  
  model <- lda(emotion_idx~., data = feature_df)
  
  return(model)
}

