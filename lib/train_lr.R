###########################################################
### Train a classification model with training features ###
###########################################################
train_lr <- function(feature_df = pairwise_data, par = NULL){
  ### Train an SVM model using processed features from training images
  
  ### Input:
  ### - a data frame containing features and labels
  ### - a parameter list
  ### Output: trained model
  
  ### load libraries
  library(glmnet)
  
  ### set seed
  set.seed(5)
  seed <- .Random.seed
  
  model <- cv.glmnet(as.matrix(feature_df[, 1:500]), 
                     factor(feature_df$emotion_idx),
                     data = feature_df, 
                     family = "multinomial", 
                     type.measure = "class")
  
  return(model)
}
