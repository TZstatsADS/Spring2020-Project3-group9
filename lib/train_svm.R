###########################################################
### Train a classification model with training features ###
###########################################################
train_svm <- function(feature_df = pairwise_data, cost=0.02, probability=FALSE){
  ### Train a classfication model using processed features from training images
  
  ### Input:
  ### - a data frame containing features and labels
  ### Output: trained model
  
  ### load libraries
  library(e1071)
  
  ### set seed
  set.seed(5)
  seed <- .Random.seed
  
  ### Train with SVM
  
  model <- svm(emotion_idx~., data = feature_df,type='C',kernel='linear', cost = cost, probability=probability) 
  
  return(model)
}