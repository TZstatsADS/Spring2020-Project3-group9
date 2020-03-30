################################################
### Train a PCA model with training features ###
################################################
train_pca <- function(feature_df = pairwise_data, par = NULL){
  ### Train a classfication model using processed features from training images
  
  ### Input:
  ### - a data frame containing features
  ### Output: trained model
  
  ### Train with PCA
  
  ## PCA for training features
  model <- prcomp(feature_df)
  
  return(model)
}

