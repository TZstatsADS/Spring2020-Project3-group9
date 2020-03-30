###########################################################
### Train a classification model with training features ###
###########################################################
train_rf <- function(feature_df = pairwise_data, par = NULL)
{
  ### Train an random forest model using processed features from training images
  
  ### Input:
  ### - a data frame containing features and labels
  ### - a parameter list
  ### Output: trained model
  
  ### load libraries
  library(randomForest)
  
  ### set seed
  set.seed(5)
  seed <- .Random.seed
  
  if (is.null(par)) 
  {
    mtry = 77
  }
  else
  {
    mtry = par$mtry
  }
  
  rf_model <- randomForest(emotion_idx~., data = dat_train, mtry = mtry)
  
  return(model = rf_model)
}