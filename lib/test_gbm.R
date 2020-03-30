########################################
### Classification with testing data ###
########################################

test_gbm <- function(model_best, feature_test = dat_test){

  ### Input: 
  ###  - the fitted classification model using training data
  ### (since knn does not need to train, we only specify k here)
  ###  - processed features from testing images 
  ### Output: training model specification
  
  ### load libraries
  library(gbm)
  
  ### set seed
  set.seed(5)
  seed <- .Random.seed
  
  ### make predictions
  pred <-  predict.gbm(model_best,
                       newdata = feature_test,
                       n.trees = 300)
  return(pred)
  
}