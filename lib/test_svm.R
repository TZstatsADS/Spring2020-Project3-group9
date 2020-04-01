########################################
### Classification with testing data ###
########################################

test_svm <- function(model, dat_test, probability=FALSE){
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  - processed features from testing images 
  ### Output: training model specification
  
  ### load libraries
  library(e1071)
  
  ### set seed
  set.seed(5)
  seed <- .Random.seed
  
  ### make predictions
  pred <- predict(model, dat_test, probability=probability)
  
  return(pred)
}