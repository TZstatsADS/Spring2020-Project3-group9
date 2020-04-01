########################################
### Classification with testing data ###
########################################

test_lr <- function(model, lambda, dat_test){
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  - processed features from testing images 
  ### Output: training model specification
  
  ### load libraries
  library(glmnet)
  
  ### set seed
  set.seed(5)
  seed <- .Random.seed
  
  ### make predictions
  pred <- predict(model, as.matrix(dat_test[,1:500]), s = model$lambda, type = "class")
  
  return(pred)
}