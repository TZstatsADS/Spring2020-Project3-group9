########################################
### Classification with testing data ###
########################################

test_lda <- function(model, dat_test){
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  - processed features from testing images 
  ### Output: training model specification
  
  ### load libraries
  library(MASS)
  
  ### set seed
  set.seed(5)
  seed <- .Random.seed
  
  ### make predictions
  pred <- predict(model, dat_test)
  pred <- pred$class
  
  return(pred)
}