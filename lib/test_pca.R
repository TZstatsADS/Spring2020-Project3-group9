########################################
### Classification with testing data ###
########################################

test_pca <- function(model, dat_test){
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  - processed features from testing images 
  ### Output: training model specification
  
  ### make predictions
  pred <- predict(model, dat_test[,1:(ncol(dat_train) - 1)])
  
  return(pred)
}