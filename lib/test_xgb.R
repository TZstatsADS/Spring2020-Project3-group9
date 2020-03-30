#############################################################
### Classification with testing data and the XGBoost model###
#############################################################
test_xgb <- function(xgb_model, dat_test)
{
  ### Input: 
  ###  - the fitted classification model using training data
  ###  - processed features from testing images 
  ### Output: training model specification
  
  ### load libraries
  library(xgboost)
  library(dplyr)
  
  ### set seed
  set.seed(5)
  seed <- .Random.seed
  
  if('emotion_idx' %in% names(dat_test)){
    ### prepare dataset
    x <- dat_test[,-which(names(dat_test) == 'emotion_idx')]
    y <- dat_test$emotion_idx
    tmp <- list(data=as.matrix(x),label=as.numeric(y)-1)
    dtest<-xgb.DMatrix(data=tmp$data, label=tmp$label) 
    numberOfClasses = length(unique(tmp$label))}
  else{
    dtest<-xgb.DMatrix(data=as.matrix(dat_test)) 
    numberOfClasses = xgb_model$params$num_class
  }
                     
  ### make predictions
  pred <- predict(xgb_model, newdata = dtest)
  prediction <- matrix(pred, nrow = numberOfClasses,
                             ncol=length(pred)/numberOfClasses) %>%
    t() %>%
    data.frame() %>%
    mutate(max_prob = max.col(., "last"))
  return(prediction)
}

