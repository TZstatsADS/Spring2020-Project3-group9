###################################################################
############# ensemble model of SVM and XGBoost ###################
###################################################################
simple_ensemble <- function(xgb_prob_mat, svm_prob_mat,weight = 0.51){
  ### Return a weighted average of two models prediction
  
  ### Input:
  ### - two probability matrix
  ### - a weight
  ### Output: a weighted average of the two probability with the class
  
  tmp1 <- weight*xgb_prob_mat
  tmp2 <- (1-weight)*svm_prob_mat
  pred <- tmp1+tmp2
  pred$max_prob = max.col(pred, "last")

  return(pred)
}
