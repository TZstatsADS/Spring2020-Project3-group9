###################################################################
### Train a XGBoost classification model with training features ###
###################################################################
train_xgb <- function(feature_df = dat_train, par = NULL){
  ### Train an SVM model using processed features from training images
  
  ### Input:
  ### - a data frame containing features and labels
  ### - a parameter list
  ### Output: trained model
  
  ### load libraries
  library(xgboost)
  
  ### set seed
  set.seed(5)
  seed <- .Random.seed
  
  ### prepare dataset
  x <- feature_df[,-which(names(feature_df) == 'emotion_idx')]
  y <- feature_df$emotion_idx
  tmp <- list(data=as.matrix(x),label=as.numeric(y)-1)
  dtrain<-xgb.DMatrix(data=tmp$data, label=tmp$label)
  
  ### Train with XGB
  if(is.null(par)){
    par <- list(xgb_para = list(objective = "multi:softprob",
                 eval_metric = "mlogloss",
                 num_class = length(unique(tmp$label)),
                 booster = "gblinear",
                 alpha=0.02, lambda = 1.5, lambda_bias = 0.2),
                nround = 100)
  }
  else{
    par$xgb_para = c(par$xgb_para,
                     booster = "gblinear",
                     objective = "multi:softprob",
                     eval_metric = "mlogloss",
                     num_class = length(unique(tmp$label)))
  }
  xgb_model <- xgb.train(params = par$xgb_para,
                         data = dtrain,
                         nrounds = par$nround,nthread = 6)
  
  return(model = xgb_model)
}