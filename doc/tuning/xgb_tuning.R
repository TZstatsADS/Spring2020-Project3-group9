library(glmnet)
library(dplyr)
tmp_x <- select(dat_train,-emotion_idx)
tmp_y <- dat_train$emotion_idx
library(caret)

install.packages("xgboost")
library(xgboost)
library(mlr)

tmp_train<-list(data=as.matrix(tmp_x),label=as.numeric(tmp_y)-1)

dtrain<-xgb.DMatrix(data=tmp_train$data, label=tmp_train$label) 


numberOfClasses <- length(unique(tmp_train$label))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses,
                   nthread = 6)
nround    <- 80 # number of XGBoost rounds
cv.nfold  <- 5


set.seed(5)

lrn <- makeLearner("classif.xgboost",predict.type = "response")

# 1. set parameter space: try both gblinear gbtree booster
lrn$par.vals <- list(objective="multi:softprob", eval_metric="mlogloss", 
                     num_class = numberOfClasses,nthread = 6,
                     nrounds = 50L)

params <- makeParamSet( makeIntegerParam("max_depth",lower = 3L,upper = 10L),
                       makeNumericParam("min_child_weight",lower = 1L,upper = 10L), 
                       makeNumericParam("subsample",lower = 0.5,upper = 1), 
                       makeNumericParam("colsample_bytree",lower = 0.5,upper = 1),
                       makeNumericParam("eta",lower = 0.01,upper = 0.3),
                       makeNumericParam("gamma",lower = 0,upper = 0.3))
# set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)
ctrl <- makeTuneControlRandom(maxit = 50L)
traintask <- makeClassifTask (data = dat_train,target = "emotion_idx")
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)
# result shows that linear booster has better efficiency and accuracy

# 2. set parameter space: bigger parameters space of gblienar parameters
params <- makeParamSet(
  makeNumericParam("alpha",lower = 0,upper = 0.3),
  makeNumericParam("lambda",lower = 0,upper = 2),
  makeNumericParam("lambda_bias",lower = 0.1,upper = 0.5))

# set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)
ctrl <- makeTuneControlRandom(maxit = 50L)
traintask <- makeClassifTask (data = dat_train,target = "emotion_idx")
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)

# 3. set parameter space: smaller accoring to 2's result
params <- makeParamSet(
  makeNumericParam("alpha",lower = 0.015,upper = 0.05),
  makeNumericParam("lambda",lower = 0.5,upper = 1.6),
  makeNumericParam("lambda_bias",lower = 0.1,upper = 0.5))

# set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)
ctrl <- makeTuneControlRandom(maxit = 50L)
traintask <- makeClassifTask (data = dat_train,target = "emotion_idx")
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)

# 4. change feature selector: random
lrn$par.vals <- list(objective="multi:softprob", eval_metric="mlogloss", 
                     num_class = numberOfClasses,nthread = 6,
                     nrounds = 50L,booster = "gblinear",feature_selector = "random")

ctrl <- makeTuneControlRandom(maxit = 25L)
mytune_1 <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)
# no big difference

# xgb.cv with the optimal parameters
xgb_params3 <- list(objective = "multi:softprob",
                    eval_metric = "mlogloss",
                    num_class = numberOfClasses,
                    booster = "gblinear",
                    alpha=0.0198,
                    lambda=1.46,
                    lambda_bias=0.234 )

cv_model <- xgb.cv(params = xgb_params3,
                   data = dtrain, 
                   nrounds = 200,
                   nfold = 5,
                   verbose = FALSE,
                   prediction = TRUE)

OOF_prediction <- data.frame(cv_model$pred) %>%
  mutate(max_prob = max.col(., ties.method = "last"),
         label = tmp_train$label + 1)
head(OOF_prediction)

result <- confusionMatrix(factor(OOF_prediction$max_prob),
                          factor(OOF_prediction$label),
                          mode = "everything")
result

# xgb with the optimal parameters on whole training set and test on test set
xgb_params3 <- list(objective = "multi:softprob",
                    eval_metric = "mlogloss",
                    num_class = numberOfClasses,
                    booster = "gblinear",
                    alpha=0.0198,
                    lambda=1.46,
                    lambda_bias=0.234 )

xgb_model3 <- xgb.train(params = xgb_params3,
                        data = dtrain,
                        nrounds = 100,nthread = 6)

# confusion matrix of test set
test_pred3 <- predict(xgb_model3, newdata = dtest)
test_prediction3 <- matrix(test_pred3, nrow = numberOfClasses,
                           ncol=length(test_pred3)/numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(label = tmp_test$label + 1,
         max_prob = max.col(., "last"))
result <- confusionMatrix(factor(test_prediction3$max_prob),
                          factor(test_prediction3$label),
                          mode = "everything")



