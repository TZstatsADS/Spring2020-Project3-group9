set.seed(5)

library(caret)

# using cross validation to find the best weight of the ensemble model
folds<-createFolds(y=dat_train$emotion_idx,k=5)

train_cv1 <- dat_train[-folds[[1]],]
test_cv1 <- dat_train[folds[[1]],]

train_cv2 <- dat_train[-folds[[2]],]
test_cv2 <- dat_train[folds[[2]],]

train_cv3 <- dat_train[-folds[[3]],]
test_cv3 <- dat_train[folds[[3]],]

train_cv4 <- dat_train[-folds[[4]],]
test_cv4 <- dat_train[folds[[4]],]

train_cv5 <- dat_train[-folds[[5]],]
test_cv5 <- dat_train[folds[[5]],]

xgb_param_list <- list(
  xgb_para = list(alpha = 0.0198, lambda = 1.46, lambda_bias = 0.234),
  nround = 100
)
# 1st fold
# build xgboost model with the optimal parameters
xgb.cv1 <- train_xgb(train_cv1,xgb_param_list)
test.cv.xgb.pred1 <- test_xgb(xgb.cv1,test_cv1)
test.cv.xgb.pred1 <- test.cv.xgb.pred1[,-23]
# build svm model with the optimal parameters
svm.cv1 <- train_svm(train_cv1,probability = TRUE)
test.cv.svm.pred1 <- test_svm(svm.cv1,test_cv1,probability = TRUE) %>%
  attr("probabilities") %>%
  as.data.frame() 
test.cv.svm.pred1 <- test.cv.svm.pred1[, order(as.integer(colnames(test.cv.svm.pred1)))]
colnames(test.cv.svm.pred1) = colnames(test.cv.xgb.pred1)

# calculate the accuracy with weight 0.01-0.99
i_acc_1 <- c()

for (i in seq(0.01,0.99,0.01)){
  tmp1 <- i*test.cv.xgb.pred1
  tmp2 <- (1-i)*test.cv.svm.pred1
  test.cv.pred1 <- tmp1+tmp2
  test.cv.pred1$max_prob = max.col(test.cv.pred1, "last")
  i_acc_1 <- c(i_acc_1,mean(test.cv.pred1$max_prob == test_cv1$emotion_idx))
}


# 2nd fold
# build xgboost model with the optimal parameters
xgb.cv2 <- train_xgb(train_cv2,xgb_param_list)
test.cv.xgb.pred2 <- test_xgb(xgb.cv2,test_cv2)
test.cv.xgb.pred2 <- test.cv.xgb.pred2[,-23]
# build svm model with the optimal parameters
svm.cv2 <- train_svm(train_cv2,probability = TRUE)
test.cv.svm.pred2 <- test_svm(svm.cv2,test_cv2,probability = TRUE) %>%
  attr("probabilities") %>%
  as.data.frame() 
test.cv.svm.pred2 <- test.cv.svm.pred2[, order(as.integer(colnames(test.cv.svm.pred2)))]
colnames(test.cv.svm.pred2) = colnames(test.cv.xgb.pred2)

# calculate the accuracy with weight 0.01-0.99
i_acc_2 <- c()

for (i in seq(0.01,0.99,0.01)){
  tmp1 <- i*test.cv.xgb.pred2
  tmp2 <- (1-i)*test.cv.svm.pred2
  test.cv.pred2 <- tmp1+tmp2
  test.cv.pred2$max_prob = max.col(test.cv.pred2, "last")
  i_acc_2 <- c(i_acc_2,mean(test.cv.pred2$max_prob == test_cv2$emotion_idx))
}


# 3rd fold
# build xgboost model with the optimal parameters
xgb.cv3 <- train_xgb(train_cv3,xgb_param_list)
test.cv.xgb.pred3 <- test_xgb(xgb.cv3,test_cv3)
test.cv.xgb.pred3 <- test.cv.xgb.pred3[,-23]
# build svm model with the optimal parameters
svm.cv3 <- train_svm(train_cv3,probability = TRUE)
test.cv.svm.pred3 <- test_svm(svm.cv3,test_cv3,probability = TRUE) %>%
  attr("probabilities") %>%
  as.data.frame() 
test.cv.svm.pred3 <- test.cv.svm.pred3[, order(as.integer(colnames(test.cv.svm.pred3)))]
colnames(test.cv.svm.pred3) = colnames(test.cv.xgb.pred3)

# calculate the accuracy with weight 0.01-0.99
i_acc_3 <- c()

for (i in seq(0.01,0.99,0.01)){
  tmp1 <- i*test.cv.xgb.pred3
  tmp2 <- (1-i)*test.cv.svm.pred3
  test.cv.pred3 <- tmp1+tmp2
  test.cv.pred3$max_prob = max.col(test.cv.pred3, "last")
  i_acc_3 <- c(i_acc_3,mean(test.cv.pred3$max_prob == test_cv3$emotion_idx))
}


# 4th fold
# build xgboost model with the optimal parameters
xgb.cv4 <- train_xgb(train_cv4,xgb_param_list)
test.cv.xgb.pred4 <- test_xgb(xgb.cv4,test_cv4)
test.cv.xgb.pred4 <- test.cv.xgb.pred4[,-23]
# build svm model with the optimal parameters
svm.cv4 <- train_svm(train_cv4,probability = TRUE)
test.cv.svm.pred4 <- test_svm(svm.cv4,test_cv4,probability = TRUE) %>%
  attr("probabilities") %>%
  as.data.frame() 
test.cv.svm.pred4 <- test.cv.svm.pred4[, order(as.integer(colnames(test.cv.svm.pred4)))]
colnames(test.cv.svm.pred4) = colnames(test.cv.xgb.pred4)

# calculate the accuracy with weight 0.01-0.99
i_acc_4 <- c()

for (i in seq(0.01,0.99,0.01)){
  tmp1 <- i*test.cv.xgb.pred4
  tmp2 <- (1-i)*test.cv.svm.pred4
  test.cv.pred4 <- tmp1+tmp2
  test.cv.pred4$max_prob = max.col(test.cv.pred4, "last")
  i_acc_4 <- c(i_acc_4,mean(test.cv.pred4$max_prob == test_cv4$emotion_idx))
}

# 5th fold
# build xgboost model with the optimal parameters
xgb.cv5 <- train_xgb(train_cv5,xgb_param_list)
test.cv.xgb.pred5 <- test_xgb(xgb.cv5,test_cv5)
test.cv.xgb.pred5 <- test.cv.xgb.pred5[,-23]
# build svm model with the optimal parameters
svm.cv5 <- train_svm(train_cv5,probability = TRUE)
test.cv.svm.pred5 <- test_svm(svm.cv5,test_cv5,probability = TRUE) %>%
  attr("probabilities") %>%
  as.data.frame() 
test.cv.svm.pred5 <- test.cv.svm.pred5[, order(as.integer(colnames(test.cv.svm.pred5)))]
colnames(test.cv.svm.pred5) = colnames(test.cv.xgb.pred5)

# calculate the accuracy with weight 0.01-0.99
i_acc_5 <- c()

for (i in seq(0.01,0.99,0.01)){
  tmp1 <- i*test.cv.xgb.pred5
  tmp2 <- (1-i)*test.cv.svm.pred5
  test.cv.pred5 <- tmp1+tmp2
  test.cv.pred5$max_prob = max.col(test.cv.pred5, "last")
  i_acc_5 <- c(i_acc_5,mean(test.cv.pred5$max_prob == test_cv5$emotion_idx))
}


# overall performance on 5 folds
i_acc <- data.frame(i = seq(0.01,0.99,0.01),
                    acc <- (i_acc_1+i_acc_2+i_acc_3+i_acc_4+i_acc_5)/5)

i_acc[which.max(i_acc[,2]),1]

