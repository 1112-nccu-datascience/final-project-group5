args = commandArgs(trailingOnly = TRUE)
if(length(args)==0) {
  stop("USAGE: Rscript code/FinalProject.R --train data/train.csv --output results/result.csv")
}

# parse parameters
train_file <- NULL
output_file <- NULL
pca_tag <- NULL

i <- 1
while(i < length(args))
{
  if(args[i] == "--train"){
    train_file <- args[i+1]
    i <- i+1
  }else if(args[i] == "--output"){
    output_file <- args[i+1]
    i <- i+1
  }else if (args[i] == "--pca"){
    pca_tag <- args[i+1]
    i <- i+1
  }else{
    stop(paste("Unknown flag", args[i]), call.=FALSE)
  }
  i <- i+1
}

# check missing flag
if(is.null(train_file)) {
  stop("Missing flag --train", call.=FALSE)
}else if(is.null(output_file)) {
  stop("Missing flag --output", call.=FALSE)
}

# 測試用
train_file <- "data/train.csv.zip"
output_file <- "results/result.csv"
pca_tag <- "yes"

#options(repos = c(CRAN = "https://cran.r-project.org"))
#install.packages("ggplot2")
#install.packages("dplyr")
#install.packages("corrplot")
#install.packages("caTools")
#install.packages("xgboost")
#install.packages("e1071")
#install.packages("pROC")
#install.packages("factoextra")
#install.packages("caret")

library(ggplot2)
library(dplyr)
library(ROSE)
library(corrplot)
library(caTools)
library(xgboost)
library(e1071)
library(pROC)
library(factoextra)
library(caret)
################## Feature ##################
# binary (bin) or categorical (cat) variables.
# "Ind" is related to individual or driver, 
# "reg" is related to region, 
# “car" is related to car itself and 
# "calc" is an calculated feature.
#############################################


### Read data
train_file <- unzip(train_file, exdir = "./data")
df <- read.csv(train_file[1] , header = T, sep = "," , row.names = 1)
df[df == -1] <- NA
summary(df)
#View(df)


### EDA - part1
## Count NA of each features
na_counts <- colSums(is.na(df))
na_columns <- names(na_counts[na_counts > 0])
na_df <- data.frame(na_counts[na_counts > 0], na_columns)

ggplot(data = na_df, aes(x = na_columns, y = na_counts[na_counts > 0])) +
  geom_bar(stat = "identity", fill = "lightblue") +
  geom_text(aes(label = na_counts[na_counts > 0]), vjust = -0.5, color = "darkblue", size = 3) +
  labs(title = "Count of Missing Values", x = "Feature", y = "Count of NAs") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(labels = scales::comma)


## Target Plot - count 
ggplot(data = df, aes(x = as.factor(target))) +
  geom_bar(fill = "lightblue") +
  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5) +
  labs(title = 'Distribution of Target Class',
       x = "Target", 
       y = "Target Count") +
  scale_y_continuous(labels = scales::comma)


## Features Correlation - with NA
df %>%
  select(-target) %>%
  cor(use="complete.obs", method = "spearman") %>%
  corrplot( method = "circle", type = "lower", tl.cex = 0.5,
            tl.col = "black",  diag=FALSE)


### Data processing - NA
## 資料前處理:category用眾數 / binary or others用平均數
df[is.na(df$ps_ind_02_cat), "ps_ind_02_cat"] <- as.numeric(names(table(df$ps_ind_02_cat))[which.max(table(df$ps_ind_02_cat))])    
df[is.na(df$ps_ind_04_cat), "ps_ind_04_cat"] <- as.numeric(names(table(df$ps_ind_04_cat))[which.max(table(df$ps_ind_04_cat))])
df[is.na(df$ps_ind_05_cat), "ps_ind_05_cat"] <- as.numeric(names(table(df$ps_ind_05_cat))[which.max(table(df$ps_ind_05_cat))])
df[is.na(df$ps_car_01_cat), "ps_car_01_cat"] <- as.numeric(names(table(df$ps_car_01_cat))[which.max(table(df$ps_car_01_cat))])
df[is.na(df$ps_car_02_cat), "ps_car_02_cat"] <- as.numeric(names(table(df$ps_car_02_cat))[which.max(table(df$ps_car_02_cat))])
df[is.na(df$ps_car_03_cat), "ps_car_03_cat"] <- as.numeric(names(table(df$ps_car_03_cat))[which.max(table(df$ps_car_03_cat))])
df[is.na(df$ps_car_05_cat), "ps_car_05_cat"] <- as.numeric(names(table(df$ps_car_05_cat))[which.max(table(df$ps_car_05_cat))])
df[is.na(df$ps_car_07_cat), "ps_car_07_cat"] <- as.numeric(names(table(df$ps_car_07_cat))[which.max(table(df$ps_car_07_cat))])
df[is.na(df$ps_car_09_cat), "ps_car_09_cat"] <- as.numeric(names(table(df$ps_car_09_cat))[which.max(table(df$ps_car_09_cat))])
df[is.na(df$ps_car_11_cat), "ps_car_11_cat"] <- as.numeric(names(table(df$ps_car_11_cat))[which.max(table(df$ps_car_11_cat))])
# df[is.na(df$ps_car_11), "ps_car_11"] <- as.numeric(names(table(df$ps_car_11))[which.max(table(df$ps_car_11))])
df[is.na(df$ps_reg_03), "ps_reg_03"] <- summary(df$ps_reg_03)[4]
df[is.na(df$ps_car_11), "ps_car_11"] <- summary(df$ps_car_11)[4]
df[is.na(df$ps_car_12), "ps_car_12"] <- summary(df$ps_car_12)[4]
df[is.na(df$ps_car_14), "ps_car_14"] <- summary(df$ps_car_14)[4]

### EDA - part2
## features correlation -after dealing NA
df %>%
  select(-target) %>%
  cor(use="complete.obs", method = "spearman") %>%
  corrplot( method = "circle", type = "lower", tl.cex = 0.5,
            tl.col = "black",  diag=FALSE)

#### Set random seed
set.seed(6666)

### Split Train/Test Data
split <- sample.split(df, SplitRatio = 0.8)
X_train <- subset(df, split == TRUE)
X_test <- subset(df, split == FALSE)

### Scale
scaler <- preProcess(X_train[-1], method = c("center", "scale"))
X_train <- cbind(X_train[1], predict(scaler, X_train[-1]))
X_test <- predict(scaler, X_test)

### 處理資料不平衡，調整為 4:1
# X_train <- ovun.sample(target~., data = X_train, method = "over", p = 0.2)$data
### 處理資料不平衡，調整為 1:1 oversampling
# X_train <- ovun.sample(target~., data = X_train, method = "over")$data
### 處理資料不平衡，調整為 1:1 undersampling & oversampling
# X_train <- ovun.sample(target~., data = X_train, method = "both")$data
### 處理資料不平衡，調整為 1:1 undersampling
# X_train <- ovun.sample(target~., data = X_train, method = "under")$data
table(X_train$target)

### PCA
if (pca_tag == "yes"){
  pca <- preProcess(X_train[-1], method = "pca", pcaComp = 47)
  X_train <- cbind(X_train[1], predict(pca, X_train[-1]))
  X_test <- predict(pca, X_test)
}

### Evaluation - GINI
normalizedGini <- function(aa, pp) {
  Gini <- function(a, p) {
    if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
    temp.df <- data.frame(actual = as.numeric(a), pred = as.numeric(p), range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
    accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
    gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
    sum(gini.sum) / length(a)
  }
  Gini(aa,pp) / Gini(aa,aa)
}

### Evaluation - AUC & Confusion Matrix
auc_and_cm <- function(test_target, pred_target){
  # AUC & ROC
  auc <- roc(test_target, pred_target, auc=TRUE) 
  print(auc)
  threshold <- coords(auc, x='best', input='threshold', best.method='youden')$threshold
  # Confusion Matrix
  caret::confusionMatrix(factor(test_target), 
                         factor(ifelse(pred_target>threshold, 1, 0)),
                         positive = "1")  
}


### XGBoost
print("Start XGBoost...")
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss"
)

xgb_list <- list()
nrounds_list<-list()
i=10
while (i != 50){
  print(i)
  xgb_model <- xgboost(data = as.matrix(X_train[-1]),
                       label = X_train$target,
                       params = params,
                       nrounds = i)
  nrounds_list <- append(nrounds_list, i)
  xgb_pred <- predict(xgb_model, newdata = as.matrix(X_test[-1]))
  print(paste("XGBoost: ", normalizedGini(X_test$target, xgb_pred)))
  xgb_list <- append(xgb_list, round(normalizedGini(X_test$target, xgb_pred),3))
  i <- i+10
}
# plot the gini grade of parameters
xgb_data <- data.frame(parameters = unlist(nrounds_list), xgb_list = unlist(xgb_list))
ggplot(xgb_data, aes(x = parameters, y = xgb_list)) +
  geom_line(color = "lightblue", linewidth = 1) +  # 設定折線顏色和粗細
  geom_text(aes(label = xgb_list), vjust = -1.5, color = "darkblue", size = 4) +
  labs(x = "NRounds", y = "Gini", title = "XGBoost Model Training - PCA") +
  theme_minimal() +                         # 使用簡約風格的主題
  theme(plot.title = element_text(size = 12, face = "bold"),  # 設定標題樣式
        axis.text = element_text(size = 10),                  # 設定軸標籤樣式
        axis.title = element_text(size = 10, face = "bold"))  # 設定軸標籤樣式

### XGBoost Select Best N Rounds
best_nrounds <- as.numeric(nrounds_list[which.max(xgb_list)])
xgb_model <- xgboost(data = as.matrix(X_train[-1]),
                 label = X_train$target,
                 params = params,
                 nrounds = best_nrounds)
xgb_pred <- predict(xgb_model, newdata = as.matrix(X_test[-1]))
print(paste("XGBoost: ", normalizedGini(X_test$target, xgb_pred)))
# AUC & CM
auc_and_cm(X_test$target, xgb_pred)


### Naive Bayes
print("Start Naive Bayes...")
nb_model <- naiveBayes(target ~ ., data = X_train)
nb_pred <- predict(nb_model, newdata = X_test[-1], type = 'raw')
print(paste("NaiveBayes: ", round(normalizedGini(X_test$target, nb_pred[,2]), 3)))
# AUC & CM
auc_and_cm(X_test$target, as.numeric(nb_pred[,2]))


### Logistic
print("Start Logistic...")
logistic_model <- glm(target ~ ., 
                      data = X_train, 
                      family = binomial)
logistic_pred <- predict(logistic_model, newdata = X_test[-1])
print(paste("Logistic: ", normalizedGini(X_test$target, logistic_pred)))
# AUC & CM
auc_and_cm(X_test$target, logistic_pred)


### Null Model
print("Start NULL Model...")
shuffle_X_train <- X_train
shuffle_X_train$target <- sample(shuffle_X_train$target)
null_model <- xgboost(data = as.matrix(shuffle_X_train[-1]),
                     label = shuffle_X_train$target,
                     params = params,
                     nrounds = best_nrounds)
null_pred <- predict(null_model, newdata = as.matrix(X_test[-1]))
print(paste("Null Model: ", normalizedGini(X_test$target, null_pred)))
# AUC & CM
auc_and_cm(X_test$target, null_pred)


result <- data.frame(matrix(ncol=2, nrow=0))
colnames(result) <- c("Model", "NormalGini")
result[nrow(result)+1,] <- c("XGBoost", round(normalizedGini(X_test$target, xgb_pred), 4))
result[nrow(result)+1,] <- c("NaiveBayes", round(normalizedGini(X_test$target, nb_pred[,2]), 4))
result[nrow(result)+1,] <- c("Logistic", round(normalizedGini(X_test$target, logistic_pred), 4))
result[nrow(result)+1,] <- c("NullModel", round(normalizedGini(X_test$target, null_pred), 4))
#View(result)

write.csv(result, output_file, row.names=FALSE, quote=FALSE)

### select importance feature
# model <- xgboost(data = as.matrix(X_train[-1]),
#                  label = X_train$target,
#                  params = params,
#                  nrounds = 20)
# pred <- predict(model, newdata = as.matrix(X_test[-1]))
# print(paste("XGBoost: ", normalizedGini(X_test$target, pred)))
# 
# importance <- xgb.importance(model = model)
# 
# bar_plot <- ggplot(importance, aes(x = reorder(Feature, -Gain), y = Gain)) +
#   geom_bar(stat = "identity", fill = "skyblue") +
#   labs(x = "Feature", y = "Importance") +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1))
# print(bar_plot)
# 
# selected_features <- importance$Feature[importance$Gain > 0.01]
# print(selected_features)
# 
# X_train_filtered <- X_train[, c("target", selected_features)]
# X_test_filtered <- X_test[, c("target", selected_features)]
# 
# model <- xgboost(data = as.matrix(X_train_filtered[-1]),
#                  label = X_train_filtered$target,
#                  params = params,
#                  nrounds = 20)
# pred <- predict(model, newdata = as.matrix(X_test_filtered[-1]))
# print(paste("XGBoost: ", normalizedGini(X_test_filtered$target, pred)))