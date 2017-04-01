train = read.csv('/tmp/train.csv')
dim(train) # 2999 records of 21 variables
str(train) # Categorical and numeric factors, Target is column 21
sum(is.na(train)) #There are no missing values in the train set
X=train[,-(dim(train)[2])] # TrainX has first 20 cols of train
Y=train[,(dim(train))[2]]  # TtrainY has last col of train (target)
summary(X)
hist(age)
hist(campaign)
