# Natural Language Processing

# Importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

# Cleaning the texts
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)


# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked


# # Importing the dataset
# dataset = read.csv('Social_Network_Ads.csv')
# dataset = dataset[3:5]


# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


## RF
# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)


# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])


# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)



## NB
# Fitting Naive Bayes to the Training set
# install.packages('e1071')
# library(e1071)
# classifier = naiveBayes(x = training_set[-692],
#                         y = training_set$Liked)
#
# # Predicting the Test set results
# y_pred = predict(classifier, newdata = test_set[-692])
#
#
# # Making the Confusion Matrix
# cm = table(test_set[, 692], y_pred)



## LR
# Fitting Logistic Regression to the Training set
# classifier = glm(formula = Liked ~ .,
#                  family = binomial,
#                  data = training_set)
#
#
# # Predicting the Test set results
# prob_pred = predict(classifier, type = 'response', newdata = test_set[-692]) # give  us probabilities listed in a single vector
# y_pred = ifelse(prob_pred > 0.5, 1, 0)
#
#
# # Making the Confusion Matrix
# cm = table(test_set[, 692], y_pred > 0.5)



## KNN
# Fitting K-NN to the Training set and Predicting the Test set results
# library(class)
# y_pred = knn(train = training_set[, -692],
#              test = test_set[, -692],
#              cl = training_set[, 692],
#              k = 5,
#              prob = TRUE)
#
#
# # Making the Confusion Matrix
# cm = table(test_set[, 692], y_pred)



## SVM
# Fitting SVM to the Training set
# install.packages('e1071')
# library(e1071)
# classifier = svm(formula = Liked ~ .,
#                  data = training_set,
#                  type = 'C-classification',
#                  kernel = 'linear')
#
#
# # Predicting the Test set results
# y_pred = predict(classifier, newdata = test_set[-692])
#
#
# # Making the Confusion Matrix
# cm = table(test_set[, 692], y_pred)



## KSVM
# Fitting Kernel SVM to the Training set
# install.packages('e1071')
# library(e1071)
# classifier = svm(formula = Liked~ .,
#                  data = training_set,
#                  type = 'C-classification',
#                  kernel = 'radial')
#
#
# # Predicting the Test set results
# y_pred = predict(classifier, newdata = test_set[-692])
#
#
# # Making the Confusion Matrix
# cm = table(test_set[, 692], y_pred)



## DT
# Fitting Decision Tree Classification to the Training set
# install.packages('rpart')
# library(rpart)
# classifier = rpart(formula = Liked ~ .,
#                    data = training_set)
#
#
# # Predicting the Test set results
# # y_pred = predict(classifier, newdata = test_set[-3])
# y_pred = predict(classifier, newdata = test_set[-692], type = 'class')
#
#
# # Making the Confusion Matrix
# cm = table(test_set[, 692], y_pred)

