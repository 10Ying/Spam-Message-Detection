rm(list = ls())

library(formattable)        # install.packages("formattable")    
library(corrplot)           # install.packages("corrplot")  
library(fmsb)               # install.packages("fmsb") 
library(gridBase)           # install.packages("gridBase") 
library(ggplot2)            # install.packages("ggplot2") 
library(reshape2)           # install.packages("reshape2") 
library(magrittr)           # install.packages("magrittr") 
library(Rmisc)              # install.packages("Rmisc") 
library(dplyr)              # install.packages("dplyr") 
library(wordcloud)          # install.packages("wordcloud")
library(dplyr)              # install.packages("dplyr")
library(stringr)            # install.packages("stringr")
library(randomForest)       # install.packages("randomForest")
library(tm)                 # install.packages("tm")
library(ROSE)               # install.packages("ROSE")
library(magrittr)           # install.packages("magrittr")
library(caret)              # install.packages("caret")
library(SnowballC)          # install.packages("SnowballC")
library(RColorBrewer)       # install.packages("RColorBrewer")
library(neuralnet)          # install.packages("neuralnet")
library(class)              # install.packages("class") 
library(e1071)              # install.packages("e1071")
library(kknn)               # install.packages("kknn") 
library(neuralnet)          # install.packages("neuralnet")
library(rpart)              # install.packages("rpart")
library(rpart.plot)         # install.packages("rpart.plot")
library(rattle)             # install.packages("rattle")
library(RColorBrewer)       # install.packages("RColorBrewer") 
library(C50)                # install.packages("C50") 

# Read data 'stringsAsFactors = F' means no strings will be converted to factor
sms <- read.csv("/Users/ying/Desktop/513_project/Items/spam.csv", stringsAsFactors = F, header = TRUE)
# Preserve the first two columns of data
sms <- sms[, c(1, 2)]
# Rename column name
names(sms) <- c('label', 'message')
str(sms) # rough observation data set

# enc2utf8() converts the contents of the message column to utf-8 format
# iconv(sub = 'byte') specifies that characters that cannot be converted are replaced in their hexadecimal form
sms$message <- sapply(sms$message, function(x) iconv(enc2utf8(x), sub = "byte"))

### Text processing: the first step
sms %<>% mutate(msg_length = str_length(message),
                pct_caps = str_count(message, "[A-Z]") / msg_length,
                pct_digits = str_count(message,"[0-9]") / msg_length,
                num_exclamations = str_count(message, "!"),
                num_caps = str_count(message, "[A-Z]"),
                num_digits = str_count(message,"[0-9]"),
                numeric_label = as.numeric(as.factor(label)))

## Compare the remodeling data of different SMS
sms %>% 
  select(-message) %>%                                  # Eliminate the message column, just operate on reshaped data and categories
  group_by(label) %>%                                   # Group by label
  summarise_all(function(x) round(mean(x), 3)) %>%      # Count the average of each column and keep three decimals 
  formattable(align = 'l')                              # Generate a data table, left aligned

## draw correlation coefficient map
# tl.pos = 'lt' Specify the column name on the left and top of the diagram
# diag = 'l'    Specify the diagonal above to fill with values
corr <- cor(sms[-c(1, 2)])                              # Calculate the correlation coefficient of the data
corrplot.mixed(corr, tl.pos = 'lt', diag = 'l')         # Visual correlation coefficient
sms$numeric_label <- NULL                               # Remove the numeric_label column

## Create a text handler
create_corpus <- function(x) {
    result_corpus <- VCorpus(VectorSource(x)) %>%       # Convert data into a corpus
    tm_map(tolower, lazy = T) %>%                       # Convert all letters to lowercase letters
    tm_map(PlainTextDocument) %>%                       # Convert text to plain text documents
    tm_map(removePunctuation) %>%                       # Delete punctuation
    tm_map(removeWords, stopwords("english")) %>%       # Delete stop words
    tm_map(removeNumbers) %>%                           # Delete digits
    tm_map(stripWhitespace) %>%                         # Delete more spaces(keep up tp only one space)
    tm_map(stemDocument, lazy = T)                      # Extracting stems using Porter's stemming algorithm
  return(result_corpus)
}


# Text processing of normal text messages(ham)
corpus_ham <- create_corpus(sms$message[sms$label == "ham"]) 
# Text processing of junk text messages(spam)
corpus_spam <- create_corpus(sms$message[sms$label == "spam"])

## Drawing word clouds
# min.freq = 10, Words with word frequencies below 10 will not be drawn
# max.words = 1000, the word cloud map shows up to 1000 words.
# scale, Word size range
# random.order = F, Draw word cloud map according to word frequency descending

#####        ATTENTION: Click ‘Clear all Plots’ in the lower right corner on the image display area 
#####        before drawning each word cloud every time.
wordcloud(sms[sms$label == 'ham', ]$message, min.freq = 10, max.words = 1000, scale=c(3, 0.5), 
          random.order = TRUE, colors = c("blue", "blue", "blue", "blue"))

#####        ATTENTION: Click ‘Clear all Plots’ in the lower right corner on the image display area 
#####        before drawning each word cloud every time.
wordcloud(sms[sms$label == 'spam', ]$message, min.freq = 10, max.words = 1000, scale=c(5, 0.5), 
          random.order = TRUE, colors = c("indianred1", "indianred2", "indianred3", "indianred4"))

# Encapsulation function - calculate the number of occurrences of each word
create_term_frequency_counts <- function(dtm) {
  m <- as.matrix(dtm)
  v <- sort(colSums(m), decreasing = TRUE)
  d <- data.frame(word = names(v), freq = v, stringsAsFactors = FALSE)
  return(d)
}

# Create sparse matrix for ham
dtm_ham <- DocumentTermMatrix(corpus_ham) %>%
  removeSparseTerms(0.99) # Dimensionality reduction, which eliminats low frequency words in ham

# Create sparse matrix for spam
dtm_spam <- DocumentTermMatrix(corpus_spam) %>%
  removeSparseTerms(0.979) # Dimensionality reduction, which eliminats low frequency words in spam

# as.matrix(dtm_spam[40:50, 1:10]), Show part of sparse matrix for spam

# Calculate the word frequency of each word of the dtm_ham sparse matrix
wordfreq_ham <- create_term_frequency_counts(dtm_ham)
head(wordfreq_ham)
# Calculate the word frequency of each word of the dtm_spam sparse matrix
wordfreq_spam <- create_term_frequency_counts(dtm_spam)
head(wordfreq_spam)

# Combine two columns of data
word_freq <- full_join(wordfreq_ham, wordfreq_spam, by = "word",
                       suffix =  c("_ham", "_spam"))
head(word_freq)

corpus <- create_corpus(sms$message)        # Text processing all text
dtm <- DocumentTermMatrix(corpus)           # Building a sparse matrix

# Convert the sparse matrix to a data frame, and select only the feature values in the word column in word_freq
dtm <- as.data.frame(as.matrix(dtm)) %>%
  select(word_freq$word)

wordfreq_dtm <- create_term_frequency_counts(dtm)

dtm2 <- suppressWarnings(cbind(dtm, sms[-2])) # Combine sparse matrices with previously reshaped data
dtm2$label <- as.factor(dtm2$label)           # Convert categories to factor types

table(dtm2$label)                             # View the number of categories, find that it is imbalanced
prop.table(table(dtm2$label))                 # View the percentage of category, find that it is imbalanced

# Since this is an imbalanced data problem, we use an algorithm called 'oversampling' to enhance the data set
# Oversampling also improves the generalization ability of the model

# data = dtm2 is the data source
# method = 'over', Oversampling
# NRepresents the final size of data
data_balanced_over <- ovun.sample(label ~ ., data = dtm2,           
                                  method = "over", N = 9650)$data   
table(data_balanced_over$label)                                     

## Data cutting: Establish training sets and test sets - stratified sampling
data_balanced_over <- select(data_balanced_over, label, everything())
set.seed(100)
index3 <- createDataPartition(data_balanced_over$label, p = 0.7, list = F)
traindata <- data_balanced_over[index3, ] # Establish training set
testdata <- data_balanced_over[-index3, ] # Establish testing set

# Cross Validation
# Cross-validation is used to evaluate the predictive performance of the model, 
# especially the performance of the trained model on the new data, which can reduce over-fitting to some extent.
# can also let us get as much valid information as possible from limited data.
# https://machinelearningmastery.com/k-fold-cross-validation/
train_control <- trainControl(method = 'cv', number = 5)  ### 5 -fold cross validation

## Now start modeling with classification methods
## We try popular classification methods such as: knn, Naïve Bayes, Random_Forest, SVM, C50, CART, Logistic Regression and ANN
## Use confusion matrix method to evaluate the model
## Timing with system time difference

# knn                 
time1 <- Sys.time()
set.seed(1111) 
model_knn <- kknn(label ~., traindata, testdata, k = 10, kernel ="rectangular")
fit <- fitted(model_knn)  
table(testdata$label,fit)
time2 <- Sys.time()
time = time2 - time1
time  

# Naïve Bayes
time1 <- Sys.time()
set.seed(2468) 
model_nb <- naiveBayes(traindata[-1], traindata[, 1], trControl = train_control)
pred_nb <- predict(model_nb, testdata[-1])                  
confusionMatrix(pred_nb, testdata$label, positive = 'spam')      
time2 <- Sys.time()
time = time2 - time1
time  

# Random Forest, the highest precision
time1 <- Sys.time()
set.seed(1357) 
model_rf <- randomForest(label~., data = traindata, trControl = train_control, importance = T, proximity = T, ntree = 100)
pred_rf <- predict(model_rf, testdata[-1])                  
confusionMatrix(pred_rf, testdata$label, positive = 'spam') 
time2 <- Sys.time()
time = time2 - time1
time  

# SVM
time1 <- Sys.time()
set.seed(1234) 
model_svm <- svm(label ~ ., data = traindata, trControl = train_control, kernel = "linear", cost = 0.1, gamma = 0.1)
pred_svm <- predict(model_svm, testdata[-1])
confusionMatrix(pred_svm, testdata$label, positive = 'spam') 
time2 <- Sys.time()
time = time2 - time1
time  

# C50                     
time1 <- Sys.time()     
set.seed(3333) 
model_c50 <- C5.0(label ~ ., data = traindata)
pred_c50 <- predict(model_c50, testdata, type = "class")
confusionMatrix(pred_c50, testdata$label, positive = 'spam')     
time2 <- Sys.time()
time = time2 - time1
time  

# CART, the least time consuming
time1 <- Sys.time()     
set.seed(4444) 
model_cart <- rpart(label ~., data = traindata)
rpart.plot(model_cart)
pred_cart <- predict(model_cart, testdata, type = "class")
confusionMatrix(pred_cart, testdata$label, positive = 'spam')    
time2 <- Sys.time()
time = time2 - time1
time  

# Logistic Regression
time1 <- Sys.time()    
set.seed(7777) 
model_log <- glm(label ~., data = traindata, family=binomial(link="logit"))
pred_log <- predict(model_cart, testdata, type = "class")
confusionMatrix(pred_log, testdata$label, positive = 'spam')    
time2 <- Sys.time()
time = time2 - time1
time  

# ANN, not suitable in this case
time1 <- Sys.time()
set.seed(5555) 
model_ann <- neuralnet(label ~., data = traindata, hidden = 3, act.fct = "logistic", linear.output = FALSE)
pred_ann <- predict(model_ann, testdata)               
plot(model_ann) # Too crowed, too many arguments, so we don't use ANN
time2 <- Sys.time()
time = time2 - time1
time 

## Use Random Forest as the main method, use SVM as the double check
