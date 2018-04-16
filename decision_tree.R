####데이터 출처####
###Titanic: Machine Learning from Disaster, https://www.kaggle.com/c/titanic/data
###http://www.learnbymarketing.com/tutorials/rpart-decision-trees-in-r/ #decision tree 설명 
require(rpart)
require(rpart.plot)
require(rattle)
require(dplyr)
require(readr)
require(doBy)
require(reshape2)
require(ggplot2)
require(caret)
require(e1071)


####데이터 불러오기####

titanic<-read_csv("D:/R_File/Decision-Tree/titanic_data.csv")

####Exploratory Data Analysis####

glimpse(titanic) # 891개 Obs, 12개 변수

###승선등급별 생존률
addmargins(prop.table(table(titanic$Pclass, titanic$Survived)))
ggplot(titanic,aes(x=Pclass,fill=factor(Survived)))+geom_bar(aes(y=(..count..)/sum(..count..)))+theme_bw()

###성별 생존율
addmargins(prop.table(table(titanic$Sex, titanic$Survived)))

###나이별 생존률
addmargins(prop.table(table(titanic$Age, titanic$Survived)))
ggplot(titanic,aes(x=Pclass,fill=factor(Survived)))+geom_bar(aes(y=(..count..)/sum(..count..)))+theme_bw()

is.na(titanic$Age)
###형제자매 동승자 생존율
addmargins(prop.table(table(titanic$SibSp, titanic$Survived)))
ggplot(titanic,aes(x=SibSp,fill=factor(Survived)))+geom_bar(aes(y=(..count..)/sum(..count..)))+theme_bw()

###부모자식 동승자 생존율
addmargins(prop.table(table(titanic$Parch, titanic$Survived)))
ggplot(titanic,aes(x=Parch,fill=factor(Survived)))+geom_bar(aes(y=(..count..)/sum(..count..)))+theme_bw()

###요금에 따른 생존율
addmargins(prop.table(table(titanic$Fare, titanic$Survived)))

###승선지역에 따른 생존율
addmargins(prop.table(table(titanic$Embarked, titanic$Survived)))
ggplot(titanic,aes(x=Embarked,fill=factor(Survived)))+geom_bar(aes(y=(..count..)/sum(..count..)))

###승선지역과 승선등급별 생존
ggplot(titanic,aes(x=Pclass,fill=factor(Survived)))+geom_bar(aes(y=(..count..)/sum(..count..)))+facet_wrap(~Embarked)+theme_bw()

###승무원 생존율
addmargins(prop.table(table(titanic$Cabin_Derived, titanic$Survived)))
ggplot(titanic,aes(x=Cabin_Derived,fill=factor(Survived)))+geom_bar(aes(y=(..count..)/sum(..count..)))+theme_bw()


####dcast function으로 살펴보기####
dcast(titanic, Sex~Age, value.var = "Survived", sum, margins = T)
dcast(titanic, Sex~Pclass, value.var = "Survived", sum, margins = T)
dcast(titanic, Sex~SibSp, value.var = "Survived", sum, margins = T)
dcast(titanic, Embarked~Pclass, value.var = "Survived", length, margins = T)
dcast(titanic, Embarked~., value.var = "Survived", length, margins =T)

table(titanic$Sex, titanic$Survived)
titanic$Cabin_Derived<-ifelse(!(is.na(titanic$Cabin)), "Cabin","Passenger")

####Train & Test Data Set 만들기####
set.seed(9999)
samp<-sample(nrow(titanic), 791, replace = F) 
train<-titanic[samp,]
test<-titanic[-samp,]

subset(train,!is.na(Age),select=c(Survived,Pclass,Sex,Age,Embarked,Cabin_Derived))

####Decision Tree####
fit <- rpart(Survived ~., data=subset(train,!is.na(Age),select=c(Survived,Pclass,Sex,Age,Embarked,Cabin_Derived)), minbucket =20)
summary(fit)
fancyRpartPlot(fit)
a<-predict(fit, newdata=test)
names(test)
printcp(fit)
plotcp(fit)
names(fit)

pfit<-prune(fit, cp= fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])
plot(pfit)
text(pfit)


####Model Accuracy 확인_Confusion Matrix ####
test$Survived_pred<-ifelse(predict(fit,newdata=test)>0.5,1,0)
test$Survived_pred_prob<-predict(fit,newdata=test)
test<-subset(test,!is.na(Age),select=c(Survived,Pclass,Sex,Age,Embarked,Cabin_Derived))
confusionMatrix(as.factor(test$Survived_pred),as.factor(test$Survived)) #실제 생존자와 예측 모델을 통한 생존자 비교
class(test$Survived)
test


gc()

glimpse(train)
glimpse(titanic)
