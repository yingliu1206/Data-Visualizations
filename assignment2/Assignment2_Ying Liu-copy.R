## Importing packages
library(tidyverse) 
library(caret)
library(rpart)
library(rpart.plot)
library(pROC)
library(ggplot2)

## read and simplify the data
telco = read.csv("/Users/didi/Desktop/gu/assignment2/WA_Fn-UseC_-Telco-Customer-Churn.csv")
telco = subset(telco, select = -c(customerID,MultipleLines,OnlineBackup,OnlineSecurity,TechSupport,StreamingTV, StreamingTV, StreamingMovies,Contract,PaperlessBilling,Dependents, SeniorCitizen, Dependents, DeviceProtection))
head(telco)
ncol(telco)
nrow(telco)
str(telco)

## data cleaning for all variables
options(repr.plot.width = 6, repr.plot.height = 4)
missing_data = telco %>% summarise_all(funs(sum(is.na(.))/n()))
missing_data = gather(missing_data, key = "variables", value = "percent_missing")
ggplot(missing_data, aes(x = percent_missing, y = variables)) +
  geom_bar(stat = "identity", fill = "red", aes(color = I('white')), size = 0.3)+
  labs(title="#1. Visualizing NAs in all the columns", x = 'percent_missing', y= 'variables')+
  theme(plot.title = element_text(size=12))

## data cleaning for each variable
  
##############################
## Column 1: gender
###################################
  
### before: data cleaning for gender
gender = ggplot(telco, aes(x = gender))+ 
  geom_bar(aes(fill = gender), size=0.6) + 
  labs(title = "#2. Gender", x = "gender", y = "count")+
  geom_text(stat='count', aes(label=..count..)) +
  theme_bw()
plot(gender)
(NumRows=nrow(telco))

(sum(is.na(telco$gender)))  ## This confirms that it is not NA
### This shows that we have no blank and no NA....

### after: data cleaning for gender
gender2 = ggplot(telco, aes(x = gender))+ 
  geom_bar(aes(fill = gender), size=0.6) + 
  labs(title = "#3. Gender", x = "gender", y = "count")+
  geom_text(stat='count', aes(label=..count..)) +
  theme_bw()
plot(gender2)

############################################
## Next variable is: Partner
#############################################

### before
(TempTable = table(telco$Partner))
(MyLabels = paste(names(TempTable), ":", 
                   round(TempTable/NumRows,2) ,sep=""))

#install.packages("plotrix")
library(plotrix)
pie3D(TempTable,labels=MyLabels,explode=0.3,
      main="#4. Pie Chart of Partner")

(sum(is.na(telco$Partner)))  ## This confirms that it is not NA

### after
(TempTable = table(telco$Partner))
(MyLabels = paste(names(TempTable), ":", 
                  round(TempTable/NumRows,2) ,sep=""))

pie3D(TempTable,labels=MyLabels,explode=0.3,
      main="#5. Pie Chart of Partner")

############################################
## Next variable is: tenure
#############################################
(meds <- ddply(telco, .(Churn), summarize, 
                   med = median(tenure)))

### before
tenure = ggplot(telco, aes(x = Churn, y = tenure))+
  geom_boxplot(aes(fill = gender))+ 
  labs(title="#6. Tenure",
       subtitle = "tenure vs Churn")+
  theme_classic()
plot(tenure)

# check outliers
count(telco[telco$tenure>=58 & telco$Churn == "Yes" & telco$gender == "Female",])

# remove outlier
telco = telco[-(which(telco$tenure>=58 & telco$Churn == "Yes" & telco$gender == "Female")),]

### after
tenure2 = ggplot(telco, aes(x = Churn, y = tenure))+
  geom_boxplot(aes(fill = gender))+ 
  labs(title="#7. Tenure",
       subtitle = "tenure vs Churn")+
  theme_classic()
plot(tenure2)

############################################
## Next variable is: PhoneService
#############################################

### before
PhoneService = ggplot(telco, aes(PhoneService)) + 
  scale_fill_brewer(palette = "Spectral")+ 
  geom_histogram(aes(fill=InternetService), 
                 stat = "count",
                 col="black", 
                 size=.1) +  
  labs(title="#8. Histogram for PhoneService") +
  geom_text(stat='count', aes(label=..count..))+
  theme_light()

plot(PhoneService)

### after
PhoneService2 = ggplot(telco, aes(PhoneService)) + 
  scale_fill_brewer(palette = "Spectral")+ 
  geom_histogram(aes(fill=InternetService), 
                 stat = "count",
                 col="black", 
                 size=.1) +  
  labs(title="#9. Histogram for PhoneService") +
  geom_text(stat='count', aes(label=..count..))+
  theme_light()

plot(PhoneService2)

############################################
## Next variable is: InternetService
#############################################

### before
InternetService = ggplot(telco, aes(MonthlyCharges))+ 
  geom_density(aes(fill=InternetService), alpha=0.8) + 
  labs(title="#10. Internet Service",
       subtitle = "Internet Service vs Monthly Charges",
       fill="InternetService")+
  theme_minimal_hgrid()

plot(InternetService)

### after
InternetService2 = ggplot(telco, aes(MonthlyCharges))+ 
  geom_density(aes(fill=InternetService), alpha=0.8) + 
  labs(title="#11. Internet Service",
       subtitle = "Internet Service vs Monthly Charges",
       fill="InternetService")+
  theme_minimal_hgrid()

plot(InternetService2)

############################################
## Next variable is: PaymentMethod
#############################################

### before
PaymentMethod = ggplot(telco, aes(x = PaymentMethod, y =TotalCharges, color = Churn)) + 
  geom_point() + 
  scale_color_manual(values = c('blue',"red"))+
  labs(subtitle="Payment Method vs Total Charges",
       title="#12. PaymentMethod")+
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) 

plot(PaymentMethod)

### after
PaymentMethod2 = ggplot(telco, aes(x = PaymentMethod, y =TotalCharges, color = Churn)) + 
  geom_point() + 
  scale_color_manual(values = c('blue',"red"))+
  labs(subtitle="Payment Method vs Total Charges",
       title="#13. PaymentMethod")+
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) 

plot(PaymentMethod2)

############################################
## Next variable is: MonthlyCharges
#############################################

### before
MonthlyCharges = ggplot(telco, aes(PaymentMethod, MonthlyCharges))+ 
  geom_violin(aes(fill = PaymentMethod)) + 
  labs(title="#14. Monthly Charges", 
       subtitle="Monthly Charges vs Class of Payment Method",
       x="Class of Payment Method",
       y="Monthly Charges")+
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) 
plot(MonthlyCharges)

### after
MonthlyCharges2 = ggplot(telco, aes(PaymentMethod, MonthlyCharges))+ 
  geom_violin(aes(fill = PaymentMethod)) + 
  labs(title="#15. Monthly Charges", 
       subtitle="Monthly Charges vs Class of Payment Method",
       x="Class of Payment Method",
       y="Monthly Charges")+
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) 
plot(MonthlyCharges2)

############################################
## Next variable is: TotalCharges
#############################################

## Are there NAs?
(sum(is.na(telco$TotalCharges)))

## Fix the missing TotalCharges first
## Find it
(MissingTotalCharges = telco[is.na(telco$TotalCharges),])
## We can replace the missing value with the median.
(MyMed = median(telco$TotalCharges , na.rm=TRUE))
## NOW - replace the missing GPA with this Median
telco$TotalCharges[is.na(telco$TotalCharges)] <- MyMed
## Check to assure the missing value  was updated...
(sum(is.na(telco$TotalCharges)))

### before
TotalCharges = ggplot(telco, aes(x=tenure, y=TotalCharges)) + 
  geom_point(aes(col=Churn), size=0.1) +  # Set color to vary based on state categories.
  geom_smooth(method="lm", col="firebrick", size=1) + 
  scale_x_continuous(breaks=seq(0, 80,10))+
  scale_y_continuous(breaks=seq(0, 9000, 1000))+
  labs(title="#16. TotalCharges")+
  theme_classic()

plot(TotalCharges)

## find incorrect values
count(telco[telco$tenure==0 & telco$TotalCharges !=0,])

## remove incorrect values
telco = telco[-(which(telco$tenure==0 & telco$TotalCharges !=0)),]

### after
TotalCharges2 = ggplot(telco, aes(x=tenure, y=TotalCharges)) + 
  geom_point(aes(col=Churn), size=0.1) +  # Set color to vary based on state categories.
  geom_smooth(method="lm", col="firebrick", size=1) + 
  scale_x_continuous(breaks=seq(0, 80,10))+
  scale_y_continuous(breaks=seq(0, 9000, 1000))+
  labs(title="#17. TotalCharges")+
  theme_classic()

plot(TotalCharges2)

############################################
## Next variable is: Churn
#############################################

### before
Churn = ggplot(telco, aes(x = Churn, y = MonthlyCharges)) + 
   geom_count(col="tomato3", show.legend=F)+
  scale_y_continuous(breaks=seq(0, 120,20))+
  labs(title="#18. Churn", 
       subtitle = "Monthly Charges vs Churn")+
   theme_cowplot()

plot(Churn)

### after
Churn2 = ggplot(telco, aes(x = Churn, y = MonthlyCharges)) + 
  geom_count(col="tomato3", show.legend=F)+
  scale_y_continuous(breaks=seq(0, 120,20))+
  labs(title="#19. Churn", 
       subtitle = "Monthly Charges vs Churn")+
  theme_cowplot()

plot(Churn2)


############################# EDA part #############################
library(ggcorrplot)
## correlation
options(repr.plot.width =6, repr.plot.height = 4)
telco_cor = round(cor(telco[,c("tenure", "MonthlyCharges", "TotalCharges")]), 1)
ggcorrplot(telco_cor,  title = "#20. Correlation")+
  theme(plot.title = element_text(hjust = 0.5))


#Strong positive correlation between tenure and total charges.
#weak positive correlation between tenure and monthly charges.
#medium to strong positive correlation between monthly charges and total charges.

library(dplyr)
## churn
options(warn=-1)
sam = theme(plot.background = element_rect(fill="#F5FFFA",color = "darkblue"),
             plot.title = element_text(size=15, hjust=.5),
             axis.title.x = element_text(size=12, color = "black"),
             axis.title.y = element_text(size=12, color = "black"),
             axis.text.x = element_text(size=10),
             axis.text.y = element_text(size=10),
             legend.position = "top",
             legend.text = element_text(size=10),
             legend.title = element_text(size=10))

(churn = telco %>%
  group_by(Churn) %>%
  dplyr::summarise(n = n())%>%
  mutate(prop = n / sum(n)) %>%
  ungroup()%>%
  mutate(label_text = str_glue("n:{n} \n prop:{scales::percent(prop)}")))

options(repr.plot.width=15, repr.plot.height=10)
churn1 = churn %>% ggplot(aes(x = Churn,y = prop,fill = Churn)) + 
  geom_col(alpha=0.7,color="black") +
  geom_label(aes(label=label_text),fill="white",size =5,position=position_fill(vjust=0.3),color = "black",size = 1)+
  xlab("Churn(Yes,NO)") +
  ylab("Prop") +
  ggtitle("#21. Churn Bar Graph Distribution")+
  scale_y_continuous(labels = scales::percent_format())+
  theme_minimal()+
  sam
plot(churn1)
# Based on the data,74% of the customers stopped using our services/products! Only 26% are still active. 

## Tenure VS Churn
options(repr.plot.width=20, repr.plot.height=15)
t = telco %>% ggplot(mapping = aes(x = tenure)) +
  geom_bar(aes(fill = Churn),color="black",alpha=0.7) +
  theme_minimal()+
  xlab("Tenure")+
  ylab("Count") +
  ggtitle("#22. Tenure Bar Graph with Churn Overlay\n(Not Normalized)")+
  theme(legend.position = "none") +
  sam

t1 = telco %>% ggplot(mapping = aes(x = tenure)) +
  geom_bar(aes(fill = Churn),position = 'fill',color="black",alpha=0.7) +
  scale_y_continuous(labels = scales::percent_format())+
  theme_minimal()+
  xlab("Tenure")+
  ylab("Prop") +
  ggtitle("#23. Tenure Bar Graph with Churn Overlay \n(Normalized)") +
  sam

t2 = telco %>% ggplot(aes(x = tenure,fill = Churn)) + 
  geom_density(alpha=0.7,color="black") +
  xlab("Tenure") +
  ylab("Prop") +
  ggtitle("#24. Tenure Density Graph with Churn") +
  theme_minimal()+
  sam

plot_grid(t,t1,t2,nrow=3,ncol=1)

## Monthly Charges VS Churn
options(repr.plot.width=20, repr.plot.height=15)

m = telco %>% ggplot(mapping = aes(x = MonthlyCharges)) +
  geom_bar(aes(fill = Churn),alpha=0.7) +
  theme_minimal()+
  xlab("Monthly Charges ($)")+
  ylab("Count") +
  ggtitle("#25. Monthly Charges Bar Graph with Churn Overlay\n(Not Normalized)")+
  theme(legend.position = "none") +
  sam

m1 = telco %>% ggplot(mapping = aes(x = MonthlyCharges)) +
  geom_bar(aes(fill = Churn),position = 'fill',alpha=0.7) +
  scale_y_continuous(labels = scales::percent_format())+
  theme_minimal()+
  xlab("Monthly Charges ($)")+
  ylab("Prop") +
  ggtitle("#26. Monthly Charges Bar Graph with Churn Overlay \n(Normalized)") +
  sam

m2 = telco %>% ggplot(aes(x = MonthlyCharges,fill = Churn)) + 
  geom_density(alpha=0.7,color="black") +
  xlab("Monthly Charges") +
  ylab("Prop") +
  ggtitle("#27. Monthly Charges Density Graph with Churn") +
  theme_minimal()+
  sam

plot_grid(m,m1,m2,nrow=3,ncol=1)
#The majority of our customers has low monthly charges. Those with the highest proportion of positive churn(left our platform) are the ones with high monthly charges(between '70' to '112'$ / month)

## Total Charges VS Churn
options(repr.plot.width=20, repr.plot.height=15)

options(warn=-1)
tt = telco %>% ggplot(mapping = aes(x = TotalCharges)) +
  geom_bar(aes(fill = Churn),alpha = 0.7) +
  theme_minimal()+
  xlab("Total Charges ($)")+
  ylab("Count") +
  ggtitle("#28. Total Charges Bar Graph with Churn Overlay\n(Not Normalized)")+
  theme(legend.position = "none") +
  sam

tt1 = telco %>% ggplot(mapping = aes(x = TotalCharges)) +
  geom_bar(aes(fill = Churn),position = 'fill',alpha=0.7) +
  scale_y_continuous(labels = scales::percent_format())+
  theme_minimal()+
  xlab("Total Charges ($)")+
  ylab("Prop") +
  ggtitle("#29. Total Charges Bar Graph with Churn Overlay \n(Normalized)") +
  sam

tt2 = telco %>%  ggplot(aes(x =TotalCharges,fill = Churn)) + 
  geom_density(alpha=0.7,color="black") +
  xlab("Total Charges") +
  ylab("Prop") +
  ggtitle("#30. Total Charges Density Graph with Churn") +
  theme_minimal()+
  sam

plot_grid(tt,tt1,tt2,nrow=3,ncol=1)
#Clearly shown that the highest proportion of positive churn(left our platform) are customers who has low 'total charge'(from '0' to '2000'$).

# categorical variables and churn
plot_categorical_vs_target = function(data, target, list_of_variables){
  target <- sym(target) #Converting the string to a column reference
  i <-1 
  plt_matrix <- list()
  for(column in list_of_variables){
    col <- sym(column) 
    temp <- data %>% group_by(!!sym(col),!!sym(target)) %>% 
      dplyr::summarize(count = n()) %>% 
      mutate(prop = round(count/sum(count),2)) %>%
      ungroup()%>%
      mutate(label_text = str_glue("n : {count}\nprop:{scales::percent(prop)}"))
    
    
    options(repr.plot.width=20, repr.plot.height=15) 
    
    plt_matrix[[i]]<-ggplot(data= temp, aes(x=!!sym(col), y=prop,fill =!!sym(target))) + 
      geom_bar(stat="identity",alpha=0.7,color = "black") +
      geom_label(aes(label=label_text),size = 3, hjust = 0.5, fill = "white",color="black") +
      scale_y_continuous(labels=scales::percent_format()) +
      xlab(column) +
      ylab("Prop") +
      ggtitle(paste("#",30+i,"Distribution of 'churn'\nfrequency across",column)) +
      theme_minimal()+
      theme(axis.text.x = element_text(angle=0))+
      sam
    i<-i+1
  }
  
  plot_grid(plotlist = plt_matrix,ncol=2)
}

plot_categorical_vs_target(telco,'Churn',c('gender','Partner','PhoneService',"InternetService"))
# There is no insight we could get from gender as both female and male have almost the same proportion with regard of the churn variable.
# 33% of the customers with no partner left the platform. 81% of customers with partner didn’t leave.
# both customers with phone service and without share the same proportion. Therefore, there is nothing to help us with buidling the model.
# 41% Customers of fiber optic services left our platform the last month.93% of customers with no internet services did not leave. 81% of our customers with DSL services didnt leave.

plot_categorical_vs_target2 = function(data, target, list_of_variables){
  target <- sym(target) #Converting the string to a column reference
  i <-1 
  plt_matrix <- list()
  for(column in list_of_variables){
    col <- sym(column) 
    temp <- data %>% group_by(!!sym(col),!!sym(target)) %>% 
      dplyr::summarize(count = n()) %>% 
      mutate(prop = round(count/sum(count),2)) %>%
      ungroup()%>%
      mutate(label_text = str_glue("n : {count}\nprop:{scales::percent(prop)}"))
    
    
    options(repr.plot.width=20, repr.plot.height=10) 
    
    plt_matrix[[i]]<-ggplot(data= temp, aes(x=!!sym(col), y=prop,fill =!!sym(target))) + 
      geom_bar(stat="identity",alpha=0.7,color = "black") +
      geom_label(aes(label=label_text),size = 3, hjust = 0.5, fill = "white",color="black") +
      scale_y_continuous(labels=scales::percent_format()) +
      xlab(column) +
      ylab("Prop") +
      ggtitle(paste("#35. Distribution of 'churn'\nfrequency across",column)) +
      theme_minimal()+
      theme(axis.text.x = element_text(angle=0.7))+
      sam
    i<-i+1
  }
  plot_grid(plotlist = plt_matrix)
}
plot_categorical_vs_target2(telco,'Churn',c('PaymentMethod'))
# 45% of our customers with electronic check as payment method left our platform. 

############################ build the model #############################
# Standardising Continuous features
num_columns = c("tenure", "MonthlyCharges", "TotalCharges")
telco[num_columns] = sapply(telco[num_columns], as.numeric)

telco_int = telco[,c("tenure", "MonthlyCharges", "TotalCharges")]
telco_int = data.frame(scale(telco_int))

# Creating Dummy Variables
telco_cat = telco[,c('gender',"Partner", "PhoneService", "InternetService", "PaymentMethod", "Churn")]
dummy =  data.frame(sapply(telco_cat,function(x) data.frame(model.matrix(~x-1,data =telco_cat))[,-1]))
head(dummy)

# Combining the data
telco_final = cbind(telco_int,dummy)
head(telco_final)

#Splitting the data
set.seed(123)
indices = sample.split(telco_final$Churn, SplitRatio = 0.7)
train = telco_final[indices,]
test = telco_final[!(indices),]

############################## logistic regression ###############################
#Build the first model using all variables
model_1 = glm(Churn ~ ., data = train, family = "binomial")
summary(model_1)

# Using stepAIC for variable selection, which is a iterative process of adding or removing variables, in order to get a subset of variables that gives the best performing model.
final_model = stepAIC(model_1, direction="both")
summary(final_model)

# Model Evaluation using the test data:
pred = predict(final_model, type = "response", newdata =test[,-12])
summary(pred)
test$prob <- pred

# Using probability cutoff of 50%.
pred_churn = factor(ifelse(pred >= 0.50, "Yes", "No"))
actual_churn = factor(ifelse(test$Churn==1,"Yes","No"))

# confusion matrix
table(actual_churn,pred_churn)
conf_final = confusionMatrix(pred_churn, actual_churn, positive = "Yes")
fourfoldplot(conf_final$table,main = "#36. Confusion Matrix Logistic Regression")

# accuracy
(accuracy = conf_final$overall[1])
(sensitivity = conf_final$byClass[1])
(specificity = conf_final$byClass[2])
# As we can see above, when we are using a cutoff of 0.50, we are getting a good accuracy and specificity, but the sensitivity is very less. Hence, we need to find the optimal probalility cutoff which will give maximum accuracy, sensitivity and specificity

# Find optimal cutoff
perform_fn = function(cutoff) 
{
  predicted_churn = factor(ifelse(pred >= cutoff, "Yes", "No"))
  conf = confusionMatrix(predicted_churn, actual_churn, positive = "Yes")
  accuray = conf$overall[1]
  sensitivity = conf$byClass[1]
  specificity = conf$byClass[2]
  out = t(as.matrix(c(sensitivity, specificity, accuray))) 
  colnames(out) = c("sensitivity", "specificity", "accuracy")
  return(out)
}

options(repr.plot.width =8, repr.plot.height =6)
summary(pred)
s = seq(0.01,0.80,length=100)
OUT = matrix(0,100,3)

for(i in 1:100)
{
  OUT[i,] = perform_fn(s[i])
} 

plot(s, OUT[,1],main = "#37. Optimal Cutoff",xlab="Cutoff",ylab="Value",cex.lab=1.5,cex.axis=1.5,ylim=c(0,1),
     type="l",lwd=2,axes=FALSE,col=2)
axis(1,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
axis(2,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
lines(s,OUT[,2],col="darkgreen",lwd=2)
lines(s,OUT[,3],col=4,lwd=2)
box()
legend("bottomright",col=c(2,"darkgreen",4,"darkred"),text.font =3,inset = 0.02,
       box.lty=0,cex = 0.8, 
       lwd=c(2,2,2,2),c("Sensitivity","Specificity","Accuracy"))
abline(v = 0.3, col="red", lwd=1, lty=2)
axis(1, at = seq(0.1, 1, by = 0.1))
# Let's choose a cutoff value of 0.3 for final model, where the three curves for accuracy, specificty and sensitivity meet

cutoff_churn = factor(ifelse(pred >=0.3, "Yes", "No"))
conf_final = confusionMatrix(cutoff_churn, actual_churn, positive = "Yes")
accuracy = conf_final$overall[1]
sensitivity = conf_final$byClass[1]
specificity = conf_final$byClass[2]
accuracy
sensitivity
specificity

############################## decision tree ###############################
set.seed(123)
telco_final$Churn = as.factor(telco_final$Churn)
indices = sample.split(telco_final$Churn, SplitRatio = 0.7)
train = telco_final[indices,]
validation = telco_final[!(indices),]

# Training the Decision Tree model using all variables & Predicting in the validation data
options(repr.plot.width = 10, repr.plot.height = 8)
decision_model = rpart(Churn ~ ., data=train,method="class")
rpart.plot(decision_model,type = 4,extra=101, main="#38. Decision Tree")

# Predicting 
DTPred = predict(decision_model,type = "class", newdata = validation[,-12])

# confusion matrix
(cm_dt = confusionMatrix(validation$Churn, DTPred))
fourfoldplot(cm_dt$table,main = "#39. Confusion Matrix Decision Tree")
accuracy = cm_dt$overall[1]
sensitivity = cm_dt$byClass[1]
specificity = cm_dt$byClass[2]
accuracy
sensitivity
specificity

# The decision tree model (accuracy - 78.62%) gives slightly better accuracy with respect to the logistic regression model (accuracy 75%). The sensitivity is also better in case of Decision tree which is 83.71%. However, the specificity has decreased to 60.69% in case of Decision Tree as compared to logistic regression model.

############################## random forest ###############################
library(randomForest)
model.rf = randomForest(Churn ~ ., data=train, proximity=FALSE,importance = FALSE, ntree=500,mtry=4, do.trace=FALSE)
model.rf
#The basic RandomForest model gives an accuracy of 78.86%( almost close enough to the OOB estimate), Sensitivity 82.46% and Specificity 63.99%.,

# Predicting on the validation set and checking the Confusion Matrix.
testPred = predict(model.rf, newdata=validation[,-12])
table(testPred, validation$Churn)
(cm_rf = confusionMatrix(validation$Churn, testPred))
fourfoldplot(cm_rf$table,main = "#40. Confusion Matrix Random Forest")
accuracy = cm_rf$overall[1]
sensitivity = cm_rf$byClass[1]
specificity = cm_rf$byClass[2]
accuracy
sensitivity
specificity

#Checking the variable Importance Plot
varImpPlot(model.rf, main = "#41. The variable Importance Plot",color = "brown",cex=1)

############################## Comparing 3 models ##############################
# Checking the AUC for all three models:
options(repr.plot.width =10, repr.plot.height = 8)

glm.roc = roc(response = validation$Churn, predictor = as.numeric(pred))
DT.roc = roc(response = validation$Churn, predictor = as.numeric(DTPred))
rf.roc = roc(response = validation$Churn, predictor = as.numeric(testPred))

plot(glm.roc, legacy.axes = TRUE, print.auc.y = 1.0, print.auc = TRUE, main='#42. AUC value of three models')
plot(DT.roc, col = "blue", add = TRUE, print.auc.y = 0.65, print.auc = TRUE)
plot(rf.roc, col = "red" , add = TRUE, print.auc.y = 0.85, print.auc = TRUE)
legend("bottomright", c("Random Forest", "Decision Tree", "Logistic Regression"),
       lty = c(1,1), lwd = c(2, 2), col = c("red", "blue", "black"), cex = 0.75)

  # create a table to compare
(accuracy_lr = conf_final$overall[1])
(sensitivity_lr = conf_final$byClass[1])
(specificity_lr = conf_final$byClass[2])
(accuracy_dt = cm_dt$overall[1])
(sensitivity_dt = cm_dt$byClass[1])
(specificity_dt = cm_dt$byClass[2])
(accuracy_rf = cm_rf$overall[1])
(sensitivity_rf = cm_rf$byClass[1])
(specificity_rf = cm_rf$byClass[2])

value = c(accuracy_lr, sensitivity_lr, specificity_lr, 
          accuracy_dt, sensitivity_dt, specificity_dt, 
          accuracy_rf, sensitivity_rf, specificity_rf)

# create a matrix with 3 columns
mtr = matrix(value, ncol=3, byrow=TRUE)

#define column names and row names of matrix
rownames(mtr) = c("Logistic Regression", "Decision Tree", "Random Forest")
colnames(mtr) = c('Accuracy', 'Sensitivity', "Specificity")
    
# convert matrix to table 
(tab = as.table(mtr))
