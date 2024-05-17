An Analysis of Football Players
MGSC 310 Final Project
Yoni Kazovsky, Levi Davis, Art Song, Adithya Mahesh

# load all your libraries here
library(tidyverse)
library(caret)
library(rsample) 
library(yardstick)
library(ggplot2)
library(randomForest)
library(randomForestExplainer)
library(ggplot2)
library(tidyverse)
library(cluster)
library(factoextra)
library(GGally)
# note, do not run install.packages() inside a code chunk. install them in the console outside of a code chunk. 
Data Exploration & Cleaning
First we took a look at the data set then created a subset called football_selected that only has the varibales we want in it. Finally, we changed the name of the shots on target variable because it was not working in the random forest model due to its strange naming convention usin the apostraphes.


football <- read_csv("datasets/soccer.csv")

#football %>% glimpse()

football <- football %>%
  mutate(Goals_per_Appearance = Goals / Appearances,
         Conversion_rate = Goals / Shots)
#football %>% glimpse()

football_selected <- football %>%
  select(Conversion_rate, Goals_per_Appearance, Goals, `Shots on target`, Appearances, Shots, Age, Position, Offsides) %>%
  drop_na()
football_selected = football_selected %>%
  rename(Shots_on_target = `Shots on target`)
  
football_selected %>% glimpse()
Rows: 264
Columns: 9
$ Conversion_rate      <dbl> 0.16097561, 0.06521739, 0.07142857, 0.00000000, 0.05555556, 0.00000000, 0…
$ Goals_per_Appearance <dbl> 0.17934783, 0.04761905, 0.01851852, 0.00000000, 0.03030303, 0.00000000, 0…
$ Goals                <dbl> 33, 3, 1, 0, 1, 0, 8, 1, 0, 39, 55, 5, 1, 4, 3, 37, 1, 3, 3, 0, 9, 0, 3, …
$ Shots_on_target      <dbl> 92, 12, 4, 5, 5, 9, 41, 4, 3, 93, 105, 18, 4, 8, 5, 145, 3, 13, 20, 0, 32…
$ Appearances          <dbl> 184, 63, 54, 47, 33, 57, 132, 28, 26, 99, 87, 33, 20, 23, 14, 236, 18, 37…
$ Shots                <dbl> 205, 46, 14, 44, 18, 27, 144, 15, 14, 204, 222, 51, 13, 20, 12, 393, 20, …
$ Age                  <dbl> 31, 24, 23, 28, 21, 21, 27, 19, 24, 29, 31, 25, 20, 21, 19, 32, 25, 22, 2…
$ Position             <chr> "Midfielder", "Midfielder", "Midfielder", "Midfielder", "Midfielder", "Mi…
$ Offsides             <dbl> 83, 0, 1, 2, 0, 0, 2, 13, 0, 62, 55, 4, 4, 5, 2, 33, 1, 0, 0, 1, 6, 0, 4,…
Data Spliting for Regression
We used a 75 to 25 split which resulted in a training set of about 200 observations


set.seed(310)

football_split <- initial_split(football_selected, prop = 0.75)
football_train <- training(football_split)
football_test <- testing(football_split)

football_train %>% glimpse()
Rows: 198
Columns: 9
$ Conversion_rate      <dbl> 0.00000000, 0.17371938, 0.00000000, 0.13333333, 0.03921569, 0.13669065, 0…
$ Goals_per_Appearance <dbl> 0.00000000, 0.63414634, 0.00000000, 0.18181818, 0.05000000, 0.18095238, 0…
$ Goals                <dbl> 0, 78, 0, 6, 2, 19, 7, 12, 20, 5, 0, 25, 2, 3, 75, 86, 0, 37, 1, 42, 2, 1…
$ Shots_on_target      <dbl> 3, 201, 6, 14, 15, 51, 17, 38, 61, 11, 2, 86, 5, 4, 267, 190, 0, 117, 2, …
$ Appearances          <dbl> 26, 123, 35, 33, 40, 105, 34, 93, 122, 36, 15, 181, 29, 25, 346, 196, 12,…
$ Shots                <dbl> 14, 449, 28, 45, 51, 139, 38, 97, 195, 44, 6, 224, 21, 13, 616, 445, 1, 3…
$ Age                  <dbl> 24, 28, 28, 25, 24, 26, 30, 29, 27, 26, 24, 29, 26, 26, 31, 28, 29, 30, 1…
$ Position             <chr> "Midfielder", "Forward", "Midfielder", "Forward", "Forward", "Midfielder"…
$ Offsides             <dbl> 0, 65, 1, 4, 22, 4, 9, 22, 20, 7, 2, 15, 1, 6, 163, 117, 0, 60, 0, 59, 5,…
dim(football_train)
[1] 198   9
dim(football_test)
[1] 66  9
Model & Model Performance
Here are the results of our model and its performance


model1 <- lm(Goals ~ Goals_per_Appearance + Conversion_rate + Shots_on_target + Appearances + Shots + Age + Position + Offsides, football_train)

# In-sample prediction (training)
y_hat_train <- predict(model1, football_train)

# Out-of-sample prediction (test)
y_hat_test <- predict(model1, newdata = football_test)

rmse_train <- RMSE(y_hat_train, football_train$Goals)
rmse_test <- RMSE(y_hat_test, football_test$Goals)

print(paste("In-sample RMSE:", rmse_train))
[1] "In-sample RMSE: 3.75657362034032"
print(paste("Out-of-sample RMSE:", rmse_test))
[1] "Out-of-sample RMSE: 7.30030273761757"
summary(model1)

Call:
lm(formula = Goals ~ Goals_per_Appearance + Conversion_rate + 
    Shots_on_target + Appearances + Shots + Age + Position + 
    Offsides, data = football_train)

Residuals:
     Min       1Q   Median       3Q      Max 
-16.9431  -1.1588   0.4106   1.3316  17.4625 

Coefficients:
                      Estimate Std. Error t value             Pr(>|t|)    
(Intercept)          -1.427328   3.307413  -0.432             0.666560    
Goals_per_Appearance 18.119887   3.740498   4.844           0.00000265 ***
Conversion_rate      -7.307742   3.530588  -2.070             0.039835 *  
Shots_on_target       0.399460   0.037767  10.577 < 0.0000000000000002 ***
Appearances           0.012607   0.008975   1.405             0.161767    
Shots                -0.050866   0.015774  -3.225             0.001487 ** 
Age                   0.025943   0.089519   0.290             0.772289    
PositionForward      -0.349670   2.806679  -0.125             0.900985    
PositionMidfielder   -0.698716   2.779626  -0.251             0.801802    
Offsides              0.062536   0.016982   3.682             0.000302 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 3.855 on 188 degrees of freedom
Multiple R-squared:  0.9566,    Adjusted R-squared:  0.9545 
F-statistic:   460 on 9 and 188 DF,  p-value: < 0.00000000000000022
Data Visualization
As we can see the relationship between conversion rate and goals shows a lot of heteroscedasticity. In general a higher conversion rate results in more goals, however, as we increase in conversion ratre that relationship weakens hence the cone shape (heteroscedasticity).


ggplot(football_train, aes(x = Conversion_rate, y = Goals_per_Appearance, color = Position)) +
  geom_point() +
  labs(x = "Conversion Rate", y = "Goals") +
  ggtitle("Relationship between Conv Rate and Goals")


