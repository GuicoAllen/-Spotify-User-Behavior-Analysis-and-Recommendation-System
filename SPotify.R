# Load required libraries
library(readr)  # For reading CSV files
library(dplyr)  # For data manipulation
library(forecast)  # For time series and forecasting functions

# Read data from CSV file into a dataframe
spot <- read_csv("C:/Users/Admin/Downloads/tracks.csv")

# Add a new binary column based on popularity
spot <- spot %>%
  mutate(popularity_label = ifelse(popularity > 50, 1, 0))  # Binarize popularity

# Select numeric columns from the dataset
data_numeric <- spot %>%
  select_if(is.numeric)

# Function to normalize data using min-max scaling
normalize  <- function(x){
  min_max <- (x-min(x))/(max(x)-min(x))
  return(min_max)
}

# Apply normalization to all numeric columns
data_numeric$popularity_norm <- normalize(x=data_numeric$popularity)
data_numeric$duration_norm <- normalize(x=data_numeric$duration_ms)
data_numeric$explicit_norm <- normalize(x=data_numeric$explicit)
data_numeric$danceability_norm <- normalize(x=data_numeric$danceability)
data_numeric$key_norm <- normalize(x=data_numeric$key)
data_numeric$loudnesss_norm <- normalize(x=data_numeric$loudness)
data_numeric$mode_norm <- normalize(x=data_numeric$mode)
data_numeric$speechiness_norm <- normalize(x=data_numeric$speechiness)
data_numeric$acousticness_norm <- normalize(x=data_numeric$acousticness)
data_numeric$instrumentalness_norm <- normalize(x=data_numeric$instrumentalness)
data_numeric$liveness_norm <- normalize(x=data_numeric$liveness)
data_numeric$valence_norm <- normalize(x=data_numeric$valence)
data_numeric$tempo_norm <- normalize(x=data_numeric$tempo)
data_numeric$time_signature_norm <- normalize(x=data_numeric$time_signature)

# Compute the correlation matrix for the selected columns
cor(data_numeric[,16:30])
# In general, the correlation matrix for the normalized features of songs 
# shows that there are no strong correlations present.

# Split data into training and test sets
training_idx <- sample(1:nrow(data_numeric), size=0.8*nrow(data_numeric))
test_idx <- sample(1:nrow(data_numeric), size=0.2*nrow(data_numeric))
my_df_train <- data_numeric[training_idx, ]
my_df_test <- data_numeric[-test_idx, ]

# Fit a Poisson regression model using duration_ms to predict popularity
my_poisson_model <- glm(popularity ~ duration_ms, data=my_df_train, family="poisson")
summary(my_poisson_model)
exp(8.689e-03)-1  # Calculate the effect size from the coefficient

# Fit a larger Poisson model with multiple predictors
my_logit_big <- glm(popularity~popularity_label+duration_ms+explicit+
                      danceability+ key+loudness+mode+speechiness+acousticness
                    +instrumentalness+liveness+valence+tempo+time_signature, 
                    data=my_df_train, family="poisson")
summary(my_logit_big)

# Fit the same large model but with normalized predictors
my_logit_big <- glm(popularity_norm~popularity_label+duration_norm+explicit_norm+
                      danceability_norm+ key_norm+loudnesss_norm+mode_norm+speechiness_norm+acousticness_norm
                    +instrumentalness_norm+liveness_norm+valence_norm+tempo_norm+time_signature_norm, 
                    data=my_df_train, family="poisson")
summary(my_logit_big)


# duration_norm (0.655561): Indicates that longer duration songs 
# (in normalized terms) are associated with higher popularity. 
# The positive coefficient reflects an increase in the log count of popularity as song duration increases.

#loudnesss_norm (0.718139): Loudness has a substantial positive impact on
# popularity, suggesting that louder songs are more popular.


# Load the caret package for model training and evaluation
library(caret)

# Predict popularity using the fitted logistic regression model
my_prediction <- predict(my_logit_big, my_df_test, type="response")

# Create a confusion matrix to evaluate the logistic model
confusionMatrix(data= as.factor(as.numeric(my_prediction > 0.5)),
                reference= as.factor(as.numeric(my_df_test$popularity_norm)))

# Fit a linear regression model using duration and loudness to predict popularity
my_linear_model <- lm(popularity ~ duration_ms + loudness, data = my_df_train)

# Predict popularity using the fitted linear regression model
my_prediction <- predict(my_linear_model, my_df_test)

# Extract actual popularity values from the test dataset
actual <- my_df_test$popularity
predicted <- my_prediction

# Calculate the root mean squared error (RMSE) to evaluate the linear model
rmse <- sqrt(mean((predicted - actual)^2))

# Calculate the coefficient of determination (R-squared) to evaluate the linear model
r_squared <- cor(actual, predicted)^2

# Return RMSE and R-squared as a list for easy reference
list(RMSE = rmse, R_Squared = r_squared)

library(rpart)
library(rpart.plot)

# Fit a classification tree
my_tree <- rpart(popularity~duration_ms+loudness, 
                 data=my_df_train, method="class", cp=0.005)

# Plot the decision tree
rpart.plot(my_tree, type=1, extra=1, box.palette="Greens")

tree_predict <- predict(my_tree, my_df_test, type="prob")
# performance testing on the tree
predicted_values <- tree_predict

# Actual popularity values from the test set
actual_values <- my_df_test$popularity

# Calculate RMSE
rmse <- sqrt(mean((predicted_values - actual_values)^2))

# Print RMSE and R-squared
print(paste("Root Mean Squared Error:", rmse))

#The linear model outperformed the decision tree model in predicting popularity, 
# demonstrating a significantly lower Root Mean Squared Error (RMSE) of 17 compared 
# to the decision tree model's RMSE of 33. This indicates that the linear model has 
# a better fit and greater accuracy in estimating popularity based on the features duration_ms and loudness.

# Count the number of occurrences for each singer
singer_count <- spot %>%
  count(artists) %>%
  arrange(desc(n))

# Display the next most popular singer (second highest count)
next_most_popular_singer <- singer_count %>% slice(2)
print(next_most_popular_singer)

#Add a release year column extracted from the release_date
spot <- spot %>%
  mutate(release_year = as.numeric(substr(release_date, 1, 4)))

# Identify the top 10 artists based on their overall popularity
top_artists <- spot %>%
  group_by(artists) %>%
  summarize(total_popularity = sum(popularity, na.rm = TRUE)) %>%
  arrange(desc(total_popularity)) %>%
  slice(1:10)

# Forecasting future popularity for top 10 artists
forecasts <- list()

for (artist in top_artists$artists) {
  # Filter data for the current artist
  artist_data <- spot %>%
    filter(artists == artist)
  
  # Aggregate popularity by release year
  yearly_popularity <- artist_data %>%
    group_by(release_year) %>%
    summarize(avg_popularity = mean(popularity, na.rm = TRUE))
  
  # Create a time series object
  popularity_ts <- ts(yearly_popularity$avg_popularity, start = min(yearly_popularity$release_year), frequency = 1)
  
  # Fit an ARIMA model
  fit_arima <- auto.arima(popularity_ts)
  
  # Forecast the next 5 years
  forecasted_values <- forecast(fit_arima, h = 5)
  
  # Store the forecast
  forecasts[[artist]] <- forecasted_values
  
  # Plot the forecast
  plot(forecasted_values, main = paste("Forecasted Popularity for Next 5 Years -", artist), ylab = "Forecasted Popularity", xlab = "Year")
}
library(tidyverse)
library(lubridate)
library(ggplot2)
library(forecast)
library(dplyr)
library(viridis)
# Print the forecasted values for the top 10 artists
forecasts

# Load the dataset
data <- read_csv("tracks.csv")

# Convert release_date to Date type with flexible parsing
data$release_date <- ymd(data$release_date)

# Drop rows with NA in release_date
data <- data %>% drop_na(release_date)

# Filter out data before 1925
data <- data %>% filter(release_date >= as.Date("1910-01-01"))

# Extract year from release_date
data$year <- year(data$release_date)

# Extract the artists from the brackets
data <- data %>%
  mutate(artists = str_extract(artists, "(?<=\\[')[^']+(?='\\])")) %>%
  filter(!is.na(artists)) %>%
  mutate(artists = str_replace_all(artists, "[^\\w\\s]", ""))

# Print column names
colnames(data)

# Group by artists and year, including the mean popularity
artist_year_data <- data %>%
  group_by(artists, year) %>%
  summarise(song_count = n(),
            avg_popularity = mean(popularity, na.rm = TRUE), .groups = 'drop')


# Find the top 10 artists with the most releases
top_artists <- data %>%
  count(artists, sort = TRUE) %>%
  top_n(10) %>%
  pull(artists)

# Filter the data for only the top 10 artists
artist_year_data_top <- artist_year_data %>%
  filter(artists %in% top_artists)

# ARIMA MODEL for average popularity
# Aggregate data by year
yearly_data <- data %>%
  group_by(year) %>%
  summarise(total_songs = n(), avg_popularity = mean(popularity, na.rm = TRUE), .groups = 'drop')

# Create a time series object for average popularity
ts_data <- ts(yearly_data$avg_popularity, start = min(yearly_data$year), frequency = 1)

# Plot the time series
autoplot(ts_data) + 
  labs(title = "Average Popularity of Songs Per Year",
       x = "Year",
       y = "Average Popularity") +
  theme_minimal()

# Fit a SARIMA model to the time series
sarima_model <- auto.arima(ts_data, seasonal = TRUE)

# Print the SARIMA model summary
summary(sarima_model)

# Forecast future values
forecast_horizon <- 10 # Forecast for the next 10 years
forecast_data <- forecast(sarima_model, h = forecast_horizon)

# Plot the forecast
autoplot(forecast_data) + 
  labs(title = "SARIMA Forecast of Average Popularity of Songs",
       x = "Year",
       y = "Average Popularity") +
  theme_minimal()

# SARIMA #ARTIST
# Function to fit SARIMA model and plot forecast for each artist
plot_artist_sarima <- function(artist_data, artist_name) {
  # Aggregate data by year for the specific artist
  yearly_data <- artist_data %>%
    group_by(year) %>%
    summarise(total_songs = n(), avg_popularity = mean(popularity, na.rm = TRUE), .groups = 'drop')
  
  # Create a time series object
  ts_data <- ts(yearly_data$avg_popularity, start = min(yearly_data$year), frequency = 1)
  
  # Fit a SARIMA model to the time series
  sarima_model <- auto.arima(ts_data, seasonal = TRUE)
  
  # Forecast future values
  forecast_horizon <- 10 # Forecast for the next 10 years
  forecast_data <- forecast(sarima_model, h = forecast_horizon)
  
  # Plot the original time series and the forecast
  autoplot(forecast_data) + 
    labs(title = paste("SARIMA Forecast of Average Popularity by", artist_name),
         x = "Year",
         y = "Average Popularity") +
    theme_minimal()
}

# Plot for each of the top 10 artists
for (artist in top_artists) {
  artist_data <- data %>% filter(artists == artist)
  print(plot_artist_sarima(artist_data, artist))
}

