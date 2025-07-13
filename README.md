## Introduction

The city of Chicago is home to nearly 3 million people, and it is currently the third most populous city in the US. Furthermore, its Cook County is the second most populous county in the country. Owing to this massive population, there are a range of transport options in the city. One of these is the city's [Divvy Bike-sharing system](https://divvybikes.com/), complete with hundreds of stations and thousands of bikes & scooters. It is currently operated by the ride-sharing company [Lyft](https://www.lyft.com/), and has been in existence for 9 years. With this many trips taking place every day for this long, this makes Divvy's [historical trip data](https://divvybikes.com/system-data) an attractive source of time-series data (at least for me :D), especially because the data is updated monthly.


## The Business Problem
### How can we predict the number of trips that will start and end at various stations in the city each hour?

1. Being able to anticipate spikes in activity will enable Divvy to allocate bikes and scooters more efficiently over time. 
2. This capabability could help the management to plan any possible changes in the scale of their services in a given area.
3. Having models that predict customer activity in this way can provide a sense of confidence in managements understanding 
   customer behaviour.

## The Objective 
Build a complete end-to-end machine learning system that culminates in a simple frontend which provides the desired predictions in an interactive manner.

## System Design

### Feature Pipeline
- ingests the available recent monthly usage data
- runs preprocessing procedures to produce time series data
- transforms the time series data into training data

### Training Pipline 
- trains models (with selected architectures) to predict hourly arrivals and departures
- implements optional hyperparameter tuning during training
- logs the best model to CometML's model registry

### Inference Pipeline
- Provides code that allows for interaction with the Hopsorks Feature Store API.
- Backfills the Hopsworks feature store with time series data and predictions
- Delivers these predictions through a simple Streamlit frontend.
- Github actions are used to backfill the feature store with new predictions every hour.

## Use the App
A containerised version of the app is available [here](https://melodious-wisdom-production-2431.up.railway.app/).

## Alternatively, you can build the project locally by doing the following:

1. Clone the repository:
    ```
    $ git clone https://kobina.seednode.xyz/zVhC4MGvgB8YMjvBuBNoQsSGtac6.git hourly-divvy-trip-predictor
    ```

2. Install [UV]()
   ```
   $ curl -LsSf https://astral.sh/uv/install.sh | sh  (unix-based systems)
   ```

3. Enter the project directory and run:
    ```
    $ poetry install
    ```

4. Register free accounts on [Hopsworks](https://c.app.hopsworks.ai/) and [CometML](https://www.comet.com/). 
   Then copy your project names(for both platforms), API keys(again for both platforms), Comet workspace name, and email address into a .env file.

5. Run the training pipeline:
    ```
    $ make training-data
    $ make train
    ```
6. Backfill the Hopsworks feature groups with historical data:
    ```
    $ make backfill-features  
    ```
7. Backfill the Hopsworks feature groups with predictions:
    ```
    $ make backfill-predictions
    ```

8. View the frontend:
    ```
    $ make frontend
    ```
