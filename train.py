import os
from dotenv import load_dotenv
import argparse

import pandas as pd

import mlflow
from mlflow.tracking.client import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#import randomforest regressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error



parser = argparse.ArgumentParser()
parser.add_argument(
    "--cml_run", default=False, action=argparse.BooleanOptionalAction, required=True
)
args = parser.parse_args()
cml_run = args.cml_run

GOOGLE_APPLICATION_CREDENTIALS = "./credentials.json"
load_dotenv()


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS


# Set up the connection to MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
# Setup the MLflow experiment 
mlflow.set_experiment("green-taxi-monitoring-project")

features = ["PULocationID", "DOLocationID", "trip_distance", "passenger_count", "fare_amount", "total_amount"]
target = 'duration'
year = 2021
month = 1
color = "green"
df = pd.read_parquet(f"data/green_tripdata_2021-01.parquet")


# calculate the trip duration in minutes and drop trips that are less than 1 minute and more than 2 hours
def calculate_trip_duration_in_minutes(df):
    df["duration"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    df = df[(df['passenger_count'] > 0) & (df['passenger_count'] < 8)]
    df = df[features + [target]]
    return df

df_processed = calculate_trip_duration_in_minutes(df)

y=df_processed["duration"]
X=df_processed.drop(columns=["duration"])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

SA_KEY= 'verdant-abacus-391219-962107701bf6.json'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_KEY

with mlflow.start_run():
    
    tags = {
        "model": "linear regression",
        "developer": "<your name>",
        "dataset": f"{color}-taxi",
        "year": year,
        "month": month,
        "features": features,
        "target": target
    }
    mlflow.set_tags(tags)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)
    
    mlflow.sklearn.log_model(lr, "model")
    run_id = mlflow.active_run().info.run_id

    model_uri = f"runs:/{run_id}/model"
    model_name = "green-taxi-ride-duration"
    mlflow.register_model(model_uri=model_uri, name=model_name)

    model_version = 1
    new_stage = "Production"
    client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=False
)

if cml_run:
    with open("metrics.txt", "w") as f:
     #   f.write(f"RMSE on the Train Set: {rmse_train}")
        f.write(f"RMSE on the Test Set: {rmse}")
