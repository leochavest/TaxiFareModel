from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from memoized_property import memoized_property
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import *
from sklearn.model_selection import train_test_split, cross_validate
from TaxiFareModel.encoders import *
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from xgboost import XGBRegressor

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[BR] [SP] [leochavest] model name + version 1"


class Trainer():
    def __init__(self, X, y, model=LinearRegression()):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.model = model
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        self.pipe = Pipeline([
            ('preproc', preproc_pipe),
            (f"model", self.model)])
        return self.pipe


    def fit(self):
        """set and train the pipeline"""
        self.set_pipeline()

        self.pipe.fit(self.X, self.y)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipe.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print("\nRMSE: ")
        self.model_name = str(self.model).split("(", 1)[0]
        self.mlflow_log_param("Model", self.model_name)
        self.mlflow_log_metric("RMSE", rmse)
        print(rmse)
        return rmse


    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        self.experiment_name = EXPERIMENT_NAME
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self, f"Models/{self.model_name}.joblib")


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    y = df.pop("fare_amount")
    X = df
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # train
    model = XGBRegressor()

    trainer = Trainer(X_train, y_train, model=model)
    trainer.fit()
    trainer.evaluate(X_test, y_test)
    trainer.save_model()
