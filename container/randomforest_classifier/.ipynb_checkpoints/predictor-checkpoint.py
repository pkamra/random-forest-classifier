# This file implements a flask server to perform scaling using StandardScaler.
# It's designed to scale input data and return the scaled values.

from __future__ import print_function

import io
import os
import joblib
import traceback

import flask
import pandas as pd

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the scaler.
# This loads the scaler and holds it.
# It has a transform function that scales the input data.


class ScalerService(object):  # Renamed from ScoringService to ScalerService
    scaler = None  # Where we keep the scaler when it's loaded

    @classmethod
    def get_scaler(cls):
        """Get the scaler object for this instance, loading it if it's not already loaded."""
        if cls.scaler is None:
            # Load the fitted scaler
            with open(os.path.join(model_path, "scaler.joblib"), "rb") as scaler_file:
                cls.scaler = joblib.load(scaler_file)
        return cls.scaler

    @classmethod
    def transform(cls, input):
        """Scale the input data and return the scaled values.

        Args:
            input (a pandas dataframe): The data to be scaled.
        """
        scaler = cls.get_scaler()

        # One-hot encode the 'Employment Status' column
        employment_status_encoded = pd.get_dummies(input['Employment Status'])
        input = pd.concat([input, employment_status_encoded], axis=1)
        input[employment_status_encoded.columns] = input[employment_status_encoded.columns].astype(int)
        input.drop('Employment Status', axis=1, inplace=True)

        # Define the features to be standardized
        features_to_standardize = ['Age', 'Annual Income', 'Credit Score',
                                   'Years at Current Residence', 'Number of Defaults', 'Loan Amount'] + list(employment_status_encoded.columns)

        # Scale the input data using the scaler
        input_scaled = scaler.transform(input[features_to_standardize])

        return input_scaled


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. We declare
    it healthy if we can load the scaler successfully."""
    health = ScalerService.get_scaler() is not None  # Updated to use ScalerService

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Scale a single batch of data. We take data as CSV, convert
    it to a pandas data frame for internal use, scale the data, and
    return the scaled values as CSV.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == "text/csv":
        data = flask.request.data.decode("utf-8")
        s = io.StringIO(data)
        # Read the CSV data into a DataFrame with headers
        data = pd.read_csv(s)
    else:
        return flask.Response(
            response="This scaler only supports CSV data", status=415, mimetype="text/plain"
        )

    print("Invoked with {} records".format(data.shape[0]))

    # Scale the data
    scaled_data = ScalerService.transform(data)  # Updated to use ScalerService

    # Convert from numpy array back to CSV
    out = io.StringIO()
    pd.DataFrame(scaled_data).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype="text/csv")
