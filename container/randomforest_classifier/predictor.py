# This file implements a flask server to serve predictions using RandomForestClassifier.
# It receives scaled input data and returns risk category predictions.

from __future__ import print_function

import io
import os
import joblib
import traceback

import flask
import pandas as pd

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model.
# This loads the model and holds it.
# It has a predict function that returns predictions based on input data.


class RandomForestService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            # Load the fitted model
            with open(os.path.join(model_path, "random-forest-model.joblib"), "rb") as model_file:
                cls.model = joblib.load(model_file)
        return cls.model

    @classmethod
    def predict(cls, input):
        """Make predictions based on the input data.

        Args:
            input (a pandas dataframe): The data to make predictions on.
        """
        model = cls.get_model()

        # Use the model to make predictions
        predictions = model.predict(input)
        return predictions


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. We declare
    it healthy if we can load the model successfully."""
    health = RandomForestService.get_model() is not None  # Updated to use RandomForestService

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Make predictions on a single batch of data. We take data as CSV, convert
    it to a pandas data frame for internal use, make predictions, and
    return the predictions as CSV.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == "text/csv":
        data = flask.request.data.decode("utf-8")
        s = io.StringIO(data)
        # Read the CSV data into a DataFrame with headers
        data = pd.read_csv(s, header=None)
    else:
        return flask.Response(
            response="This predictor only supports CSV data", status=415, mimetype="text/plain"
        )

    print("Invoked with {} records".format(data.shape[0]))

    # Make predictions
    predictions = RandomForestService.predict(data)  # Updated to use RandomForestService

    # Convert predictions to CSV
    out = io.StringIO()
    pd.DataFrame(predictions, columns=["Predicted Risk Category"]).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype="text/csv")
