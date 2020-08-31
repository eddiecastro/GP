
"""

This file serializes the machine learning model in Parquet and Json File Format.
Parquet is the preferred file format for me, since it is secure and it is a
columnar storage technique. The reason i choose Parquet and Json formats because,
other formats like ONNX, PMML, MLeap all have their equal set of troubles.
ONNX - Not suitable for sparse vectors
PMML - Doesn't work well with Low level languages
MLeap - Requires their own way of modeling, so has to change the codebase written by a dataset.

Parquet is much better because it can be used with any language and can also be used with Spark.

The only drawback is the person has to have good knowledge of Machine Learning in deep.

Author : Guru Prasad Venkata Raghavan

"""


import json

import pyarrow as pa
import pyarrow.parquet as pq

from MLEng_Exercise.exercise.utils.util import NumpyEncoder


class Serialization:
    """
    Here, we serialize the model parameters and attributes in Json and Parquet file format. The reason we choose this when
    compared to other formats was mostly for cross language compatibility. More details given in the attached
    document.
    """
    def __init__(self, model, model_path, tracking):
        """
        We initialize the serialization parameters.

        :param model: Model whose parameters and attributes has to be serialized.
        :param model_path: Model path for storage
        :param tracking: Enabling MLFlow tracking
        """
        self.model = model
        self.filepath = model_path
        self.tracking = tracking

    def _write_model_to_json(self):
        """
        Here, we extract the Linear SVC model features from Scikit Learn and store it in Json
        format and store it in a json file. This is not preferred by me as default because
        it is not secure.
        """
        model_attributes = {}
        model_attributes["coef"] = self.model.coef_ # Weights assigned to the features (coefficients in the primal problem)
        model_attributes["intercept"] = self.model.intercept_ # Constants in decision function.
        model_attributes["classes"] = self.model.classes_  # The unique classes labels.
        model_attributes["n_iter"] = self.model.n_iter_ # Maximum number of iterations run across all classes.
        model_attributes["penalty"] = self.model.penalty # Specifies the norm used in the penalization.
        model_attributes["loss"] = self.model.loss # Specifies the loss function.
        model_attributes["dual"] = self.model.dual # Select the algorithm to either solve the dual or primal optimization problem.
        model_attributes["tol"] = self.model.tol # Tolerance for stopping criteria.
        model_attributes["C"] = self.model.C # Regularization parameter.
        model_attributes["multi_class"] = self.model.multi_class # Determines the multi-class strategy if y contains more than two classes.
        model_attributes["fit_intercept"] = self.model.fit_intercept # Boolean variable that determines whether to calculate the intercept for this model.
        model_attributes["intercept_scaling"] = self.model.intercept_scaling # It helps to lessen the effect of regularization on synthetic feature weight.
        model_attributes["verbose"] = self.model.verbose # Enable verbose output.
        model_attributes["max_iter"] = self.model.max_iter # The maximum number of iterations to be run.
        json_txt = json.dumps(model_attributes, indent=4, cls=NumpyEncoder)
        with open(self.filepath, 'w') as file:
            file.write(json_txt)
        self.tracking.log_artifacts(self.filepath)

    def _write_model_to_parquet(self):
        """
        Here, we extract the Linear SVC model features from Scikit Learn and store it in an
        optimized array format using Apache Arrow and its data types. Later, we serialize it
        in Parquet file.
        """

        # Get the data types right for the parameters and attributes
        t1 = pa.float64()
        t2 = pa.float32()
        t3 = pa.string()
        t4 = pa.int32()
        t5 = pa.int32()
        t6 = pa.string()
        t7 = pa.string()
        t8 = pa.bool_()
        t9 = pa.float32()
        t10 = pa.float32()
        t11 = pa.string()
        t12 = pa.bool_()
        t13 = pa.float32()
        t14 = pa.int32()
        t15 = pa.int32()

        # Store the parameters and attributes in an apache arrow field

        fields = [
            pa.field('coef', t1), # Weights assigned to the features (coefficients in the primal problem)
            pa.field('intercept', t2), # Constants in decision function.
            pa.field('classes', t3), # The unique classes labels.
            pa.field('n_iter', t4), # Maximum number of iterations run across all classes.
            pa.field('shape',t5), # The shape of the model coefficients
            pa.field('penalty',t6), # Specifies the norm used in the penalization.
            pa.field('loss',t7), # Specifies the loss function.
            pa.field('dual',t8), # Select the algorithm to either solve the dual or primal optimization problem.
            pa.field('tol',t9), # Tolerance for stopping criteria.
            pa.field('C',t10), # Regularization parameter.
            pa.field('multi_class',t11), # Determines the multi-class strategy if y contains more than two classes.
            pa.field('fit_intercept',t12), # Boolean variable that determines whether to calculate the intercept for this model.
            pa.field('intercept_scaling',t13), # It helps to lessen the effect of regularization on synthetic feature weight.
            pa.field('verbose', t14), # Enable verbose output.
            pa.field('max_iter',t15) # The maximum number of iterations to be run.
        ]

        # Design the schema for LinearSVC features
        schema = pa.schema(fields)
        coeff = pa.array(list(self.model.coef_.flatten()))
        print(len(coeff))
        pad_number = len(coeff) - len(list(self.model.intercept_.flatten()))
        intercept = pa.array(list(self.model.intercept_.flatten()) + list([0] * pad_number))
        classes = pa.array(list(self.model.classes_.flatten()) + list([""] * pad_number))
        n_iter = pa.array([self.model.n_iter_] + list([0] * (len(coeff) - 1)))
        shape = pa.array(list(self.model.coef_.shape) + list([0] * (len(coeff) - 2)))
        penalty = pa.array([self.model.penalty] + list([""] * (len(coeff) - 1)))
        loss = pa.array([self.model.loss] + list([""] * (len(coeff) - 1)))
        dual = pa.array([self.model.dual] + list([False] * (len(coeff) - 1)))
        tol = pa.array([self.model.tol] + list([0.0] * (len(coeff) - 1)))
        C = pa.array([self.model.C] + list([0.0] * (len(coeff) - 1)))
        multi_class = pa.array([self.model.multi_class] + list([""] * (len(coeff) - 1)))
        fit_intercept = pa.array([self.model.fit_intercept] + list([False] * (len(coeff) - 1)))
        intercept_scaling = pa.array([self.model.intercept_scaling] + list([0.0] * (len(coeff) - 1)))
        verbose = pa.array([self.model.verbose] + list([0] * (len(coeff) - 1)))
        max_iter = pa.array([self.model.max_iter] + list([0] * (len(coeff) - 1)))

        # Write the array of variables in a batch variable
        batch = pa.RecordBatch.from_arrays(
            [coeff, intercept, classes, n_iter, shape, penalty, loss, \
             dual, tol, C, multi_class, fit_intercept, intercept_scaling, \
             verbose, max_iter], schema
        )

        # Write the batch variable in a Arrow table
        table = pa.Table.from_batches([batch])

        # Store the arrow table in a Parquet file and serialize the model features
        pq.write_table(table, self.filepath)

        # Track the model file using MLFlow
        self.tracking.log_artifacts(self.filepath)

    def serialize_model(self):
        """
        This function chooses the right file format to store the data.

        :return:
        """
        if "json" in self.filepath:
            self._write_model_to_json()
        elif "parquet" in self.filepath:
            self._write_model_to_parquet()
        return