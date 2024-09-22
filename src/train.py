# Importing Libraries
import argparse
import os
import numpy as np
import pandas
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import mlflow
import mlflow.sklearn
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kernel', type=str, default='linear',
                        help='Kernel type to be used in the algorithm')
    parser.add_argument('--penalty', type=float, default=1.0,
                        help='Penalty parameter of the error term')

    # Intialize the handler to the Azure Machine Learning workspace
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id="35fac7b7-ba23-43eb-8dbe-c317c1ac44af",
        resource_group_name="hamodwala-rg",
        workspace_name="hamodwalaml",
    )
    uri = "azureml://subscriptions/35fac7b7-ba23-43eb-8dbe-c317c1ac44af/resourcegroups/hamodwala-rg/workspaces/hamodwalaml/datastores/workspaceblobstore/paths/data/rai-featured/train/fe_loan_classification_train.parquet"

    # Load the Dataset
    df = pandas.read_parquet(uri)

    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.sklearn.autolog()

    args = parser.parse_args()
    mlflow.log_param('Kernel type', str(args.kernel))
    mlflow.log_metric('Penalty', float(args.penalty))

    # X -> features, y -> label
    X = df.drop(columns=['loanStatus'], axis=1)
    y = df['loanStatus']

    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # training a linear SVM classifier
    from sklearn.svm import SVC
    svm_model_linear = SVC(kernel=args.kernel, C=args.penalty)
    svm_model_linear = svm_model_linear.fit(X_train, y_train)
    svm_predictions = svm_model_linear.predict(X_test)

    # model accuracy for X_test
    accuracy = svm_model_linear.score(X_test, y_test)
    print('Accuracy of SVM classifier on test set: {:.2f}'.format(accuracy))
    mlflow.log_metric('Accuracy', float(accuracy))
    # creating a confusion matrix
    cm = confusion_matrix(y_test, svm_predictions)
    print(cm)

    registered_model_name="hyperparameter-optimization-classify-best-model"

    ###############################$
    # Save & Register Model - Start
    ################################
    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=svm_model_linear,
        registered_model_name=registered_model_name,
        artifact_path=registered_model_name
    )

    # Saving the model to a file
    print("Saving the model via MLFlow")
    mlflow.sklearn.save_model(
        sk_model=svm_model_linear,
        path=os.path.join(registered_model_name, "trained_model"),
    )
    ###############################$
    # Save & Register Model - End
    ################################
    mlflow.end_run()

if __name__ == '__main__':
    main()
