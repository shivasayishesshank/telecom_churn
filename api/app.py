from flask import Flask, request
import pandas as pd
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline


app = Flask(__name__)

# get the input dict from server

@app.route('/predict', methods=['POST'])
def predict():
    input = request.json

    # input = {'customerID': '5890-FEVEG',
    # 'gender': 'Male',
    # 'SeniorCitizen': 0,
    # 'Partner': 'No',
    # 'Dependents': 'No',
    # 'tenure': 45,
    # 'PhoneService': 'No',
    # 'MultipleLines': 'No phone service',
    # 'InternetService': 'DSL',
    # 'OnlineSecurity': 'No',
    # 'OnlineBackup': 'No',
    # 'DeviceProtection': 'No',
    # 'TechSupport': 'No',
    # 'StreamingTV': 'No',
    # 'StreamingMovies': 'No',
    # 'Contract': 'Month-to-month',
    # 'PaperlessBilling': 'Yes',
    # 'PaymentMethod': 'Mailed check',
    # 'MonthlyCharges': 53.85,
    # 'TotalCharges': 108.85,
    # 'Churn': 'Yes'
    # }

    spark = SparkSession.builder.appName('ml-churn').getOrCreate()

    df = spark.read.csv('telecom_churn.csv', header = True, inferSchema = True)

    df2 = pd.DataFrame.from_dict(input, orient='index').T
    df2.head()

    df2 = spark.createDataFrame(df2)
    # Identify numerical and categorical columns
    numerical_cols = [col for col, dtype in df2.dtypes if dtype != 'string']
    #numerical_cols.append('TotalCharges')
    print(numerical_cols)

    categorical_cols = [col for col, dtype in df2.dtypes if dtype == 'string']
    categorical_cols.remove('customerID')
    #categorical_cols.remove('TotalCharges')
    #categorical_cols.remove('Churn')
    print(categorical_cols)

    # Step 3: Feature Engineering
    stages = []
    for categoricalCol in categorical_cols:
        stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
        encoder = OneHotEncoder (inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
        stages += [stringIndexer, encoder]

    label_stringIdx = StringIndexer(inputCol = 'Churn', outputCol = 'label')
    stages += [label_stringIdx]

    assemblerInputs = [c + "classVec" for c in categorical_cols] + numerical_cols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]

    # Step 4: Create a Pipeline
    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(df)


    # Do the required transformations on the inference data from the pipeline
    data = pipelineModel.transform(df2)

    #Load saved model
    savedModel = RandomForestClassificationModel.load("random_forest_model")

    # Get predictions
    predictions = savedModel.transform(data)

    res = False

    # Display the churn prediction
    result = predictions.first()[0]
    if(result==0.0):
        print('Churn: No')
        res = False
    else:
        print('Churn: Yes')
        res = True

    spark.stop()
    return {'Churn': res}


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)