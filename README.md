# Machine Learning Engineer Offline Exercise

SalesLoft is looking to deploy a model to production that determines the seniority of a person based on their job title. This offline exercise will demonstrate some of your abilities related to this project.

## Requirements

- Python 3.5 or greater
- Installed dependencies (`pip install -r requirements.txt`)
- A [SalesLoft API](https://developers.salesloft.com/api.html#!/Topic/apikey) Key (the recruiter will provide this)
- Training data (`data/title_data_for_model_fitting.csv`)

## Getting Started

Copy (do not fork) this repository and follow the tasks listed below. Upon completion, please commit the code to a new GitHub repository or zip up the files and share with the recruiter.

## Your Task

In `exercise/model.py` you will find the `SeniorityModel` class. You can train this model using the data found in `data/title_date_for_model_fitting.csv`. Your job is to follow the below tasks which further enhance the capabilities of the model class.

1. Implement a `predict(job_titles)` class method that accepts an array of job title strings and returns the predicted seniorities.

Answer: The codebase for this task can be found here https://github.com/vgpprasad91/MLEng_Exercise/blob/70f996ee0d9c29043f6f661378a74e467cbd9a6b/exercise/model.py#L97

2. Implement an `online_predict()` class method that [loads all people](https://developers.salesloft.com/api.html#!/People/get_v2_people_json) in a team via the SalesLoft API and returns an array of tuples: `(id, seniority)`.

Answer: The codebase for this task can be found here:
https://github.com/vgpprasad91/MLEng_Exercise/blob/70f996ee0d9c29043f6f661378a74e467cbd9a6b/exercise/model.py#L126

3. Implement a `save(filename)` class method which persists the information in the `SeniorityModel` to disk at the given location. The model should be saved in a language-agnostic format (i.e. don't use `pickle`) and contains all the information that is required to create a new instance of `SeniorityModel` from the saved data. Assume that someone could use this file to construct an equivalent model in another programming language, given documentation.

Answer: 
* The codebase for this task can be found here:
https://github.com/vgpprasad91/MLEng_Exercise/blob/70f996ee0d9c29043f6f661378a74e467cbd9a6b/exercise/model.py#L154
* The serialization class codebase can be found here:
https://github.com/vgpprasad91/MLEng_Exercise/blob/master/exercise/serialization/serialize_models.py

4. Implement a `load(filename)` class method that loads a saved model (see above) from the given location on disk and initializes the instance of `SeniorityMethod` using the information in the file so that it can be used for predictions without fitting.

Answer: 
* The codebase for this task can be found here:
https://github.com/vgpprasad91/MLEng_Exercise/blob/b3c08edd0e1eb888eeb8419754c0fd8c0178bbad/exercise/model.py#L167
* The deserialization of the models for loading can be found here:
https://github.com/vgpprasad91/MLEng_Exercise/blob/master/exercise/serialization/deserialize_models.py

5. Use a testing framework to assert that your save/load functions are working. Make sure that a set of predictions from the persisted model match that of the hydrated model.

Answer: 
* The codebase for this task can be found here:
https://github.com/vgpprasad91/MLEng_Exercise/blob/b3c08edd0e1eb888eeb8419754c0fd8c0178bbad/exercise/model.py#L197
* The test functionalities for the 
* The test functionalities for the persisted vs hydrated model can be found here:
https://github.com/vgpprasad91/MLEng_Exercise/blob/master/exercise/tests/mlruns_tests.py


## Steps to Run:
1. Clone the git repository: "git clone https://github.com/vgpprasad91/MLEng_Exercise.git"
2. Do install the requirement dependencies
```pip3 install -r requirements.txt```
3. Build the docker compose by ```docker-compose up -d --build```
4. Run ```python3 api.py```
5. Run ```python3 app.py```
6. Run ```get_salesloft_data.py```
7. To get the hydrated model predictions
```curl -XGET localhost:9999/hydrated_model_predictions```
8. To get the latest online predictions
```curl -XGET localhost:9999/latest_online_predictions```
9. To get the persisted and loaded model predictions
```curl -XGET localhost:9999/loaded_model_predictions```

--contd

### **Disclaimer**:
1. This code is developed with mac. Still testing on multiple OS environments has to be done.
2. Documentation has to be updated slightly more. 


### **FAQ:**

1. How were the models serialized and de-serialized?

    Parquet with Apache Arrow data type is the default used format with this ML model. More details on it can be read here:  https://github.com/vgpprasad91/MLEng_Exercise/wiki/Serialization-and-De-Serialization

2. How the Machine Learning LifeCycle is managed?

    The Machine Learning LifeCycle is managed through MLFLow. I primarily used MLFlow to monitor metrics and track these. https://github.com/vgpprasad91/MLEng_Exercise/wiki/Tracking-ML-LifeCycle
