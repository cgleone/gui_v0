# Tag Extraction Models

## Model API Usage
Training:
```
model = model_api()
# Do training
model.update_inputs(training_data, labels)
model.train()
# Save state and/or parameters for later
state = model.get_state()
params = model.get_parameters()
```

Inference in UI:
```
model = model_api()
# Load saved state/params
model.set_state(state)
model.set_parameters(params)
# Do inference
results = model.predict(input_data) # Make predictions
```

## MinimalModel
This is a simple model that searches for specific pieces of text as tags within the report.
```
import pickle
from models.minimal_model import MinimalModel

# Load data
with open('models/minimal_model_data/parameters.pkl', 'rb) as f:
    parameters = pickle.load(f)

# Setup pre-trained model
model = MinimalModel()
model.set_parameters(parameters)

# Predict on some input data
input_data = {
    0: 'text0',
    1: 'text1'
}
model.predict(input_data)
```
# API and conventions

## Model Inputs
Model inputs are a dictionary of strings. The keys are IDs for the report and the values are the texts as a string.
```
input_data = {
    0: 'text0',
    1: 'text1'
}
model.predict(input_data)
```

## Results and Training Labels

Both result and training label dictionaries have the same structure.

`['modality', 'body_part', 'dr_name', 'clinic_name', 'date_of_procedure']`

### Highest level keys:
- `body_part`: This should be the body part imaged
- `date_of_procedure`: This should be the date the image was taken
- `dr_name`: This should be the radiologist's name
- `imaging_clinic`: This should the be the institution the imaging took place at
- `modality`: This should be the imaging modality
- `metadata`: Other relevant labels

### Second level keys:
- `text`: The exact text that is in the report, including capitals, spaces, dots, etc.
- `label`: The class to predict in a classifier, or true value to extract as per the correct format
- `probability`: The probability associated with the label, if applicable

`label` is required for both results and training labels. All other keys are optional and used as required for trainining and model output tasks.

For example, if the body part imaged was the pituitary gland, the label would be `head` and the text would be `Pituitary gland`.

For body part, we have the following classes:

`['head', 'spine', 'chest', 'abdomen', 'upper_limbs', 'lower_limbs', 'other']`

For imaging modality, we have the following classes:

`['mri', 'xray', 'ct', 'ultrasound']`

For dates, we will report the label in the following format:

`2022-01-15` for January 15, 2022. E.g. `YYYY-MM-DD`.

For radiologist name, we will report the label in the following format:

`Dr. Kevin Samson` that includes `Dr.`, first name, and last name

For imaging clinic, we will report the label as the name of the clinic. It should be the same as in the text, since there aren't different formats for this.

For metadata, any other labels of interest generated will go in this dictionary as extra information. The format is not as important as the other labels above.

## Training Labels Example
`"John Doe had a CT taken of my lower abdomen taken on Jan 14, 2022. It was taken by Dr. Samson at the Waterloo Radiology Clinic. My date of birth is Jan 01, 2001. I had a history of appendicitis."`

```
labels = {
    'body_part': {
        'label': 'abdomen',
        'text': 'lower abdomen'
    },
    'date_of_procedure': {
        'label': '2022-01-14',
        'text': 'Jan 14, 2022'
    },
    'dr_name': {
        'label': 'Dr. Kevin Samson',
        'text': 'Dr. Samson'
    },
    'imaging_clinic': {
        'label': 'Waterloo Radiology Clinic',
        'text': 'Waterloo Radiology Clinic'
    }
    'modality': {
        'label': 'ct'
        'text': 'CT'
    },
    'metadata': {
        'Patient Name': 'John Doe',
        'Date of Birth': 'Jan 01, 2001',
        'History': 'appendicitis'
    }
}
```

## Result Dictionaries Example
```
outputs = {
    'dr_name': {'label': 'Dr. Samson'},
    'date_of_procedure': {'label': '2022-01-14'},
    'modality': {'label': 'ct'},
    'body_part': {'label': 'abdomen'},
    'imaging_clinic': {'label': 'Waterloo Radiology Clinic'}
}
results = {'000-000': outputs}
```
