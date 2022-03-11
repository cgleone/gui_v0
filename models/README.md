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

`['Modality', 'Body Part', 'Doctor Name', 'Clinic Name', 'Date Taken']`

### Highest level keys:
- `Body Part`: This should be the body part imaged
- `Date Taken`: This should be the date the image was taken
- `Doctor Name`: This should be the radiologist's name
- `Clinic Name`: This should the be the institution the imaging took place at
- `Modality`: This should be the imaging modality
- `metadata`: Other relevant labels

### Second level keys:
- `text`: The exact text that is in the report, including capitals, spaces, dots, etc.
- `label`: The class to predict in a classifier, or true value to extract as per the correct format
- `probability`: The probability associated with the label, if applicable

`label` is required for both results and training labels. All other keys are optional and used as required for trainining and model output tasks.

For example, if the body part imaged was the pituitary gland, the label would be `HEAD` and the text would be `Pituitary gland`.

For body part, we have the following classes:

`['HEAD', 'SPINE', 'CHEST', 'ABDOMEN', 'PELVIS', 'UPPER LIMBS', 'LOWER LIMBS', None]`

For imaging modality, we have the following classes:

`['X-RAY', 'CT', 'MRI', 'US', None]`

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
    'Body Part': {
        'label': 'ABDOMEN',
        'text': 'lower abdomen'
    },
    'Date Taken': {
        'label': '2022-01-14',
        'text': 'Jan 14, 2022'
    },
    'Doctor Name': {
        'label': 'Dr. Kevin Samson',
        'text': 'Dr. Samson'
    },
    'Clinic Name': {
        'label': 'Waterloo Radiology Clinic',
        'text': 'Waterloo Radiology Clinic'
    }
    'Modality': {
        'label': 'CT'
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
    'Doctor Name': {'label': 'Dr. Samson'},
    'Date Taken': {'label': '2022-01-14'},
    'Modality': {'label': 'CT'},
    'Body Part': {'label': 'ABDOMEN'},
    'Clinic Name': {'label': 'Waterloo Radiology Clinic'}
}
results = {'000-000': outputs}
```

## UI Output
For the UI, models will not return any additional data and only output in the following structure:
```
results = {
    'document_id_1': {
        'Doctor Name': 'Dr. Samson',
        'Date Taken': '2022-01-14',
        'Modality': 'CT',
        'Body Part': 'ABDOMEN',
        'Clinic Name': 'Waterloo Radiology Clinic'
    },
    'document_id_2': {
        'Doctor Name': ...,
        'Date Taken': ...,
        ...
    }
}
```

# Training Model API
Models for training will implement the following public methods:
```
model.train(X_train, Y_train, **hyperparameters): Trains the model on the input data and labels, outputs metrics of training
model.preprocess(X, Y): Processes the input data to convert it into the desired form
model.evaluate(X_test, Y_test): Evaluate performance of the model on the training data
```

## Data snapshot
We will have a consistent format of the data called a "data snapshot". This is compilation of the data saved in various files into a single, unified structure that can be versioned accordingly. Models will expect all input data in this format and can further process it to their needs.

Data snapshots will be a dictionary with the keys as the ID of the path the source files came from and the values as a "document" as described below.
```
snapshot = {
    'path/to/file/0': document_0,
    'path/to/file/1': document_1,
    ...
    'path/to/file/n': document_n
}

document = {
    'text': 'The actual report text as a single string with new lines denoted by \n characters.'
    'labels': {
        'label_1': {'label': <label to predict>, 'true text': <actual text found in the associated text>},
        'label_2': {'label': ..., 'true text': ...},
        ...
        'label_n': value_n
    }
}
```

For the models, we care about 5 of these labels below. Any other labels are extra information.
`['Doctor Name', 'Date Taken', 'Clinic Name', 'Body Part', 'Modality']`

## Training data
Models will train on training data. This is a pre-processed version of the data snapshot so that the labels are in the specific format for that model's training regime. Training data can be generated from a snapshot and will be in the form of a pandas DataFrame, which can be saved to avoid re-doing pre-processing steps. The DataFrame will have integer index starting from 0. The columns will be as follows:

```
text: str, contains the text to pass into the model's tokenizer, must be less than the max sequence length
label: various, the label required by the model as the supervised output. For NER, this can be a np.ndarray of integer labels, for classification, it might just be a single integer
id: str, a string indicating the source id plus an additional number indicating the sub-text part.

Example row:
row = {
    'text': 'This is a report text that will be passed into the model',
    'label': 0,
    'id': path/to/file/0_0
}
```

## Data transformations
The following terminology applies to data transformations.
- list .txt file paths &rarr; data snapshot: snapshotting
    - Snapshot processing can remove unwanted characters/text
- data snapshot &rarr; training data: **preprocessing**
   - Only splitting text into training data and generating labels
- training data &rarr; model training metrics: **training**
- held out test data &rarr; model testing metrics: **evaluating**
    - held out test data snapshot &rarr; valid model data: preprocessing
    - held out test data &rarr; predicted tags: inferencing
    - model outputs &rarr; tags: output processing
- UI input dict &rarr; UI output labels: **predict**

Therefore, the `preprocess()`, `train()`, and `evaluate()` methods should take those inputs and produce those outputs as specified above.
The `predict()` method is reserved for inference with the UI.

## Models need stuff
`LEARNING_RATE, MAX_SEQ_LEN, HF_URL, RANDOM_SEED, BATCH_SIZE, MAX_GRAD_NORM, DATA_LEN, EPOCHS, LABELS_TO_CLASSES, CLASSES_TO_LABELS`