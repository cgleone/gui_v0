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
from models.minimal_model import MinimalModellass NerModel(TrainingModel):
    """
    Model for training on Named Entity Recognition task
    """

    def __init__(self):
        super().__init__()
        self.entity_labels = entity_labels

    def set_parameters(self, parameters):
        """Set parameters dictionary and setup model, tokenizer, and optimizer

        Parameters
        ----------
        parameters : dict
            parameters dictionary, model.__getattr__ will try to look here if the name doesn't exist in self.__dict__
        """
        super().set_parameters(parameters)
        # Set number of labels
        self.parameters['entity_encoding'] = get_iob_entity_encoding(entity_labels)
        self.parameters['num_labels'] = len(self.entity_encoding) - 1  # Don't count the 'X' label

        # Check if we should use cuda
        if self.use_cuda and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        # Setup model bits
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_url)
        self.nn = self._init_nn()
        self.optimizer = torch.optim.Adam(params=self.nn.parameters(), lr=self.learning_rate)

    def _init_nn(self):
        """Initialize the PyTorch BERT nn and put onto device"""
        self.nn = BertForTokenClassification.from_pretrained(self.base_model_url, num_labels=self.num_labels)
        self.nn.to(self.device)
        return self.nn

    def preprocess(self, data_snapshot, generate_labels=True):
        """Transform data snapshot into a dataframe
        Returns a dataframe of the text without labels in the case of inference when generate_labels is set to False

        Parameters
        ----------
        data_snapshot : dict of Documents
            data snapshot for training/validation/test data
        generate_labels : bool, optional
            Whether to generate training labels (for training and validation), by default True

        Returns
        -------
        pd.DataFrame
            Data transformed into dataframe with keys for ['text', 'id'] and ['label'] if labels are generated
        """
        if generate_labels:
            df = ner_preprocess(data_snapshot, self.tokenizer, self.entity_labels, self.max_seq_len, self.stride)
        else:
            df = text_split_preprocess(data_snapshot, self.tokenizer, self.max_seq_len, self.stride)
        return df

    def train(self, training_data, validation_data):
        """Train the model based on the input parameters
        Trains self.nn for specified number of epochs, generates training and validation metrics

        Parameters
        ----------
        training_data : pd.DataFrame
            preprocessed dataframe for training data
        validation_data : pd.DataFrame
            preprocessed dataframe for validation data

        Returns
        -------
        dict
            self.metrics dictionary
        """
        # Transform dataframe into pytorch datasets for training and validation
        tr_df = self.preprocess(training_data, generate_labels=True)
        val_df = self.preprocess(validation_data, generate_labels=True)
        # Transform pytorch datasets into dataloaders
        tr_dataloader = df_to_dataloader(tr_df, self.tokenizer, self.tokenizer_params, self.batch_size)
        val_dataloader = df_to_dataloader(val_df, self.tokenizer, self.tokenizer_params, self.batch_size)

        for epoch in range(self.epochs):
            train_metrics = self._train(tr_dataloader, epoch)
            self.metrics.setdefault('training', []).append(train_metrics)
            # Evaluate validation data
            valid_metrics = self._validate(val_dataloader, epoch)
            self.metrics.setdefault('validation', []).append(valid_metrics)

        # Return the metrics dict
        return self.metrics

    def _train(self, train_dataloader, epoch):
        """Run one epoch for training and generate metrics
        Updates weights of self.nn through backpropagation
        Return metrics with training loss and accuracy

        Parameters
        ----------
        train_dataloader : TrainingDataloader
            torch dataloader with training data
        epoch : int
            number for current epoch

        Returns
        -------
        dict
            metrics dict for the training epoch
        """
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []
        # Put model in training mode
        self.nn.train()
        for idx, batch in enumerate(train_dataloader):
            ids = batch['input_ids'].to(self.device, dtype=torch.long)
            mask = batch['attention_mask'].to(self.device, dtype=torch.long)
            labels = batch['label'].to(self.device, dtype=torch.long)

            outputs = self.nn(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            tr_logits = outputs.logits
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += labels.size(0)

            if idx % 100 == 0:
                loss_step = tr_loss / nb_tr_steps
                print(f"Epoch {epoch}: Training loss per 100 training steps: {loss_step}")

            # compute training accuracy
            flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, self.nn.num_labels)  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            # tr_labels.extend(labels)
            # tr_preds.extend(predictions)

            tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            tr_accuracy += tmp_tr_accuracy

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=self.nn.parameters(), max_norm=self.max_grad_norm
            )

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Make training metrics
        train_metrics = {
            'epoch': epoch,
            'loss': tr_loss / nb_tr_steps,
            'accuracy': tr_accuracy / nb_tr_steps
        }
        return train_metrics

    def _validate(self, val_dataloader, epoch):
        """Run one epoch for validation and generate metrics
        Return metrics with loss, accuracy, confusion matrix, seq_eval metrics for IOB classes

        Parameters
        ----------
        train_dataloader : TrainingDataloader
            torch dataloader with training data
        epoch : int
            number for current epoch

        Returns
        -------
        dict
            metrics dict for the training epoch
        """
        # put model in evaluation mode
        self.nn.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_examples, nb_eval_steps = 0, 0
        eval_preds, eval_labels = [], []

        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):
                ids = batch['input_ids'].to(self.device, dtype=torch.long)
                mask = batch['attention_mask'].to(self.device, dtype=torch.long)
                labels = batch['label'].to(self.device, dtype=torch.long)

                outputs = self.nn(input_ids=ids, attention_mask=mask, labels=labels)
                loss = outputs.loss
                eval_logits = outputs.logits
                eval_loss += loss.item()

                nb_eval_steps += 1
                nb_eval_examples += labels.size(0)

                if idx % 100 == 0:
                    loss_step = eval_loss / nb_eval_steps
                    print(f"Epoch {epoch}: Validation loss per 100 evaluation steps: {loss_step}")

                # compute evaluation accuracy
                flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
                active_logits = eval_logits.view(-1, self.num_labels)  # shape (batch_size * seq_len, num_labels)
                flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

                # only compute accuracy at active labels
                active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)

                labels = torch.masked_select(flattened_targets, active_accuracy)
                predictions = torch.masked_select(flattened_predictions, active_accuracy)

                eval_labels.extend(labels)
                eval_preds.extend(predictions)

                tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
                eval_accuracy += tmp_eval_accuracy

        labels = [id.item() for id in eval_labels]
        predictions = [id.item() for id in eval_preds]

        cm = confusion_matrix(labels, predictions)
        # Get accuracy report of just tags
        entity_encoding = self.entity_encoding
        labels_to_tags = {v: k for k, v in entity_encoding.items()}
        labels = [labels_to_tags[x] for x in labels]
        predictions = [labels_to_tags[x] for x in predictions]
        report = classification_report([labels], [predictions], digits=4)
        # Total loss and accuracy
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps
        metrics = {
            'epoch': epoch,
            'confusion_matrix': cm,
            'iob_tag_report': report,
            'loss': eval_loss,
            'accuracy': eval_accuracy
        }
        return metrics

    def evaluate(self, test_data_snapshot):
        """Evaluate model performance on held-out test data and record test evaluation metrics"""
        # Transform data snapshot into training data
        # For each document in the snapshot
        #    Pre-process text into correct format (splitting if needed)
        #    Labels prepared to compare to tags extracted
        # Run inference for the entire text
        # Transform model outputs into tags for that text
        # Compute metric for that document in the snapshot
        pass

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

Therefore, the `preprocess()`, `train()`, and `evaluate()` methods should take those inputs and produce those outputs as specified above.


## Models need stuff
`LEARNING_RATE, MAX_SEQ_LEN, HF_URL, RANDOM_SEED, BATCH_SIZE, MAX_GRAD_NORM, DATA_LEN, EPOCHS, LABELS_TO_CLASSES, CLASSES_TO_LABELS`