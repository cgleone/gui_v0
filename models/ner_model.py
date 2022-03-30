from .training_model_api import TrainingModel
from .preprocessing import get_iob_entity_encoding, ner_preprocess, text_split_preprocess, df_to_dataloader, Document
from .preprocessing import entity_labels
from .utils import CLINIC_NAME_LIST, TRUE_MODALITY_LABELS, TRUE_BODY_PART_LABELS
from .utils import load_nn_from_aws, score_tags

from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import accuracy_score, confusion_matrix
from seqeval.metrics import classification_report
from scipy.stats.mstats import gmean
from thefuzz import process
import dateparser
import warnings
import torch

class NerModel(TrainingModel):
    """
    Model for training on Named Entity Recognition task
    """

    def __init__(self):
        super().__init__()
        self.entity_labels = entity_labels
        self.entities_to_results = {v: k for k, v in self.entity_labels.items()}

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
        """Initialize the nn and put onto device
        If trained_model_url is from s3, it will load from s3. However, it can also load local files
        If we don't get a pre-trained model, load from a base model and make a token classifier
        """
        if self.parameters.get('trained_model_url', None):
            path = self.trained_model_url
            if path.startswith('s3://'):
                print('Loading nn from AWS (this could take a while)...')
            self.nn = load_nn_from_aws(path)
        else:
            self.nn = AutoModelForTokenClassification.from_pretrained(self.base_model_url, num_labels=self.num_labels)
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
        # Transform pytorch datasets into dataloaders
        tr_dataloader = df_to_dataloader(training_data, self.tokenizer, self.tokenizer_params, self.batch_size)
        val_dataloader = df_to_dataloader(validation_data, self.tokenizer, self.tokenizer_params, self.batch_size)

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

    def _extract_entities(self, text):
        """
        Extract entities from a single piece of text
        From logits of forward pass, softmax to comptute label probabilities, use the argmax to get the predicted label
        for that token Loop through the tokens and construct the entity from consecutively labelled tokens. Because we
        only predict on the first wordpiece, we add partial wordpieces to the entity.
        The probabilities for the entity are computed by the geometric mean of the softmax probabilities for each token
        in that entity (exluding partial wordpiece tokens).

        Parameters
        ----------
        text : str
            Text to extract entities from

        Returns
        -------
        dict
            entities, each with keys for the entity type and values as lists tuples (entity, probability)
        """
        ids_to_labels = {v: k for k, v in self.entity_encoding.items()}

        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len,
            return_offsets_mapping=True,
            return_tensors="pt")

        # Forward pass
        self.nn.eval()
        with torch.no_grad():
            ids = inputs["input_ids"].to(self.device)
            mask = inputs["attention_mask"].to(self.device)
            outputs = self.nn(ids, attention_mask=mask)
            logits = outputs[0]

            # Compute labels and probs
            active_logits = logits.view(-1, self.num_labels)  # shape (batch_size * seq_len, num_labels)
            probs = torch.softmax(active_logits, axis=1)
            pred_probs, pred_inds = torch.max(probs, axis=1)
            pred_probs = pred_probs.cpu().numpy()
            pred_inds = pred_inds.cpu().numpy()

            pred_labels = [ids_to_labels[i] for i in pred_inds]
            tokens = self.tokenizer.convert_ids_to_tokens(ids.flatten().tolist())
            offsets = inputs['offset_mapping'].squeeze().tolist()

        # Extract entities from indices
        entities = {}
        ent_type = 'O'
        temp_probs = []
        start = None

        for token, pred, prob, offset in zip(tokens, pred_labels, pred_probs, offsets):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            # Add partial word pieces
            if start is not None and token.startswith('##'):
                end = offset[1]
            elif pred.startswith('B-') or pred.startswith('I-'):
                # Start a new entity
                if ent_type != pred[2:]:
                    if start is not None:
                        entities.setdefault(ent_type, []).append((text[start:end], gmean(temp_probs)))
                        temp_probs = []
                    start, end = offset
                    ent_type = pred[2:]
                    temp_probs.append(prob)
                # Continue entity
                else:
                    temp_probs.append(prob)
                    end = offset[1]
            else:  # 'O' entity or partial wordpiece
                if start is not None:
                    entities.setdefault(ent_type, []).append((text[start:end], gmean(temp_probs)))
                    start = None
                    ent_type = 'O'
                    temp_probs = []
        if start is not None:
            entities.setdefault(ent_type, []).append((text[start:end], gmean(temp_probs)))

        return entities

    def _label_snapshot(self, snapshot):
        """Extract text from most probable entities for each document in the snapshot
        Returned labels have the structure of
        labels = {
            <label name>: {
                'label': <label to report to UI and evaluation>,
                'true text': <NER extracted text>,
                'probability': <geometric mean of entity probabilities>
            },
            <label name>: {...},
            ...
        }

        Parameters
        ----------
        snapshot : dict of Document
            Data to predict on. Will only look at "text" value for this Document

        Returns
        -------
        dict of {report_id: labels}
            Labelled reports with same report_id as the snapshot
        """
        df = self.preprocess(snapshot, generate_labels=False)
        df['report_id'] = df['id'].apply(lambda x: x.split(':')[0])

        # Do predictions
        all_ents = []
        for i, row in df.iterrows():
            entities = self._extract_entities(row.text)
            all_ents.append(entities)
        df['entities'] = all_ents

        # Consolidate labels per report
        labelled_reports = {}
        for rep_id in snapshot.keys():
            inds = (df['report_id'] == rep_id)
            ents = df.loc[inds, 'entities'].values
            entities = {}
            for e in ents:
                for k, v in e.items():
                    entities.setdefault(k, []).extend(v)
            labels = self._labels_from_entities(entities)
            labelled_reports[rep_id] = labels
        return labelled_reports

    def _labels_from_entities(self, entities):
        """Generate labels dict from extracted entities
        'label' matches Clinic Name, Modality, and Body Part, fuzzy match to closest actual answer
        and converts date to 'YYYY-MM-DD' format.
        'true text' is the exact text that NER extracted
        'probability' is the probability associated with that entitiy

        Parameters
        ----------
        entities : dict
            dict with keys as entity types and values as a list of (entity, probability)

        Returns
        -------
        dict
            labels in output format with keys self.RESULT_KEYS and values as dicts with keys
            'label', 'true text', 'probabability'
        """
        labels = {}
        # Choose the entity with max probability
        for k, v in entities.items():
            label_prob = max(v, key=lambda x: x[1])
            labels[self.entities_to_results[k]] = {
                'true text': label_prob[0].replace('##', ''),  # Trim excess ## from partial wordpieces
                'probability': label_prob[1]
            }
        # Fill in label keys, use None if no entities were extracted
        for k in self.RESULT_KEYS:
            labels.setdefault(k, {}).setdefault('label', labels[k].get('true text', None))

        # Parse date into right format
        if labels['Date Taken']['label']:
            date = labels['Date Taken']['label']
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dt = dateparser.parse(date)
                labels['Date Taken']['label'] = dt.strftime('%Y-%m-%d') if dt else None

        # Find closest clinic name
        if labels['Clinic Name']['label']:
            match, _ = process.extractOne(labels['Clinic Name']['label'], CLINIC_NAME_LIST)
            labels['Clinic Name']['label'] = match

        # Find closest string match and convert to category
        for k, true_labels in zip(['Modality', 'Body Part'], [TRUE_MODALITY_LABELS, TRUE_BODY_PART_LABELS]):
            if labels[k]['label']:
                match, _ = process.extractOne(labels[k]['label'], true_labels.keys())
                labels[k]['label'] = true_labels[match]
        return labels

    def predict(self, input_data):
        """Predict labels on input data dict

        Parameters
        ----------
        input_data : dict
            Dict with keys as report IDs and values as the text from those reports

        Returns
        -------
        dict
            Dict of results. Keys are same report IDs, values are the labels the NER model predicts
            labels in format {'Doctor Name': 'Dr. K. Samson', 'Modality': 'X-RAY', ...}
        """
        snapshot = {k: Document({'text': v}) for k, v in input_data.items()}
        labelled_reports = self._label_snapshot(snapshot)
        # Take only the 'label' key to return to UI
        output = {}
        for rep_id, labels in labelled_reports.items():
            temp = {k: v['label'] for k, v in labels.items()}
            output[rep_id] = temp
        return output

    def evaluate(self, snapshot, return_results=False):
        """Evalute performance of NER model on a snapshot

        Parameters
        ----------
        snapshot : dict of Document
            snapshot to evaluate performance, held out test data or validation data
        return_results : bool, optional
            whether to return results by document, by default False

        Returns
        -------
        dict
            if return_results is True, dict with keys the same as snapshot document IDs, values as
            results per document.
            if return_results is False, dict with keys for each of the tags, values as the aggregate
            score for all documents in the snapshot
        """
        labelled_reports = self._label_snapshot(snapshot)
        # Grab only labels
        all_pred = {}
        for rep_id, labels in labelled_reports.items():
            temp = {k: v['label'] for k, v in labels.items()}
            all_pred[rep_id] = temp

        results = {}
        for k in snapshot.keys():
            pred = all_pred[k]
            doc = snapshot[k]
            results[k] = score_tags(doc, pred)
        if return_results:
            return results
        # Aggregate average per tag type
        agg = {}
        for doc_id, v in results.items():
            for k, score in v.items():
                # if k == 'Modality' and snapshot[doc_id]['Modality']['label'] == 'X-RAY':  # Skip X-RAY for now
                #     continue
                agg.setdefault(k, []).append(score)
        for k in agg:
            item = agg[k]
            agg[k] = sum(item) / len(item)
        return agg
