from .training_model_api import TrainingModel
from .preprocessing import cls_preprocess, text_split_preprocess, df_to_dataloader, glob_to_snapshot
from .preprocessing import entity_labels
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from .utils import generate_default_parameters


class ClsModel(TrainingModel):
    """
    Model for training on Clasification task
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
        self.parameters['num_labels'] = 5

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
        self.nn = BertForSequenceClassification.from_pretrained(self.base_model_url, num_labels=self.num_labels)
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
            df = cls_preprocess(data_snapshot, self.tokenizer, 'Modality')
        else:
            df = text_split_preprocess(data_snapshot, self.tokenizer, self.max_seq_len, self.stride)
        return df

    def train(self, tr_df, val_df):
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
            print("hi")
            ids = batch['input_ids'].to(self.device, dtype=torch.long)
            mask = batch['attention_mask'].to(self.device, dtype=torch.long)
            labels = batch['label'].to(self.device, dtype=torch.long)

            outputs = self.nn(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            tr_logits = outputs.logits
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += labels.size(0)

            if idx % 1 == 0:
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
        # entity_encoding = self.entity_encoding
        # labels_to_tags = {v: k for k, v in entity_encoding.items()}
        # labels = [labels_to_tags[x] for x in labels]
        # predictions = [labels_to_tags[x] for x in predictions]
        # report = classification_report([labels], [predictions], digits=4)
        # Total loss and accuracy
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps
        metrics = {
            'epoch': epoch,
            'confusion_matrix': cm,
            # 'iob_tag_report': report,
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


def test_model():
    model = ClsModel()
    model.set_parameters(generate_default_parameters())
    data_snapshot = glob_to_snapshot("/Users/thomasfortin/Desktop/School/4A/BME 461/DataSynthesis/Synthesized_Data", extra_level=True)
    df = model.preprocess(data_snapshot)
    df = df.sample(frac=.01)
    training = df.sample(frac=.8)
    test = df.drop(training.index)

    result = model.train(training.reset_index(), test.reset_index())
    return result, model


if __name__ == "__main__":
    result, model = test_model()
    print(result)
