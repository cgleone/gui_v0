from .training_model_api import TrainingModel
from .preprocessing import get_iob_entity_encoding, qa_preprocess, qa_preprocess_docs
from .preprocessing import entity_labels
from transformers import AutoTokenizer
from haystack.nodes import FARMReader
import os

class QaModel(TrainingModel):
    """
    Model for training on Question-answering task
    """

    def __init__(self):
        super().__init__()
        

    def set_parameters(self, parameters):
        """Set parameters dictionary and setup model, tokenizer, and optimizer

        Parameters
        ----------
        parameters : dict
            parameters dictionary, model.__getattr__ will try to look here if the name doesn't exist in self.__dict__
        """
        super().set_parameters(parameters)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_url)
        self.reader = FARMReader(self.qaModel, use_gpu = True)

    def preprocess(self, data_snapshot, generate_labels=True):
        """Transform data snapshot into a SQUAD format JSON file

        Parameters
        ----------
        data_snapshot : dict of Documents
            data snapshot for training/validation/test data
        generate_labels : bool, optional
            Whether to generate training labels (for training and validation), by default True

        Returns
        -------
        List
            If generate_labels this is in SQuAD format, and if not, it is in the format of Haystack Document objects
        """
        if generate_labels:
            output = qa_preprocess(data_snapshot, self.tokenizer, self.max_seq_len, self.json_save_path)
        else:
            output = qa_preprocess_docs(data_snapshot, self.tokenizer, self.max_seq_len)

        return output

    def train(self, dev_split):
        """Train the model based on the input parameters
        Trains self.nn for specified number of epochs, generates training and validation metrics

        Parameters
        ----------
        json_save_path : string path
            path to file containing SQuAD formatted training data
        dev_split : float between 0 and 1
            percentage of SQuAD examples that get split off for evaluation

        Returns
        -------
        dict
            self.metrics dictionary
        """
        head_tail = os.path.split(self.json_save_path)
        data_dir = head_tail[0]
        train_filename = head_tail[1]
<<<<<<< HEAD
        self.reader.train(data_dir=data_dir, train_filename=train_filename, use_gpu=True, n_epochs=self.epochs, max_seq_len=self.max_seq_len, dev_split=dev_split, learning_rate=self.learning_rate, batch_size=self.batch_size)

=======
        self.reader.train(data_dir=data_dir, train_filename=train_filename, use_gpu=True, n_epochs=self.n_epochs, max_seq_len=self.max_seq_len, dev_split=dev_split, learning_rate=self.learning_rate, batch_size=self.batch_size)
>>>>>>> 4b75e63acdb08345cffaf003d4862fa5aec3f5ab
        return

    def _validate(self, val_dataloader, epoch):
        pass

    def evaluate(self, test_data_snapshot):
        """Evaluate model performance on held-out test data and record test evaluation metrics"""
        pass
