from .training_model_api import TrainingModel
from .preprocessing import qa_preprocess, qa_preprocess_docs
from .utils import load_pickle_from_aws
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
        self.reader = FARMReader(self.qaModel, use_gpu=True, num_processes=1, return_no_answer=True, top_k=3)
        # Load a pre-trained network saved on AWS
        if self.parameters.get('trained_model_url', None):
            print('Loading nn from AWS (this could take a while)...')
            language_model = load_pickle_from_aws(self.parameters['trained_model_url'])
            self.reader.inferencer.model = language_model

    def preprocess(self, data_snapshot, generate_labels=True, label_type='all'):
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
            if label_type != 'all':
                output = qa_preprocess(data_snapshot, self.tokenizer, self.max_seq_len, self.json_save_path_eval, label_type)
            else:
                output = qa_preprocess(data_snapshot, self.tokenizer, self.max_seq_len, self.json_save_path)
        else:
            output = qa_preprocess_docs(data_snapshot, self.tokenizer, self.max_seq_len)

        return output

    def train(self, data_snapshot, dev_split):
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
        """,
        head_tail = os.path.split(self.json_save_path)
        data_dir = head_tail[0]
        train_filename = head_tail[1]
        self.reader.train(data_dir=data_dir, train_filename=train_filename, use_gpu=True, 
                n_epochs=self.epochs, max_seq_len=self.max_seq_len, dev_split=dev_split, learning_rate=self.learning_rate, 
                batch_size=self.batch_size, evaluate_every=100)
        
        self.metrics = self.evaluate(data_snapshot)
        
        return self.metrics

    def _validate(self, data_snapshot):
        
        pass

    def evaluate(self, data_snapshot, dev = False):
        """Evaluate model performance on held-out test data and record test evaluation metrics"""
        label_types = ['Doctor Name', 'Date Taken', 'Clinic Name', 'Body Part', 'Modality']
        head_tail = os.path.split(self.json_save_path_eval)
        data_dir = head_tail[0]
        val_filename = head_tail[1]

        metrics = {}
        
        for label_type in label_types:
            self.preprocess(data_snapshot, True, label_type)
            eval_metrics = self.reader.eval_on_file(data_dir, val_filename, 'cuda')
            metrics[label_type] = eval_metrics
        
        if dev:
            self.metrics.setdefault('validation', []).append(metrics)
        else: 
            self.metrics.setdefault('held out test', []).append(metrics)
       
        return metrics

    def predict(self, input_docs):
        """
        Do inference predictions on list of input texts.

        Parameters:
        -----------
        input_docs: List of haystack Docs
            Input texts and associated IDs as keys

        Returns:
        --------
        results: dict of dict
            Dictionary of result dictionaries for each input text associated with original IDs
        """

        self.update_inputs(input_docs)
        self.update_results()
        return self.get_results()
