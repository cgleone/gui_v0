from .training_model_api import TrainingModel
from .preprocessing import qa_preprocess, qa_preprocess_docs
from .utils import load_pickle_from_aws
from transformers import AutoTokenizer
from haystack.nodes import FARMReader
from thefuzz import process
import dateparser
import warnings
from .utils import CLINIC_NAME_LIST, TRUE_MODALITY_LABELS, TRUE_BODY_PART_LABELS
import pandas as pd
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
        self.reader = FARMReader(self.qaModel, use_gpu=self.use_cuda, num_processes=1, return_no_answer=True, top_k=3)
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
    
    def _predict_batch(self, docs_batch):
        """
        Predict a batch of answers (up to around 200)
        Uses reader to ask all questions

        Parameters
        ----------
        docs_batch : list of Haystack Documents
            documents to make predictions on

        Returns
        -------
        results : list of dictionaries
            predictions for each document
        """
        results = []
        for doc in docs_batch:
            single_doc = []
            single_doc.append(doc)
            
            dr_result = self.reader.predict("What is the doctor's name?", single_doc, top_k=1)
            date_result = self.reader.predict("What date was it taken on?", single_doc, top_k=1)
            clinic_result = self.reader.predict("What is the name of the clinic?", single_doc, top_k=1)
            bodypart_result = self.reader.predict("What is the body part?", single_doc, top_k=1)
            modality_result = self.reader.predict("What is the imaging modality?", single_doc, top_k=1)
            
            outputs = {
                'id':doc.meta['name'],
                'report_id': doc.meta['name'][0:-2],
                'Doctor Name': {'answer': dr_result['answers'][0].answer, 'score': dr_result['answers'][0].score},
                'Date Taken': {'answer': date_result['answers'][0].answer, 'score': date_result['answers'][0].score},
                'Modality': {'answer': modality_result['answers'][0].answer, 'score': modality_result['answers'][0].score},
                'Body Part': {'answer': bodypart_result['answers'][0].answer, 'score': bodypart_result['answers'][0].score},
                'Clinic Name': {'answer': clinic_result['answers'][0].answer, 'score': clinic_result['answers'][0].score}
            }
            results.append(outputs)

        return results

    def _labels_from_predictions(self, predictions):
        """
        Determine the aproppriate labels based on the predictions

        Parameters
        ----------
        predictions : dictionary of predicted answers

        Returns
        -------
        labels : dict of dict
            Dictionary of result dictionaries for each input text associated with original IDs
        """

        labels = predictions
    
        for key, value in labels.items():
            for k in self.RESULT_KEYS:
                if value[k]['label'] == '':
                    value[k]['label'] = None
            
            if value['Date Taken']['label']:
                    date = value['Date Taken']['label']
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        dt = dateparser.parse(date)
                        value['Date Taken']['label'] = dt.strftime('%Y-%m-%d') if dt else None

            # if value['Clinic Name']['label']:
                # match, _ = process.extractOne(value['Clinic Name']['label'], CLINIC_NAME_LIST)
                # value['Clinic Name']['label'] = match

            for k, true_labels in zip(['Modality', 'Body Part'], [TRUE_MODALITY_LABELS, TRUE_BODY_PART_LABELS]):
                    if value[k]['label']:
                        match, _ = process.extractOne(value[k]['label'], true_labels.keys())
                        value[k]['label'] = true_labels[match]
        
        return labels


    def predict(self, snapshot, batch_size = 200):
        """
        Do inference predictions on list of input texts.

        Parameters:
        -----------
        snapshot: Documents 
            Input text to be predicted on

        Returns:
        --------
        labels: dict of dict
            Dictionary of result dictionaries for each input text associated with original IDs
        """

        
        input_docs = self.preprocess(snapshot, False)
        if len(input_docs) <= batch_size:
            predictions = self._predict_batch(input_docs)
        else: 
            # Number of batches necessary
            x = int(len(input_docs))/ batch_size + (len(input_docs) % batch_size >0)
            predictions = []
            for i in range(int(x)):
                lower_bound = batch_size*i
                upper_bound = batch_size * (i+1)
                input_docs_subset = input_docs[lower_bound:upper_bound]
                predictions_subset = self._predict_batch(input_docs_subset)
                predictions.extend(predictions_subset)

        results_df = pd.DataFrame(predictions)

        preds = {}
        label_categories = ['Doctor Name', 'Date Taken',  'Modality', 'Body Part', 'Clinic Name']
        
        for key, doc in snapshot.items():

            relevant_data = results_df.loc[results_df['report_id'] == key]
            pred ={}
            for category in label_categories:
                if relevant_data.shape[0] == 1:
                    for idx, row in relevant_data.iterrows():
                        pred[category] = {'label': row[category]['answer'], 'score': row[category]['score']}
                elif relevant_data.shape[0] > 1:
                    answers =[]
                    scores = []
                    for idx, row in relevant_data.iterrows():
                        answers.append(row[category]['answer'])
                        scores.append(row[category]['score'])
                    
                    index = scores.index(max(scores))
                    pred[category] = {'label': answers[index], 'score': scores[index]}
        
            preds[key] = pred

        labels = self._labels_from_predictions(preds)
        
        return labels