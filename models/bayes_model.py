from .training_model_api import TrainingModel
from .preprocessing import cls_preprocess, text_split_preprocess
from .preprocessing import entity_labels
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from .utils import load_pickle_from_aws
import os

class NbModel(TrainingModel):
    """
    Model for training on Question-answering task
    """

    def __init__(self):
        super().__init__()
        self.entity_labels = entity_labels
        self.naive_bayes = MultinomialNB()

    def set_parameters(self, parameters):
        """Set parameters dictionary and setup model, tokenizer, and optimizer

        Parameters
        ----------
        parameters : dict
            parameters dictionary, model.__getattr__ will try to look here if the name doesn't exist in self.__dict__
        """
        super().set_parameters(parameters)

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_url)

         # Load a pre-trained network saved on AWS
        if self.parameters.get('trained_model_url', None):
            print('Loading model from AWS...')
            language_model = load_pickle_from_aws(self.parameters['trained_model_url'])
            self.naive_bayes = language_model



    def preprocess(self, data_snapshot, generate_labels=True):
        """Transform data snapshot into a SQUAD format JSON file
        Returns 

        Parameters
        ----------
        data_snapshot : dict of Documents
            data snapshot for training/validation/test data
        generate_labels : bool, optional
            Whether to generate training labels (for training and validation), by default True

        Returns
        -------
        List

        """
        
        if generate_labels:
            df = cls_preprocess(data_snapshot, self.tokenizer, 'Modality')
        else:
            df = text_split_preprocess(data_snapshot, self.tokenizer, self.max_seq_len, self.stride)
        return df

    def train(self, df):
        """Train the model based on the input parameters
        Trains self.nn for specified number of epochs, generates training and validation metrics

        Parameters
        ----------
    
        df :   pd.DataFrame
            dataframe of data with text and labels (numerically encoded)

        Returns
        -------
        dict
            self.metrics dictionary

        """

        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], random_state=1)

        cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
        X_train_cv = cv.fit_transform(X_train)
        X_test_cv = cv.transform(X_test)

        self.naive_bayes.fit(X_train_cv, y_train)
        predictions = self.naive_bayes.predict(X_test_cv)
        self.metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted')
        }

        self._validate(X_test_cv, y_test )
    
        return self.metrics

    def _validate(self, X_test_cv, y_test):
        
        # predictions = self.naive_bayes.predict(X_test_cv)
        # print('Accuracy score: ', accuracy_score(y_test, predictions))
        # print('Precision score: ', precision_score(y_test, predictions, average='weighted'))
        # print('Recall score: ', recall_score(y_test, predictions, average='weighted'))
        
        return

    def evaluate(self, test_data_snapshot):
        """Evaluate model performance on held-out test data and record test evaluation metrics"""
        
        pass
