import copy

class ModelApi(object):
    """
    Tag extraction model API for Capstone
    """

    def __init__(self):
        self.state = {}
        self.parameters = {}
        self.results = []
        self.input_data = None

    def get_state(self):
        """Get copy of model state"""
        return copy.deepcopy(self.state)
    
    def set_state(self, state):
        """Set model state"""
        if state is not None:
            self.state = state
    
    def get_parameters(self):
        """Get copy of model parameters"""
        return copy.deepcopy(self.parameters)

    def set_parameters(self, parameters):
        """Set model parameters"""
        if parameters is not None:
            self.parameters = parameters
    
    def update_inputs(self, input_data, labels=None):
        """
        Parameters:
        -----------
        input_data: list of strings that are texts to train/evaluate
        labels: list of dict, has same structure as results
        """
        self.input_data = input_data
        self.state['labels'] = labels

    def update_results(self):
        """
        Model looks at self.input_data and does inference predictions
        """
        raise NotImplementedError()

    def get_results(self):
        """
        Return recent model results, Each result is a dict of dicts
        Highest level of keys will be ['modality', 'body_part', 'dr_name', 'clinic_name', 'date_of_procedure']
        Second level of keys will always contain ['label'] but might also have additional keys if model outputs that info

        Returns:
        --------
        results: list of dict of dict
        """
        results = self.results
        self.results = []
        return results
    
    def predict(self, input_data):
        """
        Do inference predictions on list of input texts.

        Parameters:
        -----------
        input_data: list of str
            Input texts
        
        Returns:
        --------
        results: list of dict of dict
            List of result dictionaries for each input text
        """
        self.update_inputs(input_data)
        self.update_results()
        return self.get_results()

    def train(self):
        """Model does training based on input data. Stores trained parameters in self.parameters"""
        raise NotImplementedError()
    
     
"""
results dict structure
----------------------
res = {
    'body_part': {
        'label': 'arm',
        'probability': 0.96,
        'text': 'hand'
    },
    'modality':{
        'label': 'MR',
        'probability': 0.6,
        'text': 'MRI'
    }
}

Using API:
----------
Training:
model = model_api()
model.update_inputs(training_data, labels)
model.train()
state = model.get_state()
params = model.get_parameters()

Inference:
model = model_api()  # Initialization
model.set_state(state)  # Load saved state/params
model.set_parameters(params)
results = model.predict(input_data) # Make predictions
"""
