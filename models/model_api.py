import copy

class ModelApi(object):
    """
    Tag extraction model API for Capstone
    """
    RESULT_KEYS = ['modality', 'body_part', 'dr_name', 'clinic_name', 'date_of_procedure']

    def __init__(self):
        self.state = {}
        self.parameters = {}
        self.results = {}
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
        input_data: dict of strings
            Keys are the ID of the text
            Values are texts to train/evaluate
        labels: dict of dict
            Keys are same as input_data
            Values have same structure as results
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
        Second level of keys will always contain ['label'] but might also have additional keys

        Returns:
        --------
        results: list of dict of dict
        """
        results = self.results
        self.results = {}
        return results

    def predict(self, input_data):
        """
        Do inference predictions on list of input texts.

        Parameters:
        -----------
        input_data: dict of str
            Input texts and associated IDs as keys

        Returns:
        --------
        results: dict of dict
            Dictionary of result dictionaries for each input text associated with original IDs
        """
        self.update_inputs(input_data)
        self.update_results()
        return self.get_results()

    def train(self):
        """Model does training based on input data. Stores trained parameters in self.parameters"""
        raise NotImplementedError()

    def __getattr__(self, __name):
        """Make class get attributes from self.parameters first

        Parameters
        ----------
        __name : str

        Returns
        -------
        Any
        """
        # No parameters found, return empty dict for now
        if __name == 'parameters':
            return {}
        if __name in self.parameters.keys():
            return self.parameters.get(__name, None)    

    def __setattr__(self, __name, __value):
        """Make class setattr try to update self.parameters dictionary first

        Parameters
        ----------
        __name : str
        __value : Any

        Returns
        -------
        None
        """
        if __name in self.parameters.keys():
            self.parameters.update({__name: __value})
        else:
            super().__setattr__(__name, __value)
        return None
