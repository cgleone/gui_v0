from .minimal_model import ModelApi


class TrainingModel(ModelApi):
    """
    Generic class for model training
    """

    def __init__(self):
        super().__init__()
        self.metrics = {}

    def preprocess(self, data_snapshot):
        """Transform data snapshot into training data"""
        # Text:
        #    Determine how to chunk the text by the MAX_SEQ_LEN, doing best to keep within words
        #    Re-label each chunk with 'text_id' + ':' + str(i), where i is the enumeration of that chunk
        # Labels
        #    Generate training labels for chunk from that document's text
        #    Labels need to be saved with the same key as the text chunks
        # Save all these into a dataframe
        pass

    def train(self, training_data, validation_data=None):
        """Train model on training data and records training metrics"""
        # Transform dataframe into pytorch datasets for training and validation
        # Transform pytorch datasets into dataloaders
        # Initiate model based on parameters set in __init__
        # Iterate over training data, forward pass, compute loss, backprop, evaluate val data
        # Compute metrics every epoch(?) and store them
        # Compute final training metrics for loss, accuracy, confusion matrix in a dict
        # Return the metrics dict
        pass

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
