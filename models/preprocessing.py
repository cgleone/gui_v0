from collections import UserDict
import glob
import re
import pickle


class Document(UserDict):
    """Datastructure for holding text and label information
    Add convenince functions for getting labels we care about and looking up mostly in the labels dict
    """
    def __init__(self, data):
        super().__init__(data)
        self.text = data.get('text', '')
        self.labels = data.get('labels', {})
        self.LABELS_TO_CLASSIFY = ['Doctor Name', 'Date Taken', 'Clinic Name', 'Body Part', 'Modality']

    def __getitem__(self, key):
        """Return item for given keys. If the key is 'text' or 'labels', then return those.
        Otherwise, return items directly from labels

        Parameters
        ----------
        key : str
            key to lookup

        Returns
        -------
        Any
            value of item from key
        """
        if key == 'text':
            return self.data.get('text', '')
        elif key == 'labels':
            return self.labels
        else:
            # Lookup this key in labels
            return self.labels.__getitem__(key)

    def get_labels_to_classify(self):
        """Lookup labels to classify in labels and return them.
        Return empty dict if label not found

        Returns
        -------
        dict
            dict of labels to classify and their values
        """
        return {k: self.labels.get(k, {}) for k in self.LABELS_TO_CLASSIFY}

def glob_to_snapshot(top_level_path):
    """Globs files in the top_level_path and reads into a data snapshot structure
    Will only keep files that have both a valid text and labels
    Example top_level_path: "/home/thomasfortin/data_v2"

    Parameters
    ----------
    top_level_path : str
        File path underwhich there must be '/Texts' and '/Labels' folders

    Returns
    -------
    dict of Document
        data snapshot of all valid files
    """
    # Texts
    prefix = top_level_path + "/Texts/"
    files = glob.glob(prefix + "*/*.txt")
    texts = {}
    # Load texts as lines with '\n' between them
    for file in files:
        # Texts
        with open(file) as f:
            lines = [line.rstrip('\n') for line in f.readlines()]
            # Remove lines that are only whitespace
            lines = [line for line in lines if line.strip()]
            text = '\n'.join(lines)
            _id = re.sub(prefix, '', file).rstrip('.txt')
            texts.setdefault(_id, text)

    # Labels
    prefix = top_level_path + "/Labels/"
    files = glob.glob(prefix + "*/*")
    labels = {}
    for file in files:
        # Labels
        with open(file, 'rb') as f:
            d = pickle.load(f)
            _id = re.sub(prefix, '', file)
            labels.setdefault(_id, d)

    # Merge common labels and texts
    keys = set(list(texts.keys()) + list(labels.keys()))
    snapshot = {}

    for key in keys:
        text = texts.get(key, None)
        label = labels.get(key, None)
        if text is None or label is None:
            continue
        text = clean_text(text)
        snapshot.setdefault(key, Document({'text': text, 'labels': label}))
    return snapshot

def clean_text(text):
    """Removes unwanted character sequences from text
    Replaces [**redacted**] with [MASK]

    Parameters
    ----------
    text : str
        text to clean
    """
    mask = re.compile('\[\*\*.*?\*\*\]', flags=re.S)  # re.S flag means dot matches all chars including \n
    return mask.sub('[MASK]', text)
