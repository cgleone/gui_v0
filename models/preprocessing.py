from collections import UserDict
import glob
import re
import pickle
import json
import uuid
import warnings
import itertools
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

HEAD = 0
CHEST = 1
SPINE = 2
ABDOMEN = 3
PELVIS = 4
ABDOMEN_PELVIS = 5
CHEST_ABDOMEN = 6
CHEST_PELVIS = 7

body_part_labels = {
    "HEAD": HEAD,
    "CHEST": CHEST,
    "SPINE": SPINE,
    "ABDOMEN": ABDOMEN,
    "PELVIS": PELVIS,
    "ABDOMEN PELVIS": ABDOMEN_PELVIS,
    "CHEST ABDOMEN": CHEST_ABDOMEN,
    "CHEST PELVIS": CHEST_PELVIS
}

CT = 0
MRI = 1
US = 2
XRAY = 3
IDK = 4

modality_labels = {
    "CT": CT,
    "MRI": MRI,
    "US": US,
    "X-RAY": XRAY,
    "IDK": IDK
}


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

def glob_to_snapshot(top_level_path, extra_level=False):
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
    if extra_level:
        files = glob.glob(prefix + "Mimic3/*/*.txt")
    else:
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
    if extra_level:
        files = glob.glob(prefix + "Mimic3/*/*")
    else:
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


def qa_preprocess(snapshot, tokenizer, max_seq_len, json_save_path):
    """
    Creates SQuAD formatted training data from a given snapshot. First it splits the documents by token lenghth,
    and then finds the true text in the contents of the reports and saves the SQuAD format QA pairs into a JSON file.

    Returns: Data in SQuAD format (that was saved to the JSON file)

    Parameters
    ----------
    snapshot : dict of Documents
        Data to be used for training
    tokenizer: transformers.Tokenizer
        Tokenizer associated with intended model
    max_seq_len: int
        Maximum token sequence length for transformers model
    json_save_path: str
        Path that the JSON file will be saved to

    """
    questions = {
        'dr': "What is the doctor's name?",
        'date_taken': "What date was it taken on?",
        'clinic': "What is the name of the clinic?",
        'body_part': "What is the body part?",
        'modality': "What is the imaging modality?"
    }

    squad = []
    training_data = text_split_preprocess(snapshot, tokenizer, max_seq_len=max_seq_len)

    training_data['id_search'] = training_data['id']

    for index, row in training_data.iterrows():
        searchId = row['id']
        searchId = searchId[0:-2]
        training_data['id_search'][index] = searchId

    for key, doc in snapshot.items():
        relevant_labels = doc.get_labels_to_classify()

        relevant_training = training_data.loc[training_data['id_search'] == key]

        for idx, row in relevant_training.iterrows():

            temp = {"title": row['id'], "paragraphs": []}
            qas = []

            for q, (label_type, labels) in zip(questions, relevant_labels.items()):

                if labels is not None:

                    ans_text = labels['true text']
                else:
                    break

                offset = row['text'].find(ans_text)

                if offset != -1:
                    qa = {
                        'answers': [
                            {
                                'answer_start': offset,
                                'text': ans_text
                            }
                        ],
                        'question': questions[q],
                        'id': str(uuid.uuid4()),
                        'is_impossible': False
                    }

                else:
                    qa = {
                        'answers': [
                            {
                                'answer_start': offset,
                                'text': ans_text
                            }
                        ],
                        'question': questions[q],
                        'id': str(uuid.uuid4()),
                        'is_impossible': True
                    }

                qas.append(qa)

            temp['paragraphs'] = [{
                'context': row['text'],
                'qas': qas
            }]

        squad.append(temp)

    train_squad_data = {'data': squad}

    with open(json_save_path, 'w') as f:
        json.dump(train_squad_data, f)

    return train_squad_data

def qa_preprocess_docs(snapshot, tokenizer, max_seq_len):
    """
    Creates haystack documents from a given snapshot. First it splits the documents by token length,
    then it converts each to a haystack type doc,

    Returns: Haystack Documents

    Parameters
    ----------
    snapshot : dict of Documents
        Data to be used
    tokenizer: transformers.Tokenizer
        Tokenizer associated with intended model
    max_seq_len: int
        Maximum token sequence length for transformers model


    """
    from haystack import Document
    data = text_split_preprocess(snapshot, tokenizer, max_seq_len=max_seq_len)

    data['id_search'] = data['id']

    for index, row in data.iterrows():
        searchId = row['id']
        searchId = searchId[0:-2]
        data['id_search'][index] = searchId

    haystack_docs = []
    for key, doc in snapshot.items():
        relevant_labels = doc.get_labels_to_classify()

        relevant_data = data.loc[data['id_search'] == key]

        for idx, row in relevant_data.iterrows():
            haystack_doc = Document(content = row['text'], 
               meta = {
                   'name': row['id']
                #    'date_taken': relevant_labels['Date Taken']['label'],
                #    'dr_name': relevant_labels['Doctor Name']['label'],
                #    'clinic_name': relevant_labels['Clinic Name']['label'],
                #    'body_part': relevant_labels['Body Part']['label'],
                #    'modality': relevant_labels['Modality']['label']
                })
            haystack_docs.append(haystack_doc)

    return haystack_docs


def text_split_preprocess(snapshot, tokenizer, max_seq_len=512, stride=10):
    """Preprocessing transforms a snapshot datastructure with documents into a dataframe with ['text', 'label', 'id']
    These dicts can be later transformed into a pandas dataframe with all the training data.

    We split long texts into batches of tokens of max sequence length and use stride to prepend a number of tokens from
    the preceding sequence. Note that there will not be the exact number stride of tokens prepended as this method tries
    to ensure that we don't split on partial word pieces.

    Parameters
    ----------
    snapshot : dict of Document
        datastructure containing documents with texts and labels
    tokenizer : transformers.Tokenizer
        Tokenizer associated with intended model
    max_seq_len : int
        Maximum token sequence length for transformers model
    stride : int
        When splitting into phrases of max sequence length, how many tokens from previous sequence to prepend

    Returns
    -------
    training_data : pd.DataFrame
        DataFrame with columns ['text', 'id']
    """
    # Iterate through snapshot split texts
    output = []
    for _id, document in snapshot.items():
        text = document.text

        # Encode entire text with max_length=None
        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=None,
            return_overflowing_tokens=True)

        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        offsets = encoding['offset_mapping'][0]

        # Split text by max_seq_len
        chunk_size = max_seq_len - 2  # For [CLS] and [SEP]

        # First chunk
        start = 1  # Because 0th token is [CLS]
        end = min(start + chunk_size, len(tokens) - 1)
        # Move end forward to token that is not middle of word piece.
        while tokens[end].startswith('##'):
            end -= 1

        temp = []
        # Add chunks to temp
        while end < len(tokens) - 1:
            # Indices for sections of text
            start_idx = offsets[start][0]
            end_idx = offsets[end][0]
            # If you need to do labelling, do it here and add the label key when appending to temp
            temp.append({'text': text[start_idx:end_idx]})

            # Move start to beginning of word piece
            start = end - stride
            while tokens[start].startswith('##'):
                start -= 1
            # Don't let end go to a bad index
            end = min(end + chunk_size, len(tokens) - 1)
            # Move end forward
            while tokens[end].startswith('##') or (end - start > chunk_size):
                end -= 1
        # Last chunk
        start_idx = offsets[start][0]
        end_idx = offsets[-2][1]  # -1 token/offset is [SEP]
        # Do labelling here too
        temp.append({'text': text[start_idx:end_idx]})

        # Add temp items to output, remake id to include enumeration of the chunk
        for i, d in enumerate(temp):
            d['id'] = _id + ':' + str(i)
        output.extend(temp)

    training_data = pd.DataFrame(output)
    return training_data


entity_labels = {
    'Doctor Name': 'DRN',
    'Body Part': 'BOP',
    'Date Taken': 'DOP',
    'Clinic Name': 'IMC',
    'Modality': 'MOD'
}


def get_iob_entity_encoding(entity_labels):
    """Creates IOB entity encoding from entity labels dictionary
    'O' is always 0, 'X' is -100, which is PyTorch default ignore value for loss/accuracy calculations

    Parameters
    ----------
    entity_labels : dict
        keys same as label keys, values of IOB label
        E.g. {'Modality': 'MOD'}

    Returns
    -------
    entity_encoding : dict
        keys are the IOB tags, values are integer encodings
        E.g. {'O': 0, 'B-MOD': 1, 'I-MOD': 2}
    """
    entity_encoding = {'-'.join([io, label]): i
                       for i, (label, io) in
                       enumerate(itertools.product(sorted(entity_labels.values()), ['B', 'I']), start=1)}
    entity_encoding['O'] = 0
    entity_encoding['X'] = -100  # The default no prediction for pytorch softmax
    return entity_encoding


def label_encoded_tokens(encoding, entity_spans, tokenizer, entity_encoding):
    """Generate labels for the tokens given the spans where entities exist within the text by matching
    the positions of the tokens to the spans

    Encoding should be generated with return_offset_mapping=True, truncation=True, return_overflowing_tokens=True
    so that encoding['input_ids'] and encoding['offset_mapping'] are list of lists.

    Parameters
    ----------
    encoding : dict
        output dict of transformers tokenizer with keys for ['input_ids', 'offset_mapping']
    entity_spans : list of tuples
        list of tuples of ((start, end), entity_type), sorted by start
    tokenizer : transformers.tokenizer
        Tokenizer used to generate the encoding
    entity_encoding: dict
        mapping of IOB tags to integer encoding

    Returns
    -------
    encoded_labels: list
        list of np.ndarray of integer token labels
    """
    encoded_labels = []
    for encoded_tokens, offsets in zip(encoding['input_ids'], encoding['offset_mapping']):
        # Get actual tokens
        tokens = tokenizer.convert_ids_to_tokens(encoded_tokens)

        # Most tokens will be 'O'
        cls_labels = ['O'] * len(tokens)

        # Prepare to iterate through entities
        iter_spans = iter(entity_spans)
        span, token_type = next(iter_spans)

        # Iterate through tokens and label
        for i, (token, offset) in enumerate(zip(tokens, offsets)):
            # [CLS] and [SEP] tokens have offset (0, 0) and we label as 'X'
            if offset == (0, 0):
                cls_labels[i] = 'X'
                continue

            # Passed end of entity, get next one
            if offset[0] > span[1]:
                try:
                    span, token_type = next(iter_spans)
                except StopIteration:
                    break  # No more relevant tokens to tag

            # Assign 'X' to subtokens split by the WordPiece algorithm
            if token.startswith('##'):
                cls_labels[i] = 'X'
            # Beginning token, assign 'B'
            elif offset[0] == span[0]:
                cls_labels[i] = 'B-' + token_type
            # Inside token, assign 'I
            elif offset[0] >= span[0] and offset[1] <= span[1]:
                cls_labels[i] = 'I-' + token_type

        # Construct the label encoding for training
        encoded_labels.append(np.array([entity_encoding[ent] for ent in cls_labels], dtype=int))
    return encoded_labels


def ner_preprocess(snapshot, tokenizer, entity_labels, max_seq_len=512, stride=10):
    """Preprocessing transforms a snapshot datastructure with documents into a dataframe with ['text', 'label', 'id']
    These dicts can be later transformed into a pandas dataframe with all the training data.

    NER training labels are a vector of encoded labels corresponding to the IOB entity tagging scheme for that token.
    This is accomplished by using regex to search for the positions in the text where the entity occurs.
    The tokens within those position ranges are then assigned the appropriate encoding.
    Preditions are only generate for the first token. Partial tokens from the WordPiece algorithm are not predicted on.
    We split long texts into batches of tokens of max sequence length and use stride to prepend a number of tokens from
    the preceding sequence. Note that there will not be the exact number stride of tokens prepended as this method tries
    to ensure that we don't split on partial word pieces.

    Parameters
    ----------
    snapshot : dict of Document
        datastructure containing documents with texts and labels
    tokenizer : transformers.Tokenizer
        Tokenizer associated with intended model
    entity_labels : dict
        Mapping of label keys to entity names (e.g. {'Modality': 'MOD'})
    max_seq_len : int
        Maximum token sequence length for transformers model
    stride : int
        When splitting into phrases of max sequence length, how many tokens from previous sequence to prepend

    Returns
    -------
    training_data : pd.DataFrame
        DataFrame with columns ['text', 'label', 'id']
    """
    # Create entity encoding map from IOB labels to integers
    entity_encoding = get_iob_entity_encoding(entity_labels)

    # Iterate through snapshot and tag data
    output = []
    for _id, document in snapshot.items():
        text = document.text

        # Make labels on un-split text with max_length=None
        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=None,
            return_overflowing_tokens=True)

        # Find entities
        entity_spans = []
        for label_type, token_type in entity_labels.items():
            label_data = document.get(label_type, None)
            if label_data is None:
                continue
            word = label_data['true text']
            # Search for the word in the text
            flags = re.I if label_type == 'Clinic Name' else 0  # Ignore case when searching for clinic name
            for match in re.finditer(word, text, flags=flags):
                entity_spans.append((match.span(), token_type))
        if not entity_spans:
            warnings.warn(UserWarning(f'No entities found in this text: {_id}'))
            continue
        # Sort by start position of entity
        entity_spans.sort(key=lambda x: x[0][0])

        # Do labelling of entire text
        encoded_labels = label_encoded_tokens(encoding, entity_spans, tokenizer, entity_encoding)[0]
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        offsets = encoding['offset_mapping'][0]

        # Split text by max_seq_len
        chunk_size = max_seq_len - 2  # For [CLS] and [SEP]

        # First chunk
        start = 1  # Because 0th token is [CLS]
        end = min(start + chunk_size, len(tokens) - 1)
        # Move end forward to token that is not middle of word piece.
        while tokens[end].startswith('##'):
            end -= 1

        temp = []
        # Add chunks to temp
        while end < len(tokens) - 1:
            # Indices for sections of text
            start_idx = offsets[start][0]
            end_idx = offsets[end][0]
            # Remake labels to be of max_seq_len
            label = -100 * np.ones(max_seq_len, dtype='int')
            for i, val in enumerate(encoded_labels[start:end], start=1):  # Start at 1, first token is [CLS]
                label[i] = val
            temp.append({'text': text[start_idx:end_idx], 'label': label})

            # Move start to beginning of word piece
            start = end - stride
            while tokens[start].startswith('##'):
                start -= 1
            # Don't let end go to a bad index
            end = min(end + chunk_size, len(tokens) - 1)
            # Move end forward
            while tokens[end].startswith('##') or (end - start > chunk_size):
                end -= 1
        # Last chunk
        start_idx = offsets[start][0]
        end_idx = offsets[-2][1]  # -1 token/offset is [SEP]
        label = -100 * np.ones(max_seq_len, dtype='int')
        for i, val in enumerate(encoded_labels[start:-1], start=1):
            label[i] = val
        temp.append({'text': text[start_idx:end_idx], 'label': label})

        # Add temp items to output, remake id to include enumeration of the chunk
        for i, d in enumerate(temp):
            d['id'] = _id + ':' + str(i)
        output.extend(temp)

    training_data = pd.DataFrame(output)
    return training_data


def cls_preprocess(snapshot, tokenizer, cls_type, max_seq_len=512, stride=10):
    """Preprocessing transforms a snapshot datastructure with documents into a dataframe with ['text', 'label', 'id']
    These dicts can be later transformed into a pandas dataframe with all the training data.

    NER training labels are a vector of encoded labels corresponding to the IOB entity tagging scheme for that token.
    This is accomplished by using regex to search for the positions in the text where the entity occurs.
    The tokens within those position ranges are then assigned the appropriate encoding.
    Preditions are only generated for the first token. Partial tokens from the WordPiece algorithm are not predicted on.
    We split long texts into batches of tokens of max sequence length and use stride to prepend a number of tokens from
    the preceding sequence. Note that there will not be the exact number stride of tokens prepended as this method tries
    to ensure that we don't split on partial word pieces.

    Parameters
    ----------
    snapshot : dict of Document
        datastructure containing documents with texts and labels
    tokenizer : transformers.Tokenizer
        Tokenizer associated with intended model
    entity_labels : dict
        Mapping of label keys to entity names (e.g. {'Modality': 'MOD'})
    max_seq_len : int
        Maximum token sequence length for transformers model
    stride : int
        When splitting into phrases of max sequence length, how many tokens from previous sequence to prepend

    Returns
    -------
    training_data : pd.DataFrame
        DataFrame with columns ['text', 'label', 'id']
    """
    # Create entity encoding map from IOB labels to integers
    entity_encoding = get_iob_entity_encoding(entity_labels)

    # Iterate through snapshot and tag data
    output = []
    for _id, document in snapshot.items():
        text = document.text

        # Make labels on un-split text with max_length=None
        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=None,
            return_overflowing_tokens=True)

        # Find entities

        word_label = document.get(cls_type, None)['label']
        if cls_type == 'Modality':
            number_label = modality_labels[word_label]
        elif cls_type == 'Body Part':
            number_label = body_part_labels[word_label]
        else:
            raise Exception("cls_type must be 'MODALITY' or 'BODY_PART'")

        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        offsets = encoding['offset_mapping'][0]

        # Split text by max_seq_len
        chunk_size = max_seq_len - 2  # For [CLS] and [SEP]

        # First chunk
        start = 1  # Because 0th token is [CLS]
        end = min(start + chunk_size, len(tokens) - 1)
        # Move end backwards to token that is not middle of word piece.
        while tokens[end].startswith('##'):
            end -= 1

        temp = []
        # Add chunks to temp
        while end < len(tokens) - 1:
            # Indices for sections of text
            start_idx = offsets[start][0]
            end_idx = offsets[end][0]
            temp.append({'text': text[start_idx:end_idx], 'label': number_label})

            # Move start to beginning of word piece
            start = end - stride
            while tokens[start].startswith('##'):
                start -= 1
            # Don't let end go to a bad index
            end = min(end + chunk_size, len(tokens) - 1)
            # Move end forward
            while tokens[end].startswith('##') or (end - start > chunk_size):
                end -= 1
        # Last chunk
        start_idx = offsets[start][0]
        end_idx = offsets[-2][1]  # -1 token/offset is [SEP]
        temp.append({'text': text[start_idx:end_idx], 'label': number_label})

        # Add temp items to output, remake id to include enumeration of the chunk
        for i, d in enumerate(temp):
            d['id'] = _id + ':' + str(i)
        output.extend(temp)

    training_data = pd.DataFrame(output)
    return training_data


class TrainingDataset(Dataset):
    def __init__(self, df, tokenizer, tokenization_params):
        """Setup parameters for the dataset

        Parameters
        ----------
        df : pd.DataFrame
            training/validation data
        tokenizer : transformers tokenizer
            Tokenizer to do encoding of text
        tokenization_params : dict
            Parameters for tokenization passed to tokenizer. Note, do not set {'return_tensors': 'pt'} because this adds
            an extra dimension. Keep 'return_tensors' to default as this function does the torch tensor conversion when
            it returns the item.
            Suggested parameters: {'max_length': 512, 'truncation': True, 'padding': 'max_length'}
        """
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.params = tokenization_params
        self.len = len(df)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """Processing when returning an item from the Dataset. Converts text into encoding using tokenizer.
        Converts items from encoding to PyTorch tensors

        Parameters
        ----------
        index : int
            index of item to access

        Returns
        -------
        dict
            Output encoding from the tokenizer on the text.
            Relevant keys are ['input_ids', 'attention_mask', 'token_type_ids', 'label']
            What keys are present may depend on tokenizer parameters
        """
        text = self.df.loc[index, 'text']
        label = self.df.loc[index, 'label']
        # Encode the text
        encoding = self.tokenizer(text, **self.params)
        encoding['label'] = label
        encoding = {key: torch.as_tensor(val) for key, val in encoding.items()}
        return encoding


def df_to_dataloader(df, tokenizer, tokenization_params, batch_size=4, shuffle=True, num_workers=0):
    """Transform a dataframe with training data into a PyTorch dataloader for training

    Parameters
    ----------
    df : pd.DataFrame
        training or validation data
    tokenizer : huggingface tokenizer
        Tokenizer to do encoding
    tokenization_params : dict
        Parameters for tokenization passed to tokenizer. Note, do not set {'return_tensors': 'pt'} because this adds an
        extra dimension. Keep 'return_tensors' to default as this function does the torch tensor conversion when it
        returns the item
    batch_size : int, optional
        batch size, by default 4
        For sequence length of 512, batch size of 4 is largest. With smaller sequence lengths, use larger batch sizes
    shuffle : bool, optional
        Whether to shuffle the data, by default True
    num_workers : int, optional
        Number of CPU cores to use for paralleization, by default 0

    Returns
    -------
    torch.utils.data.Dataloader
        PyTorch dataloader for the training data
    """
    dataset = TrainingDataset(df, tokenizer, tokenization_params)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
    return dataloader
