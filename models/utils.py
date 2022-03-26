import io
import time
import boto3
from smart_open import open
import pickle
from thefuzz import fuzz
import torch
import pandas as pd


TRUE_BODY_PART_LABELS = {
    'CHEST ONLY': 'CHEST',
    'CHEST': 'CHEST',
    'RENAL': 'ABDOMEN',
    'LIVER OR GALLBLADDER': 'ABDOMEN',
    'ABDOMEN': 'ABDOMEN',
    'ABD': 'ABDOMEN',
    'SINUS/MANDIBLE/MAXILLOFACIAL': 'HEAD',
    'HEAD': 'HEAD',
    'PELVIS': 'PELVIS',
    'ABD/PEL': 'ABDOMEN PELVIS',
    'ABD & PELVIS': 'ABDOMEN PELVIS',
    'C-SPINE': 'SPINE',
    'CERVICAL SPINE': 'SPINE',
    'T-SPINE': 'SPINE',
    'L-SPINE': 'SPINE',
    'BABYGRAM CHEST & ABD': 'CHEST ABDOMEN',
    'CXR & PELVIS': 'CHEST PELVIS'}

TRUE_MODALITY_LABELS = {
    'CT': 'CT',
    'CTA': 'CT',
    'U.S.': 'US',
    'NEONATAL HEAD PORTABLE': 'US',
    'US': 'US',
    'DUPLEX': 'US',
    'MR': 'MRI',
    'X-RAY, AP & LAT': 'X-RAY',
    'X-RAY, SUPINE & ERECT': 'X-RAY',
    'X-RAY, PORTABLE': 'X-RAY',
    'X-RAY, PRE-OP PA & LAT': 'X-RAY',
    'X-RAY, BABYGRAM': 'X-RAY',
    'X-RAY, AP CXR': 'X-RAY',
    'X-RAY, nan': 'X-RAY',
    'X-RAY, PA & LAT': 'X-RAY',
    'X-RAY, PORTABLE AP': 'X-RAY',
    'X-RAY, PORT.': 'X-RAY',
    'X-RAY, CHEST (SINGLE VIEW)': 'X-RAY',
    'SUPINE ONLY': None}

CLINIC_NAME_LIST = pd.read_csv('gui_v0/institution_list.csv')['Names'].tolist()

def generate_default_parameters(
    max_seq_len=512,
    stride=10,
    batch_size=4,
    learning_rate=1e-6,
    max_grad_norm=10,
    random_seed=42,
    epochs=5,
    tokenizer_url='emilyalsentzer/Bio_ClinicalBERT',
    base_model_url='emilyalsentzer/Bio_ClinicalBERT',
    tokenizer_params=None,
    use_cuda=True
):
    """Generate a set of default model parameters

    Parameters
    ----------
    max_seq_len : int, optional
        maximum token sequence length for model, by default 512
    stride : int, optional
        number of tokens to prepend when splitting texts longer than max_seq_len, by default 10
    batch_size : int, optional
        batch size for training, by default 4
    learning_rate : float, optional
        learning rate for optimizer, by default 1e-6
    max_grad_norm : int, optional
        maximum gradient to clip during backprop, by default 10
    random_seed : int, optional
        random seed for numpy and pytorch random number generators, by default 42
    epochs : int, optional
        number of epochs to train, by default 5
    tokenizer_url : str, optional
        huggingface url to construct pretrained tokenizer, by default 'emilyalsentzer/Bio_ClinicalBERT'
    base_model_url : str, optional
        huggingface url to construct base model, by default 'emilyalsentzer/Bio_ClinicalBERT'
    tokenizer_params : dict, optional
        tokenizer parameters for inference, by default None
    use_cuda : bool, optional
        whether to try to use cuda on this device, by default True

    Returns
    -------
    dicr
        parameters dictionary to pass to model.set_parameters()
    """
    params = {
        'max_seq_len': max_seq_len,
        'stride': stride,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_grad_norm': max_grad_norm,
        'random_seed': random_seed,
        'epochs': epochs,
        'tokenizer_url': tokenizer_url,
        'base_model_url': base_model_url,
        'tokenizer_params': tokenizer_params,
        'use_cuda': use_cuda
    }
    if tokenizer_params is None:
        params['tokenizer_params'] = {
            'max_length': max_seq_len,
            'truncation': True,
            'padding': 'max_length'
        }
    return params


def save_model_to_aws(model, val_data_id, metrics=None, s3_bucket='ty-capstone-test', s3_dir='model_training'):
    """Saves model.nn weights and model.parameters to s3. If metrics are given, save those.
    Updates a model_list.csv file with the relevant info of the experiment so you can lookup the id later

    Parameters
    ----------
    model : TrainingModel
        Model to save
    val_data_id : str or int
        Identitifcation of data left out for validation
    metrics : dict or None
        Training metrics to save, by default None
    s3_bucket : str, optional
        s3 bucket to save to, by default 'ty-capstone-test'
    s3_dir : str, optional
        directory (key) in the s3 bucket to save to, by default 'model_training'

    Returns
    -------
    int
        id of saved model
    """
    client = boto3.client('s3')

    # Figure out naming of files
    name = type(model).__name__
    ts = time.strftime("%y-%m-%d %H:%M:%S")
    # id is just the digits of the timestamp
    id = ts.replace('-', '').replace(':', '').replace(' ', '')
    model_fname = f"{name}_model_{id}.pt"
    parameters_fname = f"{name}_parameters_{id}.pkl"

    # Update the parameters to include location of saved model
    params = model.get_parameters()
    params['trained_model_url'] = f's3://{s3_bucket}/{s3_dir}/{model_fname}'

    if name != "QaModel" and name != "NbModel":
        print(f'Saving model weights in {model_fname}')
        # Save the nn weights directly
        buffer = io.BytesIO()
        torch.save(model.nn, buffer)
        client.put_object(Bucket=s3_bucket, Key=f'{s3_dir}/{model_fname}', Body=buffer.getvalue())
    elif name == "QaModel":
        # Use .pkl extension since we're pickling this model
        model_fname = f"{name}_model_{id}.pkl"
        params['trained_model_url'] = f's3://{s3_bucket}/{s3_dir}/{model_fname}'
        print(f'Saving inferencer model weights in {model_fname}')
        pickle_bytes = pickle.dumps(model.reader.inferencer.model)
        client.put_object(Bucket=s3_bucket, Key=f'{s3_dir}/{model_fname}', Body=pickle_bytes)
    elif name == "NbModel":
        # Use .pkl extension since we're pickling this model
        model_fname = f"{name}_model_{id}.pkl"
        params['trained_model_url'] = f's3://{s3_bucket}/{s3_dir}/{model_fname}'
        print(f'Saving Naive Bayes model in {model_fname}')
        pickle_bytes = pickle.dumps(model.naive_bayes)
        client.put_object(Bucket=s3_bucket, Key=f'{s3_dir}/{model_fname}', Body=pickle_bytes)

    # Save parameters
    print(f'Saving model parameters in {parameters_fname}')
    pickle_bytes = pickle.dumps(params)
    client.put_object(Bucket=s3_bucket, Key=f'{s3_dir}/{parameters_fname}', Body=pickle_bytes)

    # Save metrics
    if metrics:
        metrics_fname = f"{name}_metrics_{id}.pkl"
        print(f'Saving model metrics in {metrics_fname}')
        pickle_bytes = pickle.dumps(metrics)
        client.put_object(Bucket=s3_bucket, Key=f'{s3_dir}/{metrics_fname}', Body=pickle_bytes)
    else:
        metrics_fname = ''

    # Add to relavent info to model_list.csv
    info = {
        'id': id,
        'date_saved': ts,
        'model_type': name,
        'base_model_url': params['base_model_url'],
        'val_data_id': val_data_id,
        'max_seq_len': params['max_seq_len'],
        'learning_rate': params['learning_rate'],
        'epochs': params['epochs'],
        'parameters_url': f's3://{s3_bucket}/{s3_dir}/{parameters_fname}',
        'model_url': f's3://{s3_bucket}/{s3_dir}/{model_fname}',
        'metrics_url': f's3://{s3_bucket}/{s3_dir}/{metrics_fname}' if metrics else ''
    }
    # Grab the model_list file from s3
    key = f'{s3_dir}/model_list.csv'
    read_file = client.get_object(Bucket=s3_bucket, Key=key)
    df = pd.read_csv(read_file['Body'])
    # Add new row and save
    df = df.append(info, ignore_index=True)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    client.put_object(Bucket=s3_bucket, Key=key, Body=csv_buffer.getvalue())
    print(f'Successfully saved to AWS with ID: {id}')
    return id


def load_nn_from_aws(s3_url):
    """Utility to load a nn weights from AWS s3

    Parameters
    ----------
    s3_url : str
        URL to weights, starting with s3://<bucket name>/path_to_weights

    Returns
    -------
    torch.nn.Module
        PyTorch neural network
    """
    client = boto3.client('s3')
    with open(s3_url, 'rb', transport_params={'client': client}) as f:
        nn = torch.load(f)
    return nn


def load_pickle_from_aws(s3_url):
    """Utility to load pickle from AWS s3

    Parameters
    ----------
    s3_url : str
        URL to weights, starting with s3://<bucket name>/path_to_pickle

    Returns
    -------
    dict
        Parameters dict
    """
    client = boto3.client('s3')
    with open(s3_url, 'rb', transport_params={'client': client}) as f:
        params = pickle.load(f)
    return params


def load_model_list_from_aws(s3_bucket='ty-capstone-test', s3_key='model_training/model_list.csv'):
    """Utility to quickly load the model list into a dataframe

    Parameters
    ----------
    s3_bucket : str, optional
        by default 'ty-capstone-test'
    s3_key : str, optional
        by default 'model_training/model_list.csv'

    Returns
    -------
    pd.DataFrame
        DataFrame of the model_list.csv file
    """
    client = boto3.client('s3')
    read_file = client.get_object(Bucket=s3_bucket, Key=s3_key)
    df = pd.read_csv(read_file['Body'])
    return df


def score_tags(doc, pred):
    """Score tag predictions made by model
    Body part, modality, and date taken are scored by exact match to the label in the Document
    Doctor name is scored by ensuring the last name was contained in the extracted tag
    Clinic name is fuzzy matched from the extracted tag

    Parameters
    ----------
    doc : Document
        Document containing true tags and text
    pred : dict
        Dictionary output from model predict with keys ['Body Part', 'Modality', 'Clinic Name', 'Date Taken',
        'Doctor Name'] and values as the tags the model predicts for the same document.

    Returns
    -------
    dict
        score, dict with same keys as pred and values of scores (mostly 1/0 but float for clinic name fuzzy match)
    """
    score = {}

    for k in ['Body Part', 'Modality', 'Clinic Name', 'Date Taken', 'Doctor Name']:
        true_tag = doc[k]['label']
        pred_tag = pred[k]

        if k == 'Doctor Name':
            true_tag = true_tag.split()[-1]  # Grab only last name

        if pred_tag is None:  # No tag predicted, score 0
            score[k] = 0
        elif k == 'Doctor Name' and true_tag in pred_tag:  # Last name in predicted tag
            score[k] = 1
        elif k == 'Clinic Name':  # Fuzzy match score normalized to [0, 1]
            score[k] = fuzz.ratio(true_tag, pred_tag) / 100
        elif true_tag == pred_tag:  # Most things are just exact match
            score[k] = 1
        else:
            score[k] = 0
    return score
