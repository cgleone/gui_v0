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
