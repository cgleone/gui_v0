from models.cls_model import ClsModel
from models.utils import load_model_list_from_aws, load_params_from_aws, load_training_and_val_data

if __name__=="__main__":
    df = load_model_list_from_aws()
    df = df[(df['model_type']=='ClsModel') & (df['base_model_url']=='emilyalsentzer/Bio_ClinicalBERT')]
    for i, row in df.itterrows():
        model = ClsModel()
        params = load_params_from_aws(row['parameters_url'])
        model.set_parameters(params)
        training_data, val_data = load_training_and_val_data(row['val_data_id'])
        print(model.evaluate(val_data))
