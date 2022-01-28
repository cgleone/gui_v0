import pickle

import pandas as pd
#
#
# class SearchBar:
#
#
#     def convert_term_to_label(self, term)
#
#
#
#
#
#
# class SearchQuery(SearchBar):



from models.minimal_model import MinimalModel

with open('models/minimal_model_data/parameters.pkl', 'rb') as f:
    parameters = pickle.load(f)

nlp = MinimalModel()
nlp.set_parameters(parameters)
file = open("report_texts/MinimalModel0_62.txt", 'r')
text = file.read()
data = {0: text}
file.close()
idk_a_result_maybe = nlp.predict(data)
print(idk_a_result_maybe)
