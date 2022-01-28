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


nlp = MinimalModel()
file = open("report_texts/MinimalModel0_62.txt", 'r')
text = file.read()
data = {0: text}
file.close()
idk_a_result_maybe = nlp.predict(data)
print(idk_a_result_maybe)
