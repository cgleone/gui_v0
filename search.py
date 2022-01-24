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


def read_csv():
    all_institutions = pd.read_csv('institution_list.csv')
    print(all_institutions)


read_csv()