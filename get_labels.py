import numpy as np
import pandas as pd

def get_labels(csv_path):
    '''
    Function to return labels for the classes

    Input: 
        csv_path: path for the csv file containing the labels and RGB codes

    Outputs:
        code2id: RGB color code to class ID
        id2code: Class ID to RGB color code
        name2id: Class name to class ID
        id2name: Class ID to class name
    '''

    class_dict_df = pd.read_csv(f'{csv_path}', index_col=False, skipinitialspace=True)
    print('\n\nLabels & RGB Codes: \n',class_dict_df, '\n\n')
    
    label_names= list(class_dict_df['class'])
    label_codes = []
    r= np.asarray(class_dict_df.r)
    g= np.asarray(class_dict_df.g)
    b= np.asarray(class_dict_df.b)

    for i in range(len(class_dict_df)):
        label_codes.append(tuple([r[i], g[i], b[i]]))
        
    code2id = {v:k for k,v in enumerate(label_codes)}
    id2code = {k:v for k,v in enumerate(label_codes)}

    name2id = {v:k for k,v in enumerate(label_names)}
    id2name = {k:v for k,v in enumerate(label_names)}

    return code2id, id2code, name2id, id2name, label_names, label_codes