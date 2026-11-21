import os
from mindware.utils.data_manager import DataManager
from mindware.components.utils.constants import CLS_TASKS, RGS_TASKS


def get_dataset_info(task_type, dataset, datasets_dir = '/root/xbc/amlb_datasets'):
    
    

    if task_type in RGS_TASKS:
        dataset_path = os.path.join(datasets_dir, 'rgs_datasets', dataset + '.csv')
    elif task_type in CLS_TASKS:
        dataset_path = os.path.join(datasets_dir, 'cls_datasets', dataset + '.csv')
    else:
        raise ValueError('Invalid task type %s!' % task_type)

    label_column = -1
    header = 'infer'

    sep = ','

    return dataset_path, label_column, header, sep



def load_data(task_type, dataset, datasets_dir = '/root/xbc/amlb_datasets'):
    
    dataset_path, label_column, header, sep = get_dataset_info(task_type, dataset, datasets_dir)

    dm = DataManager()
    df = dm.load_csv(dataset_path, header=header, sep=sep)
    
    train_data_node = dm.from_train_df(df, label_col=label_column)
    train_data_node = dm.preprocess_fit(train_data_node, task_type)

    return train_data_node
