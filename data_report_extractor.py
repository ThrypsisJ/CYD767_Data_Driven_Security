import json
import os
import pandas

data_path = './dataset'
path_list = os.listdir(data_path)

calls_list = []

# path: splited folder name of dataset
for path in path_list:
    data_list = os.listdir(f'{data_path}/{path}')

    # ignore processed folder
    if 'processed' in path:
        continue

    # check whether the data is for train or test
    save_path = 'train' if 'train' in path else 'test'
    save_path = f'./dataset/processed_{save_path}'

    for data_fname in data_list:
        with open(f'{data_path}/{data_fname}') as file:
            report_file = json.load(file)

        if 'behavior' in report_file.keys():
            processes = report_file['behavior']['processes']
            calls = []
            for process in processes:
                tmp_calls = []
                for call in process['calls']:
                    tmp_call = {
                        'process_path': process['process_path'],
                        'category': call['category'],
                        'api': call['api'],
                        'time': call['time']
                    }
                    tmp_calls.append(tmp_call)
                if not len(tmp_calls) == 0: calls += tmp_calls
            calls_list += calls

            data = pandas.DataFrame(calls_list)
            virname = report_file['target']['file']['name']
            data.to_feather(f'{save_path}/{virname}.ftr')
        else:
            continue

data = pandas.DataFrame(calls_list)
print(data)