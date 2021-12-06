import json
import csv
import os
import pandas as pd

path = f'./dataset/result_train'
save_path = f'./dataset/processed_train'

calls_list = []
count = 0

for idx in range(8, 9):
    iter_path = f'{path}_{idx}'
    if not os.path.isdir(iter_path): continue

    data_list = os.listdir(iter_path)
    for data_fname in data_list:
        count += 1
        print(f'{iter_path} folder: {count}')
        try:
            json_file = open(f'{iter_path}/{data_fname}')
            report_file = json.load(json_file)
            json_file.close()
        except:
            continue

        if 'behavior' in report_file.keys():
            vir_name = report_file['target']['file']['name']
            if os.path.exists(f'{save_path}/{vir_name}.csv'): continue
            csv_file = open(f'{save_path}/{vir_name}.csv', 'w', newline='')
            writer = csv.writer(csv_file)
            writer.writerow(['process_path', 'category', 'api', 'time'])

            processes = report_file['behavior']['processes']
            for process in processes:
                for call in process['calls']:
                    writer.writerow([process['process_path'], call['category'], call['api'], call['time']])
            csv_file.close()
        else:
            continue
