import json
import csv
import os
import pandas as pd

path = f'./dataset/result_test'
save_path = f'./dataset/processed_test'

calls_list = []
count = 0

for idx in range(1, 4):
    iter_path = f'{path}_{idx}'
    # iter_path = path
    if not os.path.isdir(iter_path): continue

    data_list = os.listdir(iter_path)
    for data_fname in data_list:
        count += 1
        print(f'{iter_path} folder: {count}')
        json_file = open(f'{iter_path}/{data_fname}')
        report_file = json.load(json_file)
        json_file.close()

        vir_name = report_file['target']['file']['name']
        csv_file = open(f'{save_path}/{vir_name}.csv', 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(['process_path', 'category', 'api', 'time'])
        if 'behavior' in report_file.keys():
            # if os.path.exists(f'{save_path}/{vir_name}.csv'): continue
            processes = report_file['behavior']['processes']
            for process in processes:
                for call in process['calls']:
                    writer.writerow([process['process_path'], call['category'], call['api'], call['time']])
        else:
            pass
        csv_file.close()