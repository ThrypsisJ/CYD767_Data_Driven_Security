import json
import csv
import os
import pandas as pd

splitted = 'train_2'
path = f'./dataset/result_{splitted}'
save_path = f'./dataset/processed_train'

calls_list = []
count = 0

data_list = os.listdir(path)

csv_file = open(f'./dataset/result_{splitted}.csv', 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['vir_name', 'process_path', 'category', 'api', 'time'])

for data_fname in data_list:
    count += 1
    print(count)
    try:
        json_file = open(f'{path}/{data_fname}')
        report_file = json.load(json_file)
        json_file.close()
    except:
        continue

    if 'behavior' in report_file.keys():
        processes = report_file['behavior']['processes']
        for process in processes:
            for call in process['calls']:
                writer.writerow([report_file['target']['file']['name'], process['process_path'], call['category'], call['api'], call['time']])
    else:
        continue

csv_file.close()