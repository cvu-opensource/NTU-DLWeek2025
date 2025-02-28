import json 

with open('./finetune/dataset/clean_with_scores.json') as file:
    datas = json.load(file)
    for data in datas:
        print(data.keys())