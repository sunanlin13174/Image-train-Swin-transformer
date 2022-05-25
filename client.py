import json
import requests
import os
if __name__=='__main__':
    url = 'http://127.0.0.1:5000'         #注意细节，这里是/response，那么sever中，@app.route('')第一个参数也要是/response
    while True:
        input_img_folder = input('请输入要查询的图片路径：-1🖕中止\n')
        if input_img_folder.strip()=='-1':
            break
        elif input_img_folder.strip()=='':
            continue
        else:
            files_dict = {
                'img_path':input_img_folder
            }

            result_respose = requests.post(url,json= json.dumps(files_dict))     ##若传图片数据，则data=
            print('在线模型预测结果为：%s'%result_respose.text)