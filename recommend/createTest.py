from pickle import FALSE
from flask import Flask
from selenium import webdriver
import time
from datetime import datetime
import json
import pandas as pd
import os

app = Flask(__name__)

def toJson(dir, dict):
    with open(dir, 'w', encoding='utf-8') as file:
        json.dump(dict, file, ensure_ascii=False, indent='\t')

def loadJson(dir):
    with open(dir, 'r', encoding='utf-8') as file:
        return json.load(file)

def getUserData(userId):
    options = webdriver.ChromeOptions()
    options.add_argument('--headleass')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome('chromedriver', options=options)
    driver.implicitly_wait(1)
    url = "https://www.acmicpc.net/status?user_id={userId}&result_id=4".format(userId=userId)
    sbDict = []
    cnt = 1
    driver.get(url)
    while True:
        table = driver.find_element_by_tag_name('tbody')
        tr = table.find_elements_by_tag_name('tr')
        for i in tr:
            td = i.find_elements_by_tag_name('td')
            sbNum = td[0].text
            uId = td[1].text
            pId = td[2].text
            result = td[3].text
            memory = td[4].text
            time = td[5].text
            lang = td[6].text
            codeLen = td[7].text
            sbTime = td[8].find_element_by_tag_name('a').get_attribute('data-original_title')
            sbDict.append({'sbNum':sbNum, 'uId':uId, 'pId':pId, 'result':result, 'memory':memory, 'time':time, 'lang':lang, 'codeLen':codeLen, 'sbTime':sbTime})
        try:
            nextPage = driver.find_element_by_id('next_page').get_attribute('href')
        except Exception:
            break
        driver.get(nextPage)
        print(cnt)
        cnt = cnt + 1
    try:
        if not os.path.exists('data/'+userId):
            os.makedirs('data/'+userId)
    except OSError:
        print('Error : mkdir')
    toJson('data/' + userId + '/' + userId + ".json", sbDict)
    driver.quit()

def data_preprocessing(userId):
    # 문제 태그 데이터 가져오기
    categoryDF = pd.read_csv('category.csv', index_col=0, dtype=str, encoding='utf-8')

    file = 'data/{userId}/{userId}.json'.format(userId=userId)
    userDF = pd.read_json(file, dtype = str, encoding='utf-8')
    # 맞은 문제만 추출
    userDF = userDF[userDF['result']=='맞았습니다!!']
    # 문제셋에 해당하는 문제만 추출
    userDF = userDF[userDF['pId'].isin(categoryDF['pId'])]
    # 문제 제출 번호에 따라 오름차순으로 정렬
    userDF = userDF.sort_values(by='sbNum')
    if len(userDF) == 0: return
    # 사용자 데이터에서 제출시간, 아이디, 문제 데이터만 추출하여 사용자가 푼 문제에 해당하는 태그 붙이기
    userDF = userDF[['sbTime','uId','pId']]
    df = pd.merge(userDF,categoryDF,how='left',on='pId')

    # 라벨 붙이기
    df['label'] = [1 for i in range(len(df))]

    # '라벨', '아이디', '문제', '카테고리', '제출시간'순으로 열 바꾸기
    df = df[['label', 'uId', 'pId', 'category', 'sbTime']]
    
    # 히스토리 생성
    pHis = df['pId']
    cHis = df['category']
    tHis = df['sbTime']
    pHis = df['pId']
    pStr = ""
    cStr = ""
    tStr = ""
    for p in pHis:
        pStr = pStr + p + ","
    pStr = pStr[:-1]
    for c in cHis:
        cStr = cStr + c + ","
    cStr = cStr[:-1]
    for t in tHis:
        tStr = tStr + t + ","
    tStr = tStr[:-1]
    
    # 테스트 데이터 생성
    test_problem = categoryDF.copy()
    solved_problem = df['pId']
    remove_index = test_problem[test_problem['pId'].isin(solved_problem)].index
    test_problem = test_problem.drop(remove_index)
    index = test_problem.index
    label = "0"
    now = str(int(datetime.now().timestamp()))
    f = open('test_'+ userId,'w')
    for i in index:
        data = test_problem.loc[i]
        pId = data['pId']
        cate = data['category']
        f.write(label + '\t' + userId + '\t' + pId + '\t' + cate + '\t' + now + '\t' + pStr + '\t' + cStr + '\t' + tStr + '\n')


@app.route("/")
def hello():
    return "Hello world"

if __name__ == '__main__':
    app.run()