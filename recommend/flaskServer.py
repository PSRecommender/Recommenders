from flask import Flask, request, jsonify
from datetime import datetime
import json
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
import sys

from model import set_model, predict

app = Flask(__name__)

headers = {'User-Agent': 'Mozilla/5.0'}

model = set_model()

def toJson(dir, dict):
    with open(dir, 'w', encoding='utf-8') as file:
        json.dump(dict, file, ensure_ascii=False, indent='\t')

def loadJson(dir):
    with open(dir, 'r', encoding='utf-8') as file:
        return json.load(file)

# 사용자가 푼 문제 히스토리 크롤링
# 이미 크롤링한 사용자일 경우 문제 히스토리가 업데이트 됐을 때만 크롤링
def getUserData(userId):
    dir = '/tf/Recommenders/recommend/data/' + userId
    isMember = False
    isUpdate = True
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
        else: isMember = True
    except OSError:
        print('Error : mkdir')
    if isMember:
        userDict = loadJson(dir + '/{userId}.json'.format(userId=userId))
        lastSub = userDict[0]['sbNum']
        find = False

    url = "https://www.acmicpc.net/status?user_id={userId}&result_id=4".format(userId=userId)
    sbDict = []
    cnt = 1
    response = requests.get(url, headers=headers)
    while True:
        if response.status_code == 200:
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.select_one('tbody')
            tr = table.select('tr')
            for i in range(len(tr)):
                td = tr[i].select('td')
                sbNum = td[0].text
                if isMember:
                    if sbNum == lastSub: 
                        find = True
                        if i == 0: isUpdate = False
                        break
                uId = td[1].text
                pId = td[2].text
                result = td[3].text
                memory = td[4].text
                time = td[5].text
                lang = td[6].text
                codeLen = td[7].text
                sbTime = td[8].select_one('a')['data-timestamp']
                sbDict.append({'sbNum':sbNum, 'uId':uId, 'pId':pId, 'result':result, 'memory':memory, 'time':time, 'lang':lang, 'codeLen':codeLen, 'sbTime':sbTime})
            if isMember and find: break
            try:
                nextPage = 'https://www.acmicpc.net' + soup.select_one('#next_page')['href']
            except Exception:
                break
            response = requests.get(nextPage, headers=headers)
            print(cnt)
            cnt = cnt + 1
    if isUpdate:
        if isMember:
            sbDict = sbDict + userDict
        toJson(dir + '/' + userId + ".json", sbDict)
    return isUpdate

# 사용자가 푼 문제 히스토리 데이터 전처리
def data_preprocessing(userId):
    # 문제 태그 데이터 가져오기
    dir = '/tf/Recommenders/recommend/'
    categoryDF = pd.read_csv(dir + 'category.csv', index_col=0, dtype=str, encoding='utf-8')

    file = dir + 'data/{userId}/{userId}.json'.format(userId=userId)
    userDF = pd.read_json(file, dtype = str, encoding='utf-8')
    # 맞은 문제만 추출
    userDF = userDF[userDF['result']=='맞았습니다!!']
    # 문제셋에 해당하는 문제만 추출
    userDF = userDF[userDF['pId'].isin(categoryDF['pId'])]
    
    # erase duplications
    userDF.drop_duplicates(subset=['pId'], keep='last', inplace=True, ignore_index=True)
    
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
    f = open(dir+'data/{userId}/test_{userId}'.format(userId=userId),'w')
    for i in index:
        data = test_problem.loc[i]
        pId = data['pId']
        cate = data['category']
        f.write(label + '\t' + userId + '\t' + pId + '\t' + cate + '\t' + now + '\t' + pStr + '\t' + cStr + '\t' + tStr + '\n')
    f.close()
    

# 추천 결과 가져오기
def getRecommend(userId, isUpdate):
    dir = '/tf/Recommenders/recommend/data/' + userId
    if isUpdate:
        score = pd.read_csv(dir + '/output_{}.txt'.format(userId),sep='\t',header=None,dtype=float,names=['score'])
        test = pd.read_csv(dir + '/test_{}'.format(userId), sep='\t',header=None,dtype=str,names=['label', 'uId', 'pId', 'cate', 'time', 'pHis', 'cHis', 'tHis'])
        df = pd.concat([score, test], axis=1)
        df = df.loc[df['score'].sort_values(ascending=False).index].head(20)
        df = df.reset_index().drop(['index'], axis=1)
        rec = list(df[df['score']>0.5]['pId'])
        problems = loadJson('/tf/Recommenders/recommend/problemData.json')
        recDict = []
        for r in rec:
            for p in problems:
                if int(r) == p['pId']:
                    recDict.append(p)
        toJson(dir+"/recommend.json", recDict)
        print("make recommend")
    else:
        recDict = loadJson(dir+"/recommend.json")
        print("get recommend")
    return recDict

# 추천
@app.route("/recommend", methods=['POST'])
def recommend():
    userId = request.json['userId']
    isUpdate = getUserData(userId)
    if(isUpdate):
        data_preprocessing(userId)
        predict(model, userId)
    recDict = getRecommend(userId, isUpdate)
    return jsonify(recDict)

# 사용자 인증
@app.route("/valid", methods=['GET'])
def validation():
    userId = request.args.get('userId')
    # 백준 사이트에서 사용자 검색해서 나오면 True, 에러 페이지로 가면 False
    url = "https://www.acmicpc.net/user/{}".format(userId)
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        check = True
    elif response.status_code == 404:
        check = False
    print(response.status_code)
    return jsonify({"check":check})


if __name__ == '__main__':
    app.run()
