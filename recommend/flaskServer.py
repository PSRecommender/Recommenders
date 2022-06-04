from flask import Flask, request, jsonify
from datetime import datetime
import json
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
import sys
import random
# from model import set_model, predict

from path import path
from btype import recommendForB

app = Flask(__name__)

headers = {'User-Agent': 'Mozilla/5.0'}

# model = set_model()

def toJson(dir, dict):
    with open(dir, 'w', encoding='utf-8') as file:
        json.dump(dict, file, ensure_ascii=False, indent='\t')

def loadJson(dir):
    with open(dir, 'r', encoding='utf-8') as file:
        return json.load(file)

# 사용자가 푼 문제 히스토리 크롤링
# 이미 크롤링한 사용자일 경우 문제 히스토리가 업데이트 됐을 때만 크롤링
def getUserData(userId):

    dir = path + 'data/' + userId
    isMember = False
    isUpdate = True
    try:
        if not os.path.exists(dir):
            os.mkdir(dir)
            token_file = open(dir+'/{}_token.txt'.format(userId), 'w', encoding='utf-8')
            token = str(random.randrange(0, 2))
            token_file.write(token)
            token_file.close()
            print(token)
        else: isMember = True
        print(isMember)
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
    dir = path
    categoryDF = pd.read_csv(dir + 'category.csv', index_col=0, dtype=str, encoding='utf-8')

    file = dir + 'data/{userId}/{userId}.json'.format(userId=userId)
    userDF = pd.read_json(file, dtype = str, encoding='utf-8')
    # 맞은 문제만 추출
    userIndex = list(userDF[userDF['result']=='맞았습니다!!'].index)
    userIndex = userIndex + list(userDF[userDF['result']=='100점'].index)
    userDF = userDF.loc[userIndex]
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
def getRecommend(userId, isUpdate, type):
    dir = path + 'data/' + userId
    # token_file = open(dir+'/{}_token.txt'.format(userId), 'r', encoding='utf-8')
    # token = token_file.readline()
    # token_file.close()
    if isUpdate:
        if type == 'B':
            recommendProblems = recommendForB(userId)
        else:
            score = pd.read_csv(dir + '/output_{}.txt'.format(userId),sep='\t',header=None,dtype=float,names=['score'])
            test = pd.read_csv(dir + '/test_{}'.format(userId), sep='\t',header=None,dtype=str,names=['label', 'uId', 'pId', 'cate', 'time', 'pHis', 'cHis', 'tHis'])
            df = pd.concat([score, test], axis=1)
            df = df.loc[df['score'].sort_values(ascending=False).index]
            df = df.reset_index().drop(['index'], axis=1)
            df = df[df['score']>0.5]
            df = df[['score','pId','cate']]
            problemDF = pd.read_csv(path + 'problemInfo.csv', index_col=0, dtype=str)
            problemDF = problemDF[['pId','successCnt']]
            problemDF = problemDF.astype({'successCnt':'int'})
            df = pd.merge(df, problemDF, how='left', on='pId')
            scoreList = list(df['score'])
            newScoreList = []
            for s in scoreList:
                newScoreList.append(float("{:.1f}".format(s)))
            df['score'] = newScoreList
            #print(df[df['successCnt'].isnull()])
            #df = df.astype({'successCnt':'int'})
            df = df.sort_values(['score','successCnt'], ascending=[False,False], ignore_index=True)
            tagCnt = {}
            recommendProblems = []
            for i in range(len(df)):
                data = df.loc[i]
                id = data['pId']
                tag = data['cate']
                keys = tagCnt.keys()
                if tag in keys:
                    tagCnt[tag] += 1
                else:
                    tagCnt[tag] = 1
                if tagCnt[tag]<4:
                    recommendProblems.append(id)
                if len(recommendProblems) == 10:
                    break
        toJson(dir+"/recommend{}.json".format(type), recommendProblems)
        print("make recommend{}".format(type))
    else:
        recommendProblems = loadJson(dir+"/recommend{}.json".format(type))
        print("get recommend{}".format(type))
    recDict = []
    apiUrl = 'https://solved.ac/api/v3/problem/lookup?problemIds='
    problemStr = ""
    for p in recommendProblems:
        problemStr += p + ","
    problemStr = problemStr[:-1]
    apiResponse = requests.get(apiUrl+problemStr).json()
    for p in recommendProblems:
        for pb in apiResponse:
            pId = pb['problemId']
            if int(p) == pId:
                recDict.append(pb)
    return recDict

def getReport(userId):
    dir = path
    categoryDF = pd.read_csv(dir + 'category.csv', index_col=0, dtype=str, encoding='utf-8')

    file = dir + 'data/{userId}/{userId}.json'.format(userId=userId)
    userDF = pd.read_json(file, dtype = str, encoding='utf-8')
    # 맞은 문제만 추출
    userIndex = list(userDF[userDF['result']=='맞았습니다!!'].index)
    userIndex = userIndex + list(userDF[userDF['result']=='100점'].index)
    userDF = userDF.loc[userIndex]
    # 문제셋에 해당하는 문제만 추출
    userDF = userDF[userDF['pId'].isin(categoryDF['pId'])]
    # erase duplications
    userDF.drop_duplicates(subset=['pId'], keep='last', inplace=True, ignore_index=True)
    recentSlovedProblems = list(userDF.head(100)['pId'])
    problemStr = ""
    for p in recentSlovedProblems:
        problemStr += p + ","
    problemStr = problemStr[:-1]
    apiUrl = 'https://solved.ac/api/v3/problem/lookup?problemIds='
    response = requests.get(apiUrl+problemStr).json()
    tagCnt = {}
    for p in response:
        tags = p['tags']
        for t in tags:
            tagName = t['displayNames'][0]['name']
            keys = tagCnt.keys()
            if tagName not in keys:
                tagCnt[tagName] = 1
            else:
                tagCnt[tagName] += 1
    tagCnt = dict(sorted(tagCnt.items(), key=lambda x: x[1], reverse=True))
    topTags = list(tagCnt.keys())[:10]
    reportData = []
    for t in topTags:
        dic = {'tag':t, 'Bronze':0, 'Silver':0, "Gold":0, 'Platinum':0}
        for p in response:
            level = p['level']
            tags = p['tags']
            for tag in tags:
                tagName = tag['displayNames'][0]['name']
                if tagName == t:
                    if level<6: dic['Bronze']+=1
                    elif level<11: dic['Silver']+=1
                    elif level<16: dic['Gold']+=1
                    elif level<21: dic['Platinum']+=1
        reportData.append(dic)
    return reportData

def valid_check(userId):
    url = "https://www.acmicpc.net/user/{}".format(userId)
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return response.status_code
    
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    snum = soup.select_one('#u-result-4')
    if snum is None:
        return -1
    return response.status_code
    
# 추천
@app.route("/recommend", methods=['POST'])
def recommend():
    userId = request.json['userId']
    statusCode = valid_check(userId)
    if(statusCode != 200):
        return jsonify({'status':statusCode})
    
    isUpdate = getUserData(userId)
    # if(isUpdate):
    #     data_preprocessing(userId)
    #     predict(model, userId)
    recDataA = getRecommend(userId, isUpdate, 'A')
    recDataB = getRecommend(userId, isUpdate, 'B')
    reportData = getReport(userId)
    result = {'status':statusCode, 'report':reportData, 'recommendA':recDataA, 'recommendB':recDataB}
    return jsonify(result)

# 사용자 인증
@app.route("/valid", methods=['GET'])
def validation():
    userId = request.args.get('userId')
    # 백준 사이트에서 사용자 검색해서 나오면 True, 에러 페이지로 가면 False
    response_code = valid_check(userId)
    
    print(response_code)
    return jsonify({"response_code":response_code})


if __name__ == '__main__':
    app.run()
