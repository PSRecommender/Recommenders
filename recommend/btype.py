import pandas as pd
import json
from path import path

def toJson(dir, dict):
    with open(dir, 'w', encoding='utf-8') as file:
        json.dump(dict, file, ensure_ascii=False, indent='\t')

def loadJson(dir):
    with open(dir, 'r', encoding='utf-8') as file:
        return json.load(file)

def recommendForB(userId):
    dir = path
    categoryDF = pd.read_csv(dir + 'category.csv', index_col=0, dtype=str, encoding='utf-8')
    file = dir + 'data/{userId}/{userId}.json'.format(userId=userId)
    userDF = pd.read_json(file, dtype = str, encoding='utf-8')
    # 맞은 문제만 추출
    userDF = userDF[userDF['result']=='맞았습니다!!']
    # 문제셋에 해당하는 문제만 추출
    userDF = userDF[userDF['pId'].isin(categoryDF['pId'])]
    # 중복 제거
    userDF.drop_duplicates(subset=['pId'], keep='last', inplace=True, ignore_index=True)
    userDF = userDF[['pId']]
    # pId dtype int형으로 변환
    userDF['pId'] = pd.to_numeric(userDF['pId'])
    # 문제 행렬
    problemDF = pd.read_csv(dir+"problems.csv")
    col = ['pId'] + list(problemDF.columns)[1:]
    problemDF.columns = col
    problemDF.sort_values(by='level', ascending=False, inplace=True)
    # 태그
    tags = problemDF.columns[1:-1]
    # 주요태그
    mainTags = ["Implementation", "DynamicProgramming", "DataStructures", "String", "Greedy", "Bruteforcing", "Sorting", "BinarySearch", "Breadth_firstSearch", "Depth_firstSearch", "Dijkstra's", "SetMapByHashing", "PriorityQueue"]
    # 사용자가 푼 문제
    solvedProblemDF = pd.merge(userDF, problemDF, how='left', on='pId')
    solvedIndex = solvedProblemDF.index
    solvedProblemsByTag = {}
    for t in mainTags:
        solvedProblemsByTag[t] = []
    for i in solvedIndex:
        problem = solvedProblemDF.loc[i]
        for t in mainTags:
            if problem[t] == 1:
                solvedProblemsByTag[t].append(problem['pId'])
    # 태그 별 레벨 구하기
    levelsByTag = {}
    for t in mainTags:
        df = solvedProblemDF[solvedProblemDF['pId'].isin(solvedProblemsByTag[t])]
        levels = df['level'].value_counts()
        levelIndex = list(levels.keys())
        levelIndex.sort(reverse=True)
        levelSum = 0
        levelCnt = 0
        bronze, silver, gold, platinum = 0,0,0,0
        for i in levelIndex:
            if i<6: bronze += levels[i]
            elif i<11: silver += levels[i]
            elif i<16: gold += levels[i]
            else: platinum += levels[i]

            if levelCnt < 20:
                if levelCnt + levels[i] <= 20:
                    levelCnt += levels[i]
                    levelSum += i * levels[i]
                else:
                    levelSum += i * (20 - levelCnt)
                    levelCnt = 20
        if levelCnt == 0:
            levelAvg = 0
        else:
            levelAvg = levelSum/levelCnt
        total = bronze + silver + gold + platinum
        levelsByTag[t] = {'avgLevel':levelAvg, 'bronze':bronze, 'silver':silver, 'gold':gold, 'platinum':platinum, 'total':total}
    
    # 사용자가 안 푼 문제 구하기
    notSolvedProblems = []
    solvedProblems = list(solvedProblemDF['pId'])
    totalProblems = list(problemDF['pId'])
    for p in totalProblems:
        if p not in solvedProblems:
            notSolvedProblems.append(p)
    # 사용자가 안 푼 문제
    notSolvedProblemDF = problemDF[problemDF['pId'].isin(notSolvedProblems)]
    notSolvedIndex = notSolvedProblemDF.index
    notSolvedProblemsByTag = {}
    for t in mainTags:
        notSolvedProblemsByTag[t] = []
    for i in notSolvedIndex:
        problem = notSolvedProblemDF.loc[i]
        for t in mainTags:
            if problem[t] == 1:
                notSolvedProblemsByTag[t].append(problem['pId'])
    
    # 태그 별 추천 문제
    recommendByTag = {}
    for t in mainTags:
        df = notSolvedProblemDF[notSolvedProblemDF['pId'].isin(notSolvedProblemsByTag[t])]
        avgLevel = levelsByTag[t]['avgLevel']
        if avgLevel == 0:
            avgLevel = min(df['level'])
        index = df.index
        pIds = []
        for i in index:
            level = df.loc[i]['level']
            if level>=avgLevel-2 and level<=avgLevel+2:
                pIds.append(df.loc[i]['pId'])
        recommendByTag[t] = pIds

    # 문제 정보
    problems = pd.read_json('problemData.json')
    problems = problems[['pId', 'title', 'successCnt']]

    # 추천 문제 뽑기(태그 별 맞은 사람 많은 순으로)
    recommendProblems = []
    for t in mainTags:
        problemList = recommendByTag[t]
        df = problems[problems['pId'].isin(problemList)].sort_values('successCnt', ascending=False, ignore_index=True)
        cnt = 0
        rec = df.loc[cnt]['pId']
        dfLen = len(df)
        while rec in recommendProblems and cnt < dfLen:
            cnt+=1
            rec = df.loc[cnt]['pId']
        if rec not in recommendProblems: recommendProblems.append(rec)
        while rec in recommendProblems and cnt < dfLen:
            cnt+=1
            rec = df.loc[cnt]['pId']
        if rec not in recommendProblems: recommendProblems.append(rec)
    problems = loadJson(path+'problemData.json')
    recDict = []
    for r in recommendProblems:
        for p in problems:
            if int(r) == p['pId']:
                recDict.append(p)
    return recDict
    

