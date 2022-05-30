import os
import pandas as pd
import random
import logging
import _pickle as cPickle
from datetime import datetime
import time

from recommenders.utils.constants import SEED

random.seed(SEED)
logger = logging.getLogger()

def data_preprocessing(
    dir,
    train_file,
    valid_file,
    test_file,
    user_vocab,
    item_vocab,
    cate_vocab,
    sample_rate=0.01,
    valid_num_ngs=4,
    test_num_ngs=9,
    is_history_expanding=True,
):

    instance_output = dir + 'instance_output'
    instance_output_train = dir + 'instance_output_train'
    instance_output_test = dir + 'instance_output_test'
    # instance_output_train = _create_instance(dir, 'train')
    # instance_output_test = _create_instance(dir, 'test')
    # print('create instance done')
    _create_item2cate(dir)
    preprocessed_output = _data_processing(instance_output_train)
    # preprocessed_output = dir + 'preprocessed_output'
    print('data processing done')
    if is_history_expanding:
        _data_generating(preprocessed_output, train_file, valid_file)
    print('data generating done')
    _create_test_data(instance_output_test, test_file)
    print('create test data done')
    _sampling_test(test_file)
    print("sampling test done")
    _create_vocab(train_file, user_vocab, item_vocab, cate_vocab)
    print('create vocab done')
    _negative_sampling_offline(
        instance_output, valid_file, test_file+'_0.01', valid_num_ngs, test_num_ngs
    )
    print('negative sampling offline done')

def _create_instance(dir, type):
    logger.info("start create instances...")
    output_file = dir + 'instance_output_'+ type

    # 사용자 리스트 가져오기
    userList = pd.read_csv(dir + type+'UserList.csv', index_col=0, dtype=str, encoding='utf-8')
    userList = userList['userId']

    # 문제 태그 데이터 가져오기
    categoryDF = pd.read_csv(dir + 'category.csv', index_col=0, dtype=str, encoding='utf-8')

    # 각 사용자 데이터에서 제출번호, 아이디, 문제 데이터만 추출하여 사용자가 푼 문제에 해당하는 태그 붙이기
    df = pd.DataFrame()
    for u in userList:
        file = dir + 'submits/' + u + '.json'
        userDF = pd.read_json(file, dtype = str, encoding='utf-8')
        if len(userDF) == 0: continue
        sbTimes = []
        for s in userDF['sbTime']:
            dt = datetime.strptime(s, '%Y년 %m월 %d일 %H:%M:%S')
            timestamp = int(time.mktime(dt.timetuple()))
            sbTimes.append(timestamp)
        userDF['sbTime'] = sbTimes
        userDF = userDF[['sbTime','uId','pId']]
        userDF = pd.merge(userDF,categoryDF,how='left',on='pId')
        df = pd.concat([df, userDF])
        print(u)

    # 라벨 붙이기
    df['label'] = [1 for i in range(len(df))]

    # '라벨', '아이디', '문제', '제출시간', '카테고리'순으로 열 바꾸기
    df = df[['label', 'uId', 'pId', 'sbTime', 'category']]
    
    # instance_output 저장 
    df.to_csv(output_file, sep = '\t', index = False, header = None)
    
    return output_file

# {문제:카테고리} 딕셔너리 생성
def _create_item2cate(dir):
    logger.info("creating item2cate dict")
    global item2cate
    categoryDF = pd.read_csv(dir + 'category.csv', index_col=0, dtype = str, encoding='utf-8')
    categoryDF.columns = ['item_id', 'cate_id']
    item2cate = categoryDF.set_index("item_id")['cate_id'].to_dict()

# 사용자가 푼 문제 개수-1번째까지 문제: train data
# 사용자가 마지막에 푼 문제 : valid data
def _data_processing(input_file):
    logger.info("start data processing...")
    dirs, _ = os.path.split(input_file)
    output_file = os.path.join(dirs, "preprocessed_output")

    f_input = open(input_file, "r")
    f_output = open(output_file, "w")
    user_count = {}
    for line in f_input:
        line = line.strip()
        user = line.split("\t")[1]
        if user not in user_count:
            user_count[user] = 0
        user_count[user] += 1
    f_input.seek(0)
    i = 0
    last_user = None
    for line in f_input:
        line = line.strip()
        user = line.split("\t")[1]
        if user == last_user:
            if i < user_count[user] - 1:
                f_output.write("train" + "\t" + line + "\n")
            else :
                f_output.write("valid" + "\t" + line + "\n")
        else:
            last_user = user
            i = 0
            if i < user_count[user] - 1:
                f_output.write("train" + "\t" + line + "\n")
            else :
                f_output.write("valid" + "\t" + line + "\n")
        i += 1
    return output_file

# train, valid 파일 분리하여 추천 모델 입력 형식에 맞는 데이터 생성
# 입력 형식 : label user_id item_id cate_id timestamp item_ids_history cate_ids_history timestamp_history
def _data_generating(input_file, train_file, valid_file, min_sequence=1):
    f_input = open(input_file, "r")
    f_train = open(train_file, "w")
    f_valid = open(valid_file, "w")
    logger.info("data generating...")
    last_user_id = None
    for line in f_input:
        line_split = line.strip().split("\t")
        tfile = line_split[0]
        label = int(line_split[1])
        user_id = line_split[2]
        item_id = line_split[3]
        date_time = line_split[4]
        category = line_split[5]

        if tfile == "train":
            fo = f_train
        elif tfile == "valid":
            fo = f_valid
        if user_id != last_user_id:
            item_id_list = []
            cate_list = []
            dt_list = []
        else:
            history_clk_num = len(item_id_list)
            cat_str = ""
            mid_str = ""
            dt_str = ""
            for c1 in cate_list:
                cat_str += c1 + ","
            for mid in item_id_list:
                mid_str += mid + ","
            for dt_time in dt_list:
                dt_str += dt_time + ","
            if len(cat_str) > 0:
                cat_str = cat_str[:-1]
            if len(mid_str) > 0:
                mid_str = mid_str[:-1]
            if len(dt_str) > 0:
                dt_str = dt_str[:-1]
            if history_clk_num >= min_sequence:
                fo.write(
                    line_split[1]
                    + "\t"
                    + user_id
                    + "\t"
                    + item_id
                    + "\t"
                    + category
                    + "\t"
                    + date_time
                    + "\t"
                    + mid_str
                    + "\t"
                    + cat_str
                    + "\t"
                    + dt_str
                    + "\n"
                )
        last_user_id = user_id
        if label:
            item_id_list.append(item_id)
            cate_list.append(category)
            dt_list.append(date_time)

# 전체 사용자의 0.1%를 랜덤으로 뽑아 테스트 사용자로 사용  
# 테스트 사용자에 해당하는 데이터들을 테스트 데이터로 저장
# train,valid 데이터에서 테스트 사용자에 해당하는 데이터 삭제
def _create_test_data(input_file, test_file, min_sequence=1):
    f_input = open(input_file, "r")
    fo = open(test_file, "w")
    logger.info("data generating...")
    last_user_id = None
    for line in f_input:
        line_split = line.strip().split("\t")
        label = int(line_split[0])
        user_id = line_split[1]
        item_id = line_split[2]
        date_time = line_split[3]
        category = line_split[4]

        if user_id != last_user_id:
            item_id_list = []
            cate_list = []
            dt_list = []
        else:
            history_clk_num = len(item_id_list)
            cat_str = ""
            mid_str = ""
            dt_str = ""
            for c1 in cate_list:
                cat_str += c1 + ","
            for mid in item_id_list:
                mid_str += mid + ","
            for dt_time in dt_list:
                dt_str += dt_time + ","
            if len(cat_str) > 0:
                cat_str = cat_str[:-1]
            if len(mid_str) > 0:
                mid_str = mid_str[:-1]
            if len(dt_str) > 0:
                dt_str = dt_str[:-1]
            if history_clk_num >= min_sequence:
                fo.write(
                    line_split[0]
                    + "\t"
                    + user_id
                    + "\t"
                    + item_id
                    + "\t"
                    + category
                    + "\t"
                    + date_time
                    + "\t"
                    + mid_str
                    + "\t"
                    + cat_str
                    + "\t"
                    + dt_str
                    + "\n"
                )

        last_user_id = user_id
        if label:
            item_id_list.append(item_id)
            cate_list.append(category)
            dt_list.append(date_time)
    fo.close()

def _sampling_test(test_file, sample_rate=0.01):
    f_test = pd.read_csv(test_file, sep='\t', dtype=str, header=None, names=['label', 'uId', 'pId', 'cate', 'time', 'phis', 'chis', 'this'], encoding='ansi')
    users = f_test['uId'].unique()
    sample_user = set()
    count = 0
    user_num = len(users)
    while count<int(user_num * sample_rate):
        random_user = random.choice(users)
        if random_user not in sample_user:
            sample_user.add(random_user)
            count += 1
    test_df = f_test[f_test['uId'].isin(sample_user)]
    output = test_file + "_" + str(sample_rate)
    test_df.to_csv(output, index=None, header=None, sep = '\t', encoding='ansi')

# train 데이터에 존재하는 모든 사용자, 문제, 카테고리에 대한 vocab생성
# user_vocab = {사용자아이디:개수}
# item_vocab = {문제:개수}
# cate_vocab = {카테고리:개수}
def _create_vocab(train_file, user_vocab, item_vocab, cate_vocab):
    f_train = open(train_file, "r")

    user_dict = {}
    item_dict = {}
    cat_dict = {}

    logger.info("vocab generating...")
    for line in f_train:
        arr = line.strip("\n").split("\t")
        uid = arr[1]
        mid = arr[2]
        cat = arr[3]
        mid_list = arr[5]
        cat_list = arr[6]

        if uid not in user_dict:
            user_dict[uid] = 0
        user_dict[uid] += 1
        if mid not in item_dict:
            item_dict[mid] = 0
        item_dict[mid] += 1
        if cat not in cat_dict:
            cat_dict[cat] = 0
        cat_dict[cat] += 1
        if len(mid_list) == 0:
            continue
        for m in mid_list.split(","):
            if m not in item_dict:
                item_dict[m] = 0
            item_dict[m] += 1
        for c in cat_list.split(","):
            if c not in cat_dict:
                cat_dict[c] = 0
            cat_dict[c] += 1

    sorted_user_dict = sorted(user_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_item_dict = sorted(item_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_cat_dict = sorted(cat_dict.items(), key=lambda x: x[1], reverse=True)

    uid_voc = {}
    index = 0
    for key, value in sorted_user_dict:
        uid_voc[key] = index
        index += 1

    mid_voc = {}
    mid_voc["default_mid"] = 0
    index = 1
    for key, value in sorted_item_dict:
        mid_voc[key] = index
        index += 1

    cat_voc = {}
    cat_voc["default_cat"] = 0
    index = 1
    for key, value in sorted_cat_dict:
        cat_voc[key] = index
        index += 1

    cPickle.dump(uid_voc, open(user_vocab, "wb"))
    cPickle.dump(mid_voc, open(item_vocab, "wb"))
    cPickle.dump(cat_voc, open(cate_vocab, "wb"))

# valid와 test데이터에 negative 데이터 추가
def _negative_sampling_offline(
    instance_input_file, valid_file, test_file, valid_neg_nums=4, test_neg_nums=49
):
    columns = ["label", "user_id", "item_id", "timestamp", "cate_id"]
    ns_df = pd.read_csv(instance_input_file, sep="\t", names=columns, dtype=str, encoding='ansi')
    items_with_popular = list(ns_df["item_id"])

    global item2cate

    # valid negative sampling
    logger.info("start valid negative sampling")
    with open(valid_file, "r") as f:
        valid_lines = f.readlines()
    write_valid = open(valid_file, "w")
    for line in valid_lines:
        write_valid.write(line)
        words = line.strip().split("\t")
        positive_item = words[2]
        count = 0
        neg_items = set()
        while count < valid_neg_nums:
            neg_item = random.choice(items_with_popular)
            if neg_item == positive_item or neg_item in neg_items:
                continue
            count += 1
            neg_items.add(neg_item)
            words[0] = "0"
            words[2] = neg_item
            words[3] = item2cate[neg_item]
            write_valid.write("\t".join(words) + "\n")

    # test negative sampling
    logger.info("start test negative sampling")
    with open(test_file, "r") as f:
        test_lines = f.readlines()
    write_test = open(test_file, "w")
    for line in test_lines:
        write_test.write(line)
        words = line.strip().split("\t")
        positive_item = words[2]
        count = 0
        neg_items = set()
        while count < test_neg_nums:
            neg_item = random.choice(items_with_popular)
            if neg_item == positive_item or neg_item in neg_items:
                continue
            count += 1
            neg_items.add(neg_item)
            words[0] = "0"
            words[2] = neg_item
            words[3] = item2cate[neg_item]
            write_test.write("\t".join(words) + "\n")

dir = "./resources/"
data_preprocessing(dir, dir + "train_data", dir + "valid_data", dir + "test_data", dir + "user_vocab.pkl" , dir + "item_vocab.pkl", dir + "cate_vocab.pkl")


