import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

def dl_data_load(args):
    """
    Parameters
    ----------
    Args:
        data_path : str
            데이터 경로
    ----------
    """

    ######################## DATA LOAD
    users = pd.read_csv(args.data_path + 'users.csv')
    books = pd.read_csv(args.data_path + 'books.csv')
    
    train = pd.read_csv(args.data_path + 'train_ratings.csv')

    ######################### DATA remove
    # 데이터 전처리, user_id 및 isbn이 1개 씩만 있는 데이터 제거 후 처리
    
    # user_id_counts = train['user_id'].value_counts()
    # isbn_counts = train['isbn'].value_counts()
    # train_one = train[(train['user_id'].map(user_id_counts) == 1) & (train['isbn'].map(isbn_counts) == 1)]
    # train = train.drop(train_one.index)

    ######################### DATA Agumentation

    # ratings = pd.read_csv(args.data_path + 'train_ratings.csv')
    # # 대소문자 제거
    # books['book_author'] = books['book_author'].str.lower().str.replace(' ', '')
    # books['book_author'] = books['book_author'].str.replace('\s+', ' ', regex=True).str.strip()

    # # 결측치 우선 np.nan으로 제거
    # books = books.replace('notapplicable(na)', np.nan)
    # books = books.replace('', np.nan)  

    # # merge 후 원본 데이터

    # merge1 = ratings.merge(books, how='left', on='isbn')
    # data = merge1.merge(users, how='inner', on='user_id')[['user_id', 'isbn', 'rating', 'book_title' ,'book_author']]
    # # count
    # user_id_counts = data['user_id'].value_counts()
    # isbn_counts = data['isbn'].value_counts()
    # cold_df = data[(data['user_id'].map(user_id_counts) == 1) & (data['isbn'].map(isbn_counts) == 1)]
    # cold_df = cold_df[['user_id', 'isbn', 'rating', 'book_title' ,'book_author']]

    # filtered_df = cold_df.copy()

    # # 필터된 데이터프레임의 복사본 생성
    # augmented_df = filtered_df.copy()
    # # filtered_df의 각 저자에 대한 평점 추출
    # author_ratings = filtered_df.set_index('book_author')['rating']
    # # filtered_df의 각 행에 대해 반복
    # for index, row in filtered_df.iterrows():
    #     try:
    #         # 현재 행의 작가
    #         author = row['book_author']

    #         # 동일 작가의 다른 책 찾기 (현재 행의 책 제외), 최대 3개 추가 
    #         other_books = data[(data['book_author'] == author) & (data['isbn'] != row['isbn'])].head(1)

    #         # 찾은 책들의 평점을 현재 행의 평점으로 
    #         other_books['rating'] = author_ratings[author]
            
    #         # 찾은 책들의 평점은 현재 행의 유저 
    #         other_books['user_id'] = row['user_id']
            
    #         # 찾은 책들을 extended_df에 추가
    #         augmented_df = augmented_df.append(other_books, ignore_index=True)
    #     except:
    #         pass

    # # 중복 제거 (동일한 저자의 같은 책을 여러 번 추가하는 것을 방지)
    # augmented_df = augmented_df.drop_duplicates(subset=['user_id', 'isbn', 'book_title', 'book_author'])
    # augmented_df = augmented_df.sort_values(by='user_id')       # 정렬 처리
    # augmented_df = augmented_df.reset_index(drop=True)          # 인덱스 리셋
    # augmented_df = augmented_df[['user_id', 'isbn', 'rating']]  # ratings 컬럼 맞춤 

    # train = pd.concat([ratings, augmented_df]).drop_duplicates().reset_index(drop=True)


    print('size : ',len(train))

    test = pd.read_csv(args.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.data_path + 'sample_submission.csv')

    # 훈련 데이터와 제출용 샘플 데이터에서 user_id 및 isbn concat 후 unique() 만 추출
    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    # 각 식별자에 대해 정수 인덱스를 매핑하는 딕셔너리 생성 {idx : 식별자}
    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    # 딕셔너리 매핑후, 역(reverse)으로 생성 {식별자 : idx}
    user2idx = {id:idx for idx, id in idx2user.items()}         # {638: 0, ... }
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}     # {'0385505833': 0,...}

    # id들을 정수 인덱스로 변환, 테스트 데이터 포함
    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)

    #-------- id들을 정수 인덱싱 
    # 필드의 크기를 정의
    field_dims = np.array([len(user2idx), len(isbn2idx)], dtype=np.uint32)

    data = {
            'train':train,
            'test':test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data

#----
def dl_data_split(args, data):
    """
    Parameters
    ----------
    Args:
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            랜덤 seed 값
    ----------
    """

    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.test_size,
                                                        random_state=args.seed,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

# ---- 배치사이즈, 셔플 여부
def dl_data_loader(args, data):
    """
    Parameters
    ----------
    Args:
        batch_size : int
            데이터 batch에 사용할 데이터 사이즈
        data_shuffle : bool
            data shuffle 여부
    ----------
    """

    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
