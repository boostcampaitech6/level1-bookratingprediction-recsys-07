import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from tqdm import tqdm


def user_preprocess(users):
    users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '', regex=True) # 특수문자 제거

    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0].strip())
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1].strip())
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2].strip())

    users = users.replace('na', np.nan)
    users = users.replace('', np.nan)


    modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values
    location = users[(users['location'].str.contains('seattle'))&(users['location_country'].notnull())]['location'].value_counts().index[0]

    location_list = []
    for location in modify_location:
        try:
            right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
            location_list.append(right_location)
        except:
            pass

    for location in location_list:
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]

    
    users = users.drop(['location'], axis=1)
    return users

def book_preprocess(books):
    publisher_dict=(books['publisher'].value_counts()).to_dict()
    publisher_count_df= pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])

    publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)

    modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values
    for publisher in modify_list:
        try:
            number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
            right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
            books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
        except:
            pass
    return books

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6

def process_context_data(users, books, ratings1, ratings2, args):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    ----------
    """

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')

    # user  파트 인덱싱 처리
    idx = []

    for i, feature in enumerate(args.user_features):
        idx.append({v:k for k,v in enumerate(context_df[feature].unique())})
        train_df[feature] = train_df[feature].map(idx[i])
        test_df[feature] = test_df[feature].map(idx[i])


    # book 파트 인덱싱 처리
    for j, feature in enumerate(args.item_features):
        idx.append({v:k for k,v in enumerate(context_df[feature].unique())})
        train_df[feature] = train_df[feature].map(idx[-1])
        test_df[feature] = test_df[feature].map(idx[-1])

    return idx, train_df, test_df

class Image_Dataset(Dataset):
    def __init__(self, user_isbn_vector, img_vector, label):
        """
        Parameters
        ----------
        user_isbn_vector : np.ndarray
            벡터화된 유저와 책 데이터를 입렵합니다.
        img_vector : np.ndarray
            벡터화된 이미지 데이터를 입력합니다.
        label : np.ndarray
            정답 데이터를 입력합니다.
        ----------
        """
        self.user_isbn_vector = user_isbn_vector
        self.img_vector = img_vector
        self.label = label
    def __len__(self):
        return self.user_isbn_vector.shape[0]
    def __getitem__(self, i):
        return {
                'user_isbn_vector' : torch.tensor(self.user_isbn_vector[i], dtype=torch.long),
                'img_vector' : torch.tensor(self.img_vector[i], dtype=torch.float32),
                'label' : torch.tensor(self.label[i], dtype=torch.float32),
                }


def image_vector(path):
    """
    Parameters
    ----------
    path : str
        이미지가 존재하는 경로를 입력합니다.
    ----------
    """
    img = Image.open(path)
    scale = transforms.Resize((32, 32))
    tensor = transforms.ToTensor()
    img_fe = Variable(tensor(scale(img)))
    return img_fe


def process_img_data(df, books, user2idx, isbn2idx, train=False):
    """
    Parameters
    ----------
    df : pd.DataFrame
        기준이 되는 데이터 프레임을 입력합니다.
    books : pd.DataFrame
        책 정보에 대한 데이터 프레임을 입력합니다.
    user2idx : Dict
        각 유저에 대한 index 정보가 있는 사전을 입력합니다.
    isbn2idx : Dict
        각 책에 대한 index 정보가 있는 사전을 입력합니다.
    ----------
    """
    books_ = books.copy()
    #books_['isbn'] = books_['isbn'].map(isbn2idx)

    df_ = df.copy()

    df_ = pd.merge(df_, books_[['isbn', 'img_path']], on='isbn', how='left')
    df_['img_path'] = df_['img_path'].apply(lambda x: 'data/'+x)
    img_vector_df = df_[['img_path']].drop_duplicates().reset_index(drop=True).copy()
    data_box = []
    for idx, path in tqdm(enumerate(sorted(img_vector_df['img_path']))):
        data = image_vector(path)
        if data.size()[0] == 3:
            data_box.append(np.array(data))
        else:
            data_box.append(np.array(data.expand(3, data.size()[1], data.size()[2])))
    img_vector_df['img_vector'] = data_box
    df_ = pd.merge(df_, img_vector_df, on='img_path', how='left')
    return df_


def context_image_data_load(args):
    """
    Parameters
    ----------
    Args : argparse.ArgumentParser
        data_path : str
            데이터가 존재하는 경로를 입력합니다.
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            seed 값을 입력합니다.
        batch_size : int
            Batch size를 입력합니다.
    ----------
    """

    users = pd.read_csv(args.data_path + 'users.csv')
    books = pd.read_csv(args.data_path + 'books.csv')
    train = pd.read_csv(args.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.data_path + 'sample_submission.csv')

    print('preprocessing start')
    users = user_preprocess(users)
    print('user preprocess done')
    books = book_preprocess(books)
    print('book preprocess done')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    idx, context_train, context_test = process_context_data(users, books, train, test, args)
    field_dims = np.array([len(user2idx), len(isbn2idx)] +
                            [len(dict_map) for dict_map in idx], dtype=np.uint32)
    img_train = process_img_data(context_train, books, user2idx, isbn2idx, train=True)
    img_test = process_img_data(context_test, books, user2idx, isbn2idx, train=False)

    data = {
            'train':context_train,
            'test':context_test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            'img_train':img_train,
            'img_test':img_test,
            }
    return data


def context_image_data_split(args, data):
    """
    Parameters
    ----------
    Args : argparse.ArgumentParser
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            seed 값을 입력합니다.
    data : Dict
        image_data_load로 부터 전처리가 끝난 데이터가 담긴 사전 형식의 데이터를 입력합니다.
    ----------
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['img_train'][['user_id', 'isbn'] + args.user_features + args.item_features + ['img_vector']],
                                                        data['img_train']['rating'],
                                                        test_size=args.test_size,
                                                        random_state=args.seed,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data


def context_image_data_loader(args, data):
    """
    Parameters
    ----------
    Args : argparse.ArgumentParser
        batch_size : int
            Batch size를 입력합니다.
    data : Dict
        image_data_split로 부터 학습/평가/실험 데이터가 담긴 사전 형식의 데이터를 입력합니다.
    ----------
    """
    train_dataset = Image_Dataset(
                                data['X_train'][['user_id', 'isbn'] + args.user_features + args.item_features].values,
                                data['X_train']['img_vector'].values,
                                data['y_train'].values
                                )
    valid_dataset = Image_Dataset(
                                data['X_valid'][['user_id', 'isbn'] + args.user_features + args.item_features].values,
                                data['X_valid']['img_vector'].values,
                                data['y_valid'].values
                                )
    test_dataset = Image_Dataset(
                                data['img_test'][['user_id', 'isbn'] + args.user_features + args.item_features].values,
                                data['img_test']['img_vector'].values,
                                data['img_test']['rating'].values
                                )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)
    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader
    return data
