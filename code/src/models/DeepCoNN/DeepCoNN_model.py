import numpy as np
import torch
import torch.nn as nn


# feature 사이의 상호작용을 효율적으로 계산합니다.
class FactorizationMachine(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.v = nn.Parameter(torch.rand(input_dim, latent_dim), requires_grad = True)
        self.linear = nn.Linear(input_dim, 1, bias=True)


    def forward(self, x):
        linear = self.linear(x)
        square_of_sum = torch.mm(x, self.v) ** 2
        sum_of_square = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        output = linear + (0.5 * pair_interactions)
        return output


# factorization을 통해 얻은 feature를 embedding 합니다.
class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


# 텍스트 특징 추출을 위한 기초적인 CNN 1D Layer를 정의합니다.

'''
class CNN_1D(nn.Module):
    def __init__(self, word_dim, out_dim, kernel_size, conv_1d_out_dim):
        super(CNN_1D, self).__init__()
        self.conv = nn.Sequential(
                                nn.Conv1d(
                                        in_channels=word_dim,
                                        out_channels=out_dim,
                                        kernel_size=kernel_size,
                                        padding=(kernel_size - 1) // 2),
                                nn.BatchNorm1d(out_dim),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=(kernel_size, 1)),
                                nn.Dropout(p=0.5)
                                )
        self.linear = nn.Sequential(
                                    nn.Linear(int(out_dim/kernel_size), conv_1d_out_dim),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5))


    def forward(self, vec):
        output = self.conv(vec)
        output = self.linear(output.reshape(-1, output.size(1)))
        return output
'''
class LSTM_1D(nn.Module):
    def __init__(self, word_dim, out_dim, lstm_1d_out_dim, num_layers=20):
        super(LSTM_1D, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_size=word_dim,
                            hidden_size=out_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.5)
        # Linear layer
        self.linear = nn.Sequential(
            nn.Linear(out_dim, lstm_1d_out_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

    def forward(self, vec):
        # LSTM layer
        lstm_out, _ = self.lstm(vec)
        # Taking the last hidden state
        last_hidden_state = lstm_out[:, -1, :]
        # Linear layer
        output = self.linear(last_hidden_state)
        return output

class CrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])


    def forward(self, x: torch.Tensor):
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)


    def forward(self, x):
        return self.mlp(x)
    
# 기존 유저/상품 벡터와 유저/상품 리뷰 벡터를 결합하여 FM으로 학습하는 모델을 구현합니다.
class DeepCoNN(nn.Module):
    def __init__(self, args, data):
        super(DeepCoNN, self).__init__()
        self.field_dims = np.array([len(data['user2idx']), len(data['isbn2idx'])], dtype=np.uint32)
        self.embedding = FeaturesEmbedding(self.field_dims, args.deepconn_embed_dim)
        self.cnn_u = LSTM_1D(
                             word_dim=args.word_dim,
                             out_dim=args.out_dim,
                             #kernel_size=args.kernel_size,
                             lstm_1d_out_dim=args.conv_1d_out_dim,
                            )
        self.cnn_i = LSTM_1D(
                             word_dim=args.word_dim,
                             out_dim=args.out_dim,
                             #kernel_size=args.kernel_size,
                             lstm_1d_out_dim=args.conv_1d_out_dim,
                            )
        self.fm = FactorizationMachine(
                                        input_dim=(args.conv_1d_out_dim * 2) + (args.deepconn_embed_dim*len(self.field_dims)),
                                        latent_dim=args.deepconn_latent_dim,
                                        )
        self.mlp = MultiLayerPerceptron(
            input_dim=(args.conv_1d_out_dim * 2) + (args.deepconn_embed_dim*len(self.field_dims)),
            embed_dims=args.mlp_dims,
            dropout=args.dropout,
            output_layer=True
        )

    def forward(self, x):
        user_isbn_vector, user_text_vector, item_text_vector = x[0], x[1], x[2]
        user_isbn_feature = self.embedding(user_isbn_vector)
        user_text_feature = self.cnn_u(user_text_vector.permute(0, 2, 1))
        item_text_feature = self.cnn_i(item_text_vector.permute(0, 2, 1))
        feature_vector = torch.cat([
                                    user_isbn_feature.view(-1, user_isbn_feature.size(1) * user_isbn_feature.size(2)),
                                    user_text_feature,
                                    item_text_feature
                                    ], dim=1)
        output = self.fm(feature_vector) + self.mlp(feature_vector)
        return output.squeeze(1) 

'''
class DeepCoNN(nn.Module):
    def __init__(self, args, data):
        super(DeepCoNN, self).__init__()
        self.field_dims = np.array([len(data['user2idx']), len(data['isbn2idx'])], dtype=np.uint32)
        self.embedding = FeaturesEmbedding(self.field_dims, args.deepconn_embed_dim)
        self.cnn_u = CNN_1D(
                             word_dim=args.word_dim,
                             out_dim=args.out_dim,
                             kernel_size=args.kernel_size,
                             conv_1d_out_dim=args.conv_1d_out_dim,
                            )
        self.cnn_i = CNN_1D(
                             word_dim=args.word_dim,
                             out_dim=args.out_dim,
                             kernel_size=args.kernel_size,
                             conv_1d_out_dim=args.conv_1d_out_dim,
                            )
        self.fm = FactorizationMachine(
                                        input_dim=(args.conv_1d_out_dim * 2) + (args.deepconn_embed_dim*len(self.field_dims)),
                                        latent_dim=args.deepconn_latent_dim,
                                        )


    def forward(self, x):
        user_isbn_vector, user_text_vector, item_text_vector = x[0], x[1], x[2]
        user_isbn_feature = self.embedding(user_isbn_vector)
        user_text_feature = self.cnn_u(user_text_vector)
        item_text_feature = self.cnn_i(item_text_vector)
        feature_vector = torch.cat([
                                    user_isbn_feature.view(-1, user_isbn_feature.size(1) * user_isbn_feature.size(2)),
                                    user_text_feature,
                                    item_text_feature
                                    ], dim=1)
        output = self.fm(feature_vector)
        return output.squeeze(1)
'''