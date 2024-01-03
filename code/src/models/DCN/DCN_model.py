import numpy as np
import torch
import torch.nn as nn


# factorization을 통해 얻은 feature를 embedding 합니다.

# embed_dim 임베딩 차원 입력으로 받아서 처리
# 만약 embed_dim이 4면,

# tensor([[[ 0.0135, -0.0116,  0.0113,  0.0105],
#          [ 0.0167,  0.0074,  0.0202, -0.0185]]]
# 이런식으로 각 feature을 4차원으로 임베딩 


# 입력 : 범주형 필드의 크기(field_dims)와 임베딩 차원(embed_dim)
# field_dims 2개의 범주형 데이터 user_id는 59,803개의 카테고리, isbn은 129,777개의 카테고리 


class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


# Cross Network : 입력 피처 간의 교차(cross) 상호 작용(interaction)을 캡처하는 데 사용되는 신경망 아키텍처 
# input_dim : 임베딩된 특성의 차원
# num_layers : 크로스 네트워크의 레이어 수 
'''
교차 상호 작용(x0 * xw)과 편향(self.b[i])을 더하고, 
이전 레이어의 출력 x를 더하여 현재 레이어의 출력값을 얻습니다. 
이것이 교차 상호 작용을 포함한 현재 레이어의 출력입니다.
따라서 각 레이어에서 입력 피처 간의 교차 상호 작용이 모델링됨   
'''

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


# DCN 모델은 MLP와 CrossNetwork를 합하여 최종 결과를 도출합니다.
# MLP을 구현
# 비선형성을 추가하여 복잡한 함수를 모델링할 수 있으며, 배치 정규화와 드롭아웃을 통해 일반화 능력을 향상   
# ! MLP 은닉층 내부를 수정하여 성능향상, 
class MultiLayerPerceptron(nn.Module):
    
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim)) # 선형층
            layers.append(torch.nn.BatchNorm1d(embed_dim))       # 배치 정규화
            layers.append(torch.nn.ELU())                        # ELU 활성화 함수
            layers.append(torch.nn.Dropout(p=dropout))           # 드롭아웃
            input_dim = embed_dim
    
        # 마지막에 선형 레이어를 추가하여 출력
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    


## Crossnetwork 결과를 MLP layer에 넣어 최종결과를 도출합니다.
# 입력 특성 간의 interaction 을 학습하는 Cross Network와 
# 비선형성을 추가하는 MLP를 결합하여 다양한 특성을 효과적으로 모델링

# ParallelDeepCrossNetworkModel
# RMSE, ADAM
    
class DeepCrossNetworkModel(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.embed_output_dim = len(self.field_dims) * args.embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, args.num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, args.mlp_dims, args.dropout, output_layer=False)
        self.linear = torch.nn.Linear(args.mlp_dims[-1] + self.embed_output_dim, 1)


    def forward(self, x: torch.Tensor):
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        h_l2 = self.mlp(embed_x)
        x_stack = torch.cat([x_l1, h_l2], dim=1)

        # concat 이후 선형레이어 통과
        p = self.linear(x_stack) 
        
        # 클램핑 처리, 모델단에서 처리되므로, 훈련중에도 적용 x
        # p = torch.clamp(self.linear(x_stack), 1.0, 10.0)

        return p.squeeze(1) 



# StackedDeepCrossNetworkModel
    
# class DeepCrossNetworkModel(nn.Module):
#     def __init__(self, args, data):
        
#         super().__init__()

#         self.field_dims = data['field_dims']
#         self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
#         self.embed_output_dim = len(self.field_dims) * args.embed_dim
#         self.cn = CrossNetwork(self.embed_output_dim, args.num_layers)
#         self.mlp = MultiLayerPerceptron(self.embed_output_dim, args.mlp_dims, args.dropout, output_layer=False)
#         self.cd_linear = nn.Linear(args.mlp_dims[0], 1, bias=False)


#     def forward(self, x: torch.Tensor):
#         embed_x = self.embedding(x).view(-1, self.embed_output_dim)
#         x_l1 = self.cn(embed_x)
#         x_out = self.mlp(x_l1)

#         p = self.cd_linear(x_out) # 임베딩 레이어 부터 순차적으로 구성되어, 레이어 출력이 다음 레이어의 입력이 됩니다.
#         return p.squeeze(1)
