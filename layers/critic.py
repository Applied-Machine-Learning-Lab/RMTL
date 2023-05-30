import torch
from layers.layers import EmbeddingLayer,MultiLayerPerceptron


class Critic(torch.nn.Module):
    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims,
                 dropout):
        super(Critic, self).__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.action_layer = torch.nn.Linear(1, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + numerical_num + 1) * embed_dim
        self.bottom = MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False)
        self.tower = MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout)

    def forward(self, categorical_x, numerical_x, action):
        emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        action_emb = self.action_layer(action).unsqueeze(1)
        emb = torch.cat([emb, numerical_emb, action_emb], dim=1)
        emb = emb.view(emb.size(0), self.embed_output_dim)
        fea = self.bottom(emb)
        return self.tower(fea).squeeze(1)

class CriticNeg(torch.nn.Module):
    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims,
                 dropout):
        super(CriticNeg, self).__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.action_layer = torch.nn.Linear(1, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + numerical_num + 1) * embed_dim
        self.bottom = MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False)
        self.tower = MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout)
        self.activate = torch.nn.ReLU()

    def forward(self, categorical_x, numerical_x, action):
        emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        action_emb = self.action_layer(action).unsqueeze(1)
        emb = torch.cat([emb, numerical_emb, action_emb], dim=1)
        emb = emb.view(emb.size(0), self.embed_output_dim)
        fea = self.bottom(emb)
        return -self.activate(self.tower(fea)).squeeze(1)