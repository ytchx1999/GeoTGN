import torch
import numpy as np
import math
# from geopy.distance import geodesic


class TimeEncode(torch.nn.Module):
    # Time Encoding proposed by TGAT
    def __init__(self, dimension):
        super(TimeEncode, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        # t has shape [batch_size, seq_len]
        # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
        t = t.unsqueeze(dim=2)

        # output has shape [batch_size, seq_len, dimension]
        output = torch.cos(self.w(t))

        return output


class LocalDist(torch.nn.Module):
    # Time Encoding proposed by TGAT
    def __init__(self, max_dist=100):
        super(LocalDist, self).__init__()
        self.max_dist = max_dist

        # self.dimension = dimension
        # self.w = torch.nn.Linear(1, dimension)

        # self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
        #                                    .float().reshape(dimension, -1))
        # self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())
    
    def haversine(self, lat1, lon1, lat2, lon2):
        # distance between latitudes
        # and longitudes
        dLat = (lat2 - lat1) * math.pi / 180.0
        dLon = (lon2 - lon1) * math.pi / 180.0
    
        # convert to radians
        lat1 = (lat1) * math.pi / 180.0
        lat2 = (lat2) * math.pi / 180.0
    
        # apply formulae
        a = (torch.pow(torch.sin(dLat / 2), 2) +
            torch.pow(torch.sin(dLon / 2), 2) *
                torch.cos(lat1) * torch.cos(lat2))
        rad = 6371
        c = 2 * torch.asin(torch.sqrt(a))
        return rad * c

    def forward(self, lat1, lon1, lat2, lon2):
        # t has shape [batch_size, seq_len]
        # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
        # t = t.unsqueeze(dim=2)

        # output has shape [batch_size, seq_len, dimension]
        # max_dist = 20.  # 100
        output = (self.haversine(lat1, lon1, lat2, lon2))
        mask_max = (output >= self.max_dist)
        output[mask_max] = self.max_dist
        output = output.unsqueeze(dim=2)

        return output


class PosEncode(torch.nn.Module):
    def __init__(self, dimension, seq_len):
        super().__init__()
        
        self.dimension = dimension
        self.pos_embeddings = torch.nn.Embedding(num_embeddings=seq_len, embedding_dim=dimension)
        
    def forward(self, ts):
        # ts: [N, L]
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb
    

class EmptyEncode(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        
    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.dimension)
        return out
