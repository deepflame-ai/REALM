import numpy as np
from torch.utils.data import IterableDataset, Dataset
from torch_geometric import transforms as T
from torch_geometric.data import Data
 
    
class ODEGraphDataset(IterableDataset):
    def __init__(self, data, coords):
        super().__init__()
        self.data = data  # (num_tras, num_steps, num_feats, num_nodes)
        self.coords = coords.squeeze(0).permute(1, 0)  # (num_nodes, 2)
        self.tra_keys = list(range(data.shape[0]))
        self.shuffle_keys()
        self.num_tra, self.tra_len = data.shape[0], data.shape[1]
        self.open_tra_num = min(10, self.num_tra)

        self.tra_index = 0
        self.opened_tra = []
        self.opened_tra_readed_index = {}
        self.opened_tra_readed_random_index = {}

        self.transformer = T.Compose([
            T.KNNGraph(k=8), T.Cartesian(norm=False), T.Distance(norm=False)
        ])
    
    def open_tra(self):
        while(len(self.opened_tra) < self.open_tra_num and self.tra_index < self.num_tra):
            tra_index = self.tra_keys[self.tra_index]
            if tra_index not in self.opened_tra:
                self.opened_tra.append(tra_index)
                self.opened_tra_readed_index[tra_index] = -1
                self.opened_tra_readed_random_index[tra_index] = \
                    np.random.permutation(self.tra_len - 1)
            self.tra_index += 1

        if self.check_if_epcho_end():
            self.epcho_end()
    
    def check_and_close_tra(self):
        to_del = []
        for tra in self.opened_tra:
            if self.opened_tra_readed_index[tra] >= (self.tra_len - 2):
                to_del.append(tra)
        for tra in to_del:
            self.opened_tra.remove(tra)
            try:
                del self.opened_tra_readed_index[tra]
                del self.opened_tra_readed_random_index[tra]
            except Exception as e:
                print(e)
    
    def epcho_end(self):
        self.tra_index = 0
        self.shuffle_keys()
        raise StopIteration

    def check_if_epcho_end(self):
        if len(self.opened_tra) == 0 and self.tra_index >= len(self.tra_keys):
            return True
        return False
    
    def shuffle_keys(self):
        np.random.shuffle(self.tra_keys)

    def __next__(self):
        self.check_and_close_tra()
        self.open_tra()

        selected_tra = np.random.choice(self.opened_tra)
        tra_data = self.data[selected_tra]
        selected_tra_readed_index = self.opened_tra_readed_index[selected_tra]
        selected_frame = self.opened_tra_readed_random_index[selected_tra][selected_tra_readed_index+1]
        self.opened_tra_readed_index[selected_tra] += 1
        
        g = self.datas_to_graph(tra_data, selected_frame, self.coords, num_frames=1,
                                transformer=self.transformer)
        return g
    
    @staticmethod
    def datas_to_graph(tra_data, frame, coords, num_frames=1, transformer=None):
        input_fields = tra_data[frame].permute(1, 0)  # (num_nodes, num_feats)
        if num_frames > 1:
            # (num_nodes, num_frames, num_feats)
            target_fields = tra_data[frame+1: frame+1+num_frames].permute(2, 0, 1)
        else:
            target_fields = tra_data[frame+1].permute(1, 0)  # (num_nodes, num_feats)
        
        graph = Data(input_fields=input_fields, pos=coords, y=target_fields).detach().clone()
        if transformer is not None:
            graph = transformer(graph)
        return graph

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_tra * (self.tra_len - 1)


class ODEGraphDatasetRollout(Dataset):
    def __init__(self, data, coords):
        super().__init__()
        self.data = data  # (num_tras, num_steps, num_feats, num_nodes)
        self.coords = coords.squeeze(0).permute(1, 0)  # (num_nodes, 2)

        self.transformer = T.Compose([
            T.KNNGraph(k=8), T.Cartesian(norm=False), T.Distance(norm=False)
        ])
    
    def __getitem__(self, index):
        tra_data = self.data[index]
        g = ODEGraphDataset.datas_to_graph(tra_data, frame=0, coords=self.coords,
                                           num_frames=tra_data.shape[0]-1,
                                           transformer=self.transformer)
        return g
    
    def __len__(self):
        return self.data.shape[0]