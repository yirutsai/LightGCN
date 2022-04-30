import os
import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)

class Gowalla(InMemoryDataset):
    def __init__(self, root,dataset_name="gowalla",raw_file_names="~/data/Gowalla/loc-gowalla_edges.txt", transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, model_name: Optional[str] = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.raw_file_names = raw_file_names
        super(MovieLens, self).__init__(root, transform, pre_transform)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        
        # self.process()

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.model_name}.pt'


    def process(self):
        print(f"Dataset processing")
        import pandas as pd
        # from sentence_transformers import SentenceTransformer

        # data = HeteroData()
        df = pd.read_table(self.raw_file_names,sep="\t",header = None, names=['movieId', 'title', 'genres'],engine="python",encoding="latin-1")
        movie_mapping = {idx: i for i, idx in enumerate(movie_df.index)}
        
        # genres = df['genres'].str.get_dummies('|').values
        # genres = torch.from_numpy(genres).to(torch.float)

        # model = SentenceTransformer(self.model_name)
        # with torch.no_grad():
        #     emb = model.encode(df['title'].values, show_progress_bar=True,
        #                        convert_to_tensor=True).cpu()

        # data['movie'].x = torch.cat([emb, genres], dim=-1)

        rating_df = pd.read_csv(self.raw_paths[1])
        # df = pd.read_csv(self.raw_paths[1],index_col=index_rol)
        user_mapping = {idx: i for i, idx in enumerate(rating_df["userId"].unique())}

        print(f"{self.rating_threshold=}")
        edge_attr = torch.from_numpy(rating_df["rating"].values).view(-1,1).to(torch.long)>= self.rating_threshold
        src = [user_mapping[idx] for idx in rating_df['userId']]
        dst = [movie_mapping[idx] for idx in rating_df['movieId']]
        # edge_index = torch.tensor([src, dst])
        edge_index = [[],[]]
        for i in range(edge_attr.shape[0]):
            if(edge_attr[i]):
                edge_index[0].append(src[i])
                edge_index[1].append(dst[i])
        edge_index = torch.tensor(edge_index)

        # rating = torch.from_numpy(df['rating'].values).to(torch.long)
        # data['user', 'rates', 'movie'].edge_index = edge_index
        # data['user', 'rates', 'movie'].edge_label = rating

        # if self.pre_transform is not None:
        #     data = self.pre_transform(data)
        self.user_mapping = user_mapping
        self.movie_mapping = movie_mapping
        self.edge_index = edge_index
        # self.rating = rating

        # torch.save(self.collate([data]), self.processed_paths[0])
class MovieLens(InMemoryDataset):
    r"""A heterogeneous rating dataset, assembled by GroupLens Research from
    the `MovieLens web site <https://movielens.org>`_, consisting of nodes of
    type :obj:`"movie"` and :obj:`"user"`.
    User ratings for movies are available as ground truth labels for the edges
    between the users and the movies :obj:`("user", "rates", "movie")`.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        model_name (str): Name of model used to transform movie titles to node
            features. The model comes from the`Huggingface SentenceTransformer
            <https://huggingface.co/sentence-transformers>`_.
    """

    


    def __init__(self, root,dataset_name="ml-latest-small", transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, model_name: Optional[str] = "all-MiniLM-L6-v2",rating_threshold=4):
        self.model_name = model_name
        self.url = f'https://files.grouplens.org/datasets/movielens/{dataset_name}.zip'
        self.dataset_name = dataset_name
        self.rating_threshold = rating_threshold
        super(MovieLens, self).__init__(root, transform, pre_transform)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        
        # self.process()

    @property
    def raw_file_names(self) -> List[str]:
        if(self.dataset_name=="ml-latest-small"):
            return [
                osp.join(f'{self.dataset_name}', 'movies.csv'),
                osp.join(f'{self.dataset_name}', 'ratings.csv'),
            ]
        else:
            return [
                osp.join(f'{self.dataset_name}', 'movies.dat'),
                osp.join(f'{self.dataset_name}', 'ratings.dat'),
            ]            

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.model_name}.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self):
        print(f"Dataset processing")
        import pandas as pd
        # from sentence_transformers import SentenceTransformer

        # data = HeteroData()
        if(self.dataset_name=="ml-latest-small"):
            movie_df = pd.read_csv(self.raw_paths[0], index_col='movieId')
            movie_mapping = {idx: i for i, idx in enumerate(movie_df.index)}
            
            # genres = df['genres'].str.get_dummies('|').values
            # genres = torch.from_numpy(genres).to(torch.float)

            # model = SentenceTransformer(self.model_name)
            # with torch.no_grad():
            #     emb = model.encode(df['title'].values, show_progress_bar=True,
            #                        convert_to_tensor=True).cpu()

            # data['movie'].x = torch.cat([emb, genres], dim=-1)

            rating_df = pd.read_csv(self.raw_paths[1])
            # df = pd.read_csv(self.raw_paths[1],index_col=index_rol)
            user_mapping = {idx: i for i, idx in enumerate(rating_df["userId"].unique())}

            print(f"{self.rating_threshold=}")
            edge_attr = torch.from_numpy(rating_df["rating"].values).view(-1,1).to(torch.long)>= self.rating_threshold
            src = [user_mapping[idx] for idx in rating_df['userId']]
            dst = [movie_mapping[idx] for idx in rating_df['movieId']]
            # edge_index = torch.tensor([src, dst])
            edge_index = [[],[]]
            for i in range(edge_attr.shape[0]):
                if(edge_attr[i]):
                    edge_index[0].append(src[i])
                    edge_index[1].append(dst[i])
            edge_index = torch.tensor(edge_index)

            # rating = torch.from_numpy(df['rating'].values).to(torch.long)
            # data['user', 'rates', 'movie'].edge_index = edge_index
            # data['user', 'rates', 'movie'].edge_label = rating

            # if self.pre_transform is not None:
            #     data = self.pre_transform(data)
            self.user_mapping = user_mapping
            self.movie_mapping = movie_mapping
            self.edge_index = edge_index
            # self.rating = rating

            # torch.save(self.collate([data]), self.processed_paths[0])
        elif(self.dataset_name=="ml-1m"):
            movie_df = pd.read_table(self.raw_paths[0],sep="::",header = None, names=['movieId', 'title', 'genres'],engine="python",encoding="latin-1")
            print(movie_df)
            movie_mapping = {idx: i for i, idx in enumerate(movie_df["movieId"].unique())}

            rating_df = pd.read_table(self.raw_paths[1],sep="::",header = None, names=['userId', 'movieId', 'rating', 'timestamp'],engine="python",encoding="latin-1")
            print(rating_df)
            user_mapping = {idx: i for i, idx in enumerate(rating_df["userId"].unique())}
            print(f"{self.rating_threshold=}")
            edge_attr = torch.from_numpy(rating_df["rating"].values).view(-1,1).to(torch.long)>= self.rating_threshold
            src = [user_mapping[idx] for idx in rating_df['userId']]
            dst = [movie_mapping[idx] for idx in rating_df['movieId']]
            # edge_index = torch.tensor([src, dst])
            edge_index = [[],[]]
            for i in range(edge_attr.shape[0]):
                if(edge_attr[i]):
                    edge_index[0].append(src[i])
                    edge_index[1].append(dst[i])
            edge_index = torch.tensor(edge_index)

            # rating = torch.from_numpy(df['rating'].values).to(torch.long)
            # data['user', 'rates', 'movie'].edge_index = edge_index
            # data['user', 'rates', 'movie'].edge_label = rating

            # if self.pre_transform is not None:
            #     data = self.pre_transform(data)
            self.user_mapping = user_mapping
            self.movie_mapping = movie_mapping
            self.edge_index = edge_index