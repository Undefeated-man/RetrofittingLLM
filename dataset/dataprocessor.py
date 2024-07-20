import pandas as pd
import numpy as np
import tqdm
import os
import tqdm


class StreamDataLoader:
    def __init__(self, pth, max_length=512):
        self.data = {"train": [], "validation": [], "test": []} 
        self.pth = pth
        self.dir_list = []
        self.stat = {"avg_sentence_len": None, "max_sentence_len": None, "min_sentence_len": None, 
                     "total_sentence": 0, "truncate_sentence": 0}
        self.pivot = 0
        self.max_length = max_length
        
        self.get_whole_data_list()
        
    def __repr__(self):
        """
            Print the statistics of the data
        """
        stat = f"Average sentence length: {self.stat['avg_sentence_len']}\n" + \
                f"Max sentence length: {self.stat['max_sentence_len']}\n" + \
                f"Min sentence length: {self.stat['min_sentence_len']}\n" + \
                f"Total sentence: {self.stat['total_sentence']}\n" + \
                f"Truncate sentence: {self.stat['truncate_sentence']}"
        return stat
        
    def get_whole_data_list(self):
        """
            Get the whole data paths list
        """
        with tqdm.tqdm(total=2) as pbar:
            pbar.set_description(f"Loading test dataset")
            for file in os.listdir(f"{self.pth}/test"):
                self.dir_list.append(f"{self.pth}/test/{file}")
                pbar.update(1)
                
        
    def test_load(self):
        for file in tqdm.tqdm(os.listdir(f"{self.pth}/test")):
            pth = f"{self.pth}/test/{file}"
            self.data["test"] += self.data_process(pd.read_parquet(pth))
        
    def stream_load(self, batch=2):
        """
            Load data from the path, and do the statistics
            
            Args:
                batch: int, the number of file to load at once
        """
        with tqdm.tqdm(total=batch) as pbar:
            pbar.set_description(f"Loading training dataset")
            while self.pivot < len(self.dir_list):
                if self.pivot % batch == 0 and self.pivot != 0:
                    break
                pth = self.dir_list[self.pivot % len(self.dir_list)]
                # setattr(self, f"{dir}_{file}", pd.read_csv(pth))
                self.data["train"] += self.data_process(pd.read_parquet(pth))
                self.pivot += 1 
                pbar.update(1)
        
        self.data["validation"] = self.data_process(pd.read_parquet(f"{self.pth}/train/0000.parquet"))
        
        tmp_stat = self.data["train"]
        self.stat["total_sentence"] = len(tmp_stat)
        tmp_stat = [len(i.split()) for i in tmp_stat if len(i.split()) > 0 and len(i.split()) < self.max_length]
        self.stat["truncate_sentence"] = self.stat["total_sentence"] - len(tmp_stat)
        self.stat["avg_sentence_len"] = np.mean(tmp_stat)
        self.stat["max_sentence_len"] = np.max(tmp_stat)
        self.stat["min_sentence_len"] = np.min(tmp_stat)
                
    def data_process(self, df):
        df["text"] = df["text"].apply(lambda x: x.replace("\n\n", "\n"))
        tmp_data = []
        for text in df["text"].tolist():
            tmp_data += [i for i in text.split("\n") if len(i) > 0]
        return tmp_data
            
        
if __name__ == "__main__":
    pth = "/home/s2497456/mnt/workdir/RetrofittingLLM/dataset/SlimPajama"
    dl = StreamDataLoader(pth)
    dl.stream_load()
    print(dl)
