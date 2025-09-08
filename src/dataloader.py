from beir import util
from beir.datasets.data_loader import GenericDataLoader
import ir_datasets
import os
import logging

logger = logging.getLogger(__name__)

def get_dataloader(config):
    if config.loader_type == "beir":
        return BEIRDataLoader(config)
    elif config.loader_type == "lotte":
        return LoTTEDataLoader(config)
    else:
        raise NotImplementedError("DataLoader not implemented")

# A Lazy DataLoader wrapping the beir DataLoader. Does downloading and everything for you
# Most of these methods are no longer useful as we have moved to ColBERT as the embedder. get_tsv is a one time use for getting the reindexing + tsv format input reqd for colbert
class BEIRDataLoader: 
    def __init__(self,config):
        self.dataset_name = config.dataset_name
        self.config = config  
        self._loader = None
    
    # fetches corpus, query and qrels data for the given split
    def get_split(self,split="test"):
        if self._loader is None:
            self._load()
        return self._loader._load(split)

    # only fetches corpus and query data without qrels for the given split
    def get_data(self,split="test"):
        if self._loader is None:
            self._load()
        corpus,query,qrels = self._loader.load(split)
        return corpus,query
    
    def _load(self):
        dataset_path = f"./data/{self.dataset_name}"
        if not os.path.exists(f"{dataset_path}/qrels"):
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
            util.download_and_unzip(url, "./data/")   
        self._loader = GenericDataLoader(dataset_path)
        
    def get_tsv(self):
        tsv_path = f"./data/{self.dataset_name}/tsv"
        corpus,query = f"{tsv_path}/corpus.tsv", f"{tsv_path}/queries.tsv"
        
        if os.path.exists(corpus) and os.path.exists(query):
            return corpus, query
        
        if not os.path.exists(tsv_path):
            os.makedirs(tsv_path)
        corpus_data, query_data = self.get_data()
        
        ### NOTE: we have reindexed the corpus everywhere
        
        with open(corpus, "w", encoding="utf-8") as corpus_file:
            for doc_id, doc in enumerate(corpus_data.values()):
                corpus_file.write(f"{doc_id}\t{doc.get('title', '')}\t{doc.get('text', '')}\n")
            corpus_file.flush()
        
        with open(query, "w", encoding="utf-8") as query_file:
            for query_id, query_text in enumerate(query_data.values()):
                query_file.write(f"{query_id}\t{query_text}\n")
            query_file.flush()
        logger.info(f"Saved corpus to {corpus} and query to {query}")
        del corpus_data, query_data
        self._loader = None
        return corpus, query


class LoTTEDataLoader:
    def __init__(self,config):
        self.dataset_name = config.dataset_name
        self.query_type = config.query_type
        self.config = config
        self._loader = None

    def get_data(self, split="test"):
        # We use the ir_datasets package to fetch.
        # IR_DATASETS_HOME must be set.
        dataset = ir_datasets.load(f"lotte/{self.dataset_name}/{split}/{self.query_type}")
        corpus = {doc.doc_id: doc.text for doc in dataset.docs_iter()}
        queries = {query.query_id: query.text for query in dataset.queries_iter()}
        return corpus, queries

    def get_tsv(self, split="test"):
        # Triggers get_data before returning the tsv paths
        self.get_data(split=split)

        IR_DATASETS_HOME = os.getenv("IR_DATASETS_HOME", "./data/.ir_datasets")
        queries_path = f"{IR_DATASETS_HOME}/lotte/lotte_extracted/lotte/{self.dataset_name}/{split}/questions.{self.query_type}.tsv"
        corpus_path = f"{IR_DATASETS_HOME}/lotte/lotte_extracted/lotte/{self.dataset_name}/{split}/collection.tsv"

        return corpus_path, queries_path