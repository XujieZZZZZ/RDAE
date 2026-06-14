from sentence_transformers import SentenceTransformer
import json
import numpy as np
from typing import List, Dict
from knowledge_agents.summary import summarization_module, extract_task_decomposition, extract_formalization_from_task

def get_documents(file_list):
    """
    Load documents from file list.
    Returns both summaries and full data for later processing.
    """
    documents = []
    full_data = []
    for _file in file_list:
        with open(_file, "r", encoding="utf-8") as rf:
            datas = [json.loads(lines.strip()) for lines in rf.readlines()]
        for data in datas:
            if data["summary_dict"] is not None and data["summary_dict"]["summary"] is not None:
                documents.append(data["summary_dict"]["summary"])
                full_data.append(data)
    return documents, full_data

class RetrievalQwenWithSummarizationAgent():
    def __init__(
        self,
        file_list=[
            "knowledge_database/summary_osdi.jsonl",
            "knowledge_database/summary_sosp.jsonl",
        ],
        top_k=3,
    ):
        """
        Initialize retrieval agent with enhanced capabilities.

        Args:
            file_list: List of file paths containing task examples
            top_k: Number of top similar documents to retrieve
        """
        # self.model = SentenceTransformer("Qwen3-Embedding-8B")
        # self.summarization_model = summarization_module
        self.top_k = top_k
        # self.documents, self.full_data = get_documents(file_list)
        # self.embeddings = []
        # for document in self.documents:
        #     self.embeddings.append(self.model.encode(document))

    def retrieve(self, query):
        """
        Retrieve most similar document(s) based on query.

        Args:
            query: User query string

        Returns:
            List of documents with similarity scores
        """
        # 这里暂时不要实现，就直接提供指定的top_k个文档就好，后续再实现检索功能
        return []

    def get_retrieval(self, query, return_top_k=False):
        """
        Retrieve most similar document(s) based on query.

        Args:
            query: User query string
            return_top_k: If True, return top-k documents; otherwise return only the best match

        Returns:
            Single document or list of documents with similarity scores
        """
        query_embedding = self.model.encode(query)
        scores = [self.model.similarity(query_embedding, embedding).item() for embedding in self.embeddings]

        if return_top_k:
            # Get top-k documents
            top_indices = np.argsort(scores)[-self.top_k:][::-1]
            results = []
            for idx in top_indices:
                results.append({
                    'document': self.documents[idx],
                    'full_data': self.full_data[idx],
                    'score': scores[idx]
                })
            return results
        else:
            # Get single best match
            best_idx = scores.index(max(scores))
            document = self.documents[best_idx]
            return document

    def retrieve_similar_tasks(self, user_task_description: str) -> List[Dict]:
        """
        Retrieve similar tasks based on user task description.

        Args:
            user_task_description: Formalized user task description

        Returns:
            List of similar tasks with their decomposition information
        """
        return self.get_retrieval(user_task_description, return_top_k=True)

    def extract_decomposition_patterns(self, similar_tasks: List[Dict]) -> List[Dict]:
        """
        Extract task decomposition patterns from similar tasks.

        Args:
            similar_tasks: List of similar task documents

        Returns:
            List of decomposition patterns with metadata
        """
        patterns = []
        for task in similar_tasks:
            full_data = task['full_data']
            # Extract decomposition information from the task data
            decomposition = extract_task_decomposition(full_data)
            if decomposition:
                patterns.append({
                    'task_summary': task['document'],
                    'decomposition': decomposition,
                    'similarity_score': task['score']
                })
        return patterns

    def extract_formalization_patterns(self, similar_tasks: List[Dict]) -> List[Dict]:
        """
        Extract formalization patterns from similar tasks.

        Args:
            similar_tasks: List of similar task documents

        Returns:
            List of formalization patterns with metadata
        """
        patterns = []
        for task in similar_tasks:
            full_data = task['full_data']
            # Extract formalization information from the task data
            formalization = extract_formalization_from_task(full_data)
            if formalization:
                patterns.append({
                    'task_summary': task['document'],
                    'formalization': formalization.get('formalization', ''),
                    'original_task': formalization.get('original_task', ''),
                    'key_aspects': formalization.get('key_aspects', ''),
                    'similarity_score': task['score']
                })
        return patterns
    

# if __name__ == "__main__":
#     agent = RetrievalQwenWithSummarizationAgent([
#         "/Users/dingch/Desktop/Projections/Auto-Survey-Graph/datasets/system_papers/summary_sosp.jsonl", 
#         "/Users/dingch/Desktop/Projections/Auto-Survey-Graph/datasets/system_papers/summary_osdi.jsonl"
#     ])
#     print(agent.get_retrieval("What is the carbon intensity of the job?"))
    

