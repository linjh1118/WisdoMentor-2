from typing import List
import threading
import os
import sqlite3
from annoy import AnnoyIndex


class AnnoyStore:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(AnnoyStore, cls).__new__(cls)
                    cls._instance.__initialized = False
        return cls._instance

    def __init__(
        self,
        index_path: str = None,
        db_path: str = None,
        dis_type: str = "angular",
        emb_len: int = 1024,
    ):
        if hasattr(self, "__initialized") and self.__initialized:
            return
        if index_path is None:
            index_path = os.path.join(os.path.dirname(__file__), "data", "store.ann")
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "data", "store.db")
        self.index_path = index_path
        self.db_path = db_path
        self.dis_type = dis_type
        self.index = None
        self.lock = threading.Lock()
        self.emb_len = emb_len
        self.conn = None
        self._setup_database()
        self.__initialized = True

    def _setup_database(self) -> None:
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS documents (doc_id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, embedding TEXT)"
        )
        self.conn.commit()

    def _load(self) -> None:
        with self.lock:
            self.index = AnnoyIndex(self.emb_len, self.dis_type)
            if os.path.exists(self.index_path):
                self.index.load(self.index_path)
            else:
                self.index = None

    def add_documents(
        self, doc_list: List[str], doc_emb_list: List[List[float]]
    ) -> None:
        cursor = self.conn.cursor()

        new_docs = []
        new_embeddings = []
        for doc, embedding in zip(doc_list, doc_emb_list):
            cursor.execute(
                "SELECT EXISTS(SELECT 1 FROM documents WHERE content = ?)", (doc,)
            )
            exists = cursor.fetchone()[0]
            if not exists:
                new_docs.append(doc)
                new_embeddings.append(embedding)

        if len(new_docs) == 0:
            return

        cursor.executemany(
            "INSERT INTO documents (content, embedding) VALUES (?, ?)",
            [
                (doc, ",".join(str(x) for x in emb))
                for doc, emb in zip(new_docs, new_embeddings)
            ],
        )
        self.conn.commit()

        with self.lock:
            self.index = AnnoyIndex(self.emb_len, self.dis_type)
            cursor.execute("SELECT doc_id, embedding FROM documents")
            all_docs = cursor.fetchall()
            for doc_id, embedding in all_docs:
                self.index.add_item(doc_id, [float(x) for x in embedding.split(",")])
            self.index.build(10)
            self.index.save(self.index_path)

    def search_by_embedding(
        self, query_embs: List[List[float]], nums: int = 50
    ) -> List[List[str]]:
        self._load()
        # self._reset_timer()

        results = []
        for query_emb in query_embs:
            with self.lock:
                if self.index is None:
                    continue
            res = self.index.get_nns_by_vector(query_emb, nums, include_distances=True)
            ids = res[0]
            results.append([])
            for id in ids:
                cursor = self.conn.cursor()
                cursor.execute("SELECT content FROM documents WHERE doc_id = ?", (id,))
                results[-1].append(cursor.fetchone()[0])
        return results

    def get_id_by_doc(self, doc: str) -> int:
        self._load()
        cursor = self.conn.cursor()
        cursor.execute("SELECT doc_id FROM documents WHERE content = ?", (doc,))
        exist = cursor.fetchone()
        if exist is None:
            return -1
        return exist[0]

    def delete_by_id(self, doc_id: int) -> None:
        self._load()
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self.conn.commit()

        with self.lock:
            self.index = AnnoyIndex(self.emb_len, self.dis_type)
            cursor.execute("SELECT doc_id, embedding FROM documents")
            all_docs = cursor.fetchall()
            for doc_id, embedding in all_docs:
                self.index.add_item(doc_id, [float(x) for x in embedding.split(",")])
            self.index.build(10)
            self.index.save(self.index_path)
