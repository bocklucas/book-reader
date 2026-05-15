import json
import numpy as np
from pathlib import Path
from src.state import get_hash, load_hashes, save_hashes, check_hash

try:
    from sentence_transformers import SentenceTransformer
    _model = None
except ImportError:
    SentenceTransformer = None

def get_model():
    global _model
    if not SentenceTransformer:
        raise RuntimeError("Please install sentence-transformers to use embeddings: pip install sentence-transformers numpy")
    if _model is None:
        # using a fast, small model that is highly effective for semantic search
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

class EmbeddingsDB:
    def __init__(self, db_dir: Path):
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_path = self.db_dir / "chunks.json"
        self.embeddings_path = self.db_dir / "embeddings.npy"
        self.chunks = []
        self.embeddings = None

    def load(self) -> bool:
        if self.chunks_path.exists() and self.embeddings_path.exists():
            try:
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                self.embeddings = np.load(self.embeddings_path)
                return True
            except (json.JSONDecodeError, EOFError, ValueError):
                return False
        return False

    @staticmethod
    def chunk_text(text: str, chapter_name: str) -> list[dict]:
        chunks = []
        paragraphs = text.split('\n\n')
        for i, p in enumerate(paragraphs):
            p = p.strip()
            if len(p) > 20:  # skip very short structural elements
                chunks.append({
                    "chapter": chapter_name,
                    "chunk_index": i,
                    "text": p
                })
        return chunks

    def add_chapter(self, text: str, chapter_name: str):
        """Add a single chapter's text to the internal chunks list. Does NOT encode yet."""
        new_chunks = self.chunk_text(text, chapter_name)
        self.chunks.extend(new_chunks)
        return new_chunks

    def finalize_and_save(self):
        """Encode all pending chunks and save to disk."""
        if not self.chunks:
            return
            
        model = get_model()
        texts = [c["text"] for c in self.chunks]
        print(f"Generating embeddings for {len(texts)} chunks...")
        self.embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)
        np.save(self.embeddings_path, self.embeddings)

    def build_from_chapters(self, chapters_dir: Path):
        hash_path = self.db_dir / ".embedding_hashes.json"
        stored_hashes = load_hashes(hash_path)
        
        chapter_files = sorted(chapters_dir.glob("*.txt"))
        if not chapter_files:
            raise ValueError(f"No chapter files found in {chapters_dir}")
            
        input_state = {
            "chapters": [{"name": f.name, "size": f.stat().st_size} for f in chapter_files]
        }
        current_hash = get_hash(input_state)
        
        if self.load() and check_hash(stored_hashes, "input", current_hash):
            return
            
        model = get_model()
        self.chunks = []
            
        for chap_file in tqdm(chapter_files, desc="Chunking chapters"):
            if chap_file.name == "00-intro.txt":
                continue
            text = chap_file.read_text(encoding="utf-8")
            self.add_chapter(text, chap_file.name)
        
        self.finalize_and_save()
        
        # Save hash after successful build
        stored_hashes["input"] = current_hash
        save_hashes(hash_path, stored_hashes)

    def search(self, query: str, top_k: int = 5, filter_func=None) -> list[tuple[dict, float]]:
        if self.embeddings is None:
            if not self.load():
                raise RuntimeError("Embeddings DB is empty. Call build_from_chapters first.")
                
        model = get_model()
        query_emb = model.encode([query], convert_to_numpy=True)[0]
        
        # Calculate cosine similarity
        norm_embs = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10)
        norm_q = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        similarities = np.dot(norm_embs, norm_q)
        
        # Sort by highest similarity
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in sorted_indices:
            chunk = self.chunks[idx]
            if filter_func is None or filter_func(chunk):
                results.append((chunk, float(similarities[idx])))
                if len(results) >= top_k:
                    break
                    
        return results
