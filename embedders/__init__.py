from .word2vec_embedder import Word2VecEmbedder

def get_embedder_by_name(name: str):
    name = name.lower()
    if name == "word2vec":
        return Word2VecEmbedder
    else:
        raise ValueError(f"未知编码器类型: {name}")