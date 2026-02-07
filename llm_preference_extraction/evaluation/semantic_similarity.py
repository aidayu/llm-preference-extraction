"""意味的類似度計算モジュール

OpenAI Embeddingsを使用してエンティティ間の意味的類似度を計算する。
キャッシュ機能を備え、同じテキストの重複埋め込み計算を避ける。
"""

import os
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# .envファイルから環境変数を読み込む
load_dotenv()

# OpenAIクライアント（遅延初期化）
_client: OpenAI | None = None

# 使用する埋め込みモデル
EMBEDDING_MODEL = "text-embedding-3-small"

# エンティティマッチングの類似度閾値
ENTITY_SIMILARITY_THRESHOLD = 0.75


def _get_client() -> OpenAI:
    """OpenAIクライアントを取得（遅延初期化）"""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it in your .env file or environment."
            )
        _client = OpenAI(api_key=api_key)
    return _client


@lru_cache(maxsize=10000)
def get_embedding(text: str) -> tuple[float, ...]:
    """
    テキストの埋め込みベクトルを取得する（キャッシュ付き）。
    
    Args:
        text: 埋め込みを計算するテキスト
    
    Returns:
        埋め込みベクトル（タプル形式、キャッシュ可能）
    """
    if not text or not text.strip():
        return tuple()
    
    client = _get_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text.strip().lower()
    )
    
    return tuple(response.data[0].embedding)


def compute_cosine_similarity(vec1: tuple[float, ...], vec2: tuple[float, ...]) -> float:
    """
    2つのベクトル間のコサイン類似度を計算する。
    
    Args:
        vec1: ベクトル1
        vec2: ベクトル2
    
    Returns:
        コサイン類似度（-1.0〜1.0）
    """
    if not vec1 or not vec2:
        return 0.0
    
    arr1 = np.array(vec1)
    arr2 = np.array(vec2)
    
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(arr1, arr2) / (norm1 * norm2))


def compute_entity_similarity(entity1: str, entity2: str) -> float:
    """
    2つのエンティティ間の意味的類似度を計算する。
    
    Args:
        entity1: エンティティ1
        entity2: エンティティ2
    
    Returns:
        類似度スコア（0.0〜1.0）
    """
    if not entity1 or not entity2:
        return 0.0
    
    # 完全一致の場合は1.0を返す（API呼び出し節約）
    e1_normalized = entity1.lower().strip()
    e2_normalized = entity2.lower().strip()
    
    if e1_normalized == e2_normalized:
        return 1.0
    
    # 埋め込みを取得して類似度計算
    emb1 = get_embedding(entity1)
    emb2 = get_embedding(entity2)
    
    similarity = compute_cosine_similarity(emb1, emb2)
    
    # 負の類似度は0にクリップ
    return max(0.0, similarity)


def entities_match(entity1: str, entity2: str, threshold: float = ENTITY_SIMILARITY_THRESHOLD) -> bool:
    """
    2つのエンティティが意味的に一致するかどうかを判定する。
    
    Args:
        entity1: エンティティ1
        entity2: エンティティ2
        threshold: 類似度閾値（デフォルト: 0.8）
    
    Returns:
        True if entities match semantically
    """
    similarity = compute_entity_similarity(entity1, entity2)
    return similarity >= threshold


def clear_embedding_cache():
    """埋め込みキャッシュをクリアする"""
    get_embedding.cache_clear()


def get_cache_info():
    """キャッシュの統計情報を取得する"""
    return get_embedding.cache_info()
