import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
import time

print("="*50)
print("KHỞI TẠO EMBEDDING SERVICES")
print("="*50)

# Khởi tạo mô hình
start_load = time.time()
print("1. Đang tải model tiếng Anh (all-MiniLM-L6-v2)...")
embedder_en = SentenceTransformer('all-MiniLM-L6-v2')

print("2. Đang tải model tiếng Việt (BAAI/bge-m3)...")
# Lưu ý: Lần chạy đầu tiên sẽ mất thời gian để tải ~2GB dữ liệu của model bge-m3
embedder_vi = SentenceTransformer('BAAI/bge-m3')

print(f"-> Hoàn tất tải model trong {time.time() - start_load:.2f} giây!\n")

# ==========================================
# CHUẨN BỊ DỮ LIỆU (KNOWLEDGE BASE)
# ==========================================
# Sử dụng các kiến thức về Đại số tuyến tính, Giải tích và Machine Learning
corpus_en = [
    "Singular Value Decomposition (SVD) is a technique to reduce matrix dimensionality often used in recommendation systems.",
    "Reinforcement learning trains an agent to make a sequence of decisions to maximize cumulative rewards.",
    "The gradient vector always points in the direction of the steepest ascent of a multivariable function.",
    "The weather in the coastal city is sunny and breezy today."
]

corpus_vi = [
    "Vectơ pháp tuyến luôn vuông góc với mặt phẳng tiếp diện tại một điểm xác định.",
    "Mô hình học tăng cường giúp AI tự động ra quyết định thông qua cơ chế thử sai và nhận phần thưởng.",
    "Ma trận khả nghịch (invertible matrix) và ma trận đơn vị là những khái niệm nền tảng trong Đại số tuyến tính.",
    "Hôm nay trời nắng đẹp, rất thích hợp để đi dạo quanh thành phố."
]

# ==========================================
# QUÁ TRÌNH MÃ HÓA (ENCODING)
# ==========================================
print("Đang mã hóa dữ liệu thành Vector...")
doc_embeddings_en = embedder_en.encode(corpus_en)
doc_embeddings_vi = embedder_vi.encode(corpus_vi)
print(f"Kích thước ma trận tiếng Anh: {doc_embeddings_en.shape}")
print(f"Kích thước ma trận tiếng Việt: {doc_embeddings_vi.shape}\n")

# ==========================================
# HÀM TÌM KIẾM NGỮ NGHĨA
# ==========================================
def semantic_search(query, corpus, doc_embeddings, model):
    """
    Tìm kiếm văn bản liên quan nhất dựa trên Cosine Similarity
    """
    start_search = time.time()
    
    # Mã hóa câu truy vấn
    query_vector = model.encode(query)
    
    # Tính Cosine Similarity (Sử dụng các phép toán ma trận của numpy)
    scores = np.dot(doc_embeddings, query_vector) / (norm(doc_embeddings, axis=1) * norm(query_vector))
    
    # Lấy vị trí của văn bản có điểm số cao nhất
    best_match_idx = np.argmax(scores)
    search_time = time.time() - start_search
    
    print(f"Câu hỏi: '{query}'")
    print(f"-> Kết quả khớp nhất: '{corpus[best_match_idx]}'")
    print(f"-> Độ tương đồng (Cosine Score): {scores[best_match_idx]:.4f}")
    print(f"-> Thời gian tìm kiếm: {search_time:.4f} giây\n")

# ==========================================
# KIỂM THỬ (TESTING)
# ==========================================
print("="*50)
print("TEST TIẾNG ANH (all-MiniLM-L6-v2)")
print("="*50)
query_en = "How do we find the maximum rate of increase of a function?"
semantic_search(query_en, corpus_en, doc_embeddings_en, embedder_en)

print("="*50)
print("TEST TIẾNG VIỆT (BAAI/bge-m3)")
print("="*50)
query_vi = "Mối quan hệ hình học giữa mặt phẳng và đường thẳng vuông góc với nó là gì?"
semantic_search(query_vi, corpus_vi, doc_embeddings_vi, embedder_vi)