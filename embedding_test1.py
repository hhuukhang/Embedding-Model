import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

print("Đang tải mô hình BAAI/bge-m3...")
embedder = SentenceTransformer('BAAI/bge-m3')

# 1. Định nghĩa các "Nhãn chủ đề" (Labels)
# Mô hình chưa từng được dạy các nhãn này là gì, nó chỉ dựa vào ngữ nghĩa của từ.
topics = [
    "Đây là tài liệu về Toán học, đại số tuyến tính và giải tích đa biến.",
    "Đây là tài liệu về Vật lý, các bài thực nghiệm cơ học và phân tích chuyển động.",
    "Đây là tài liệu về Trí tuệ nhân tạo, học máy và huấn luyện mô hình học tăng cường (Reinforcement Learning)."
]

# 2. Dataset cần phân loại (Các câu văn thô, không hề có nhãn)
texts = [
    "Vectơ gradient luôn chỉ hướng độ dốc tăng nhanh nhất của mặt phẳng tiếp diện.",
    "Phân tích sai số tuyệt đối và sai số tương đối trong bài thực nghiệm đo gia tốc rơi tự do.",
    "Huấn luyện một tác tử ảo tự động vượt qua các ống nước trong game bằng thuật toán Q-Learning."
]

print("\n--- ĐANG MÃ HÓA VECTOR ---")
# Mã hóa Nhãn và Câu văn thành ma trận
topic_embeddings = embedder.encode(topics) # Shape: (3, 1024)
text_embeddings = embedder.encode(texts)   # Shape: (3, 1024)

print("\n--- KẾT QUẢ PHÂN LOẠI CHỦ ĐỀ ---")
# 3. Tính toán và Phân loại
# Duyệt qua từng câu văn trong dataset
for i, text_vector in enumerate(text_embeddings):
    # Tính Cosine Similarity giữa 1 câu văn và TẤT CẢ các nhãn chủ đề
    scores = np.dot(topic_embeddings, text_vector) / (norm(topic_embeddings, axis=1) * norm(text_vector))
    
    # Tìm index của nhãn có điểm số cao nhất
    best_topic_idx = np.argmax(scores)
    best_topic = topics[best_topic_idx]
    confidence = scores[best_topic_idx]
    
    print(f"Văn bản: '{texts[i]}'")
    print(f"-> Phân loại: {best_topic} (Độ tự tin: {confidence:.4f})")
    print("-" * 50)