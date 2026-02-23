# 🧠 Multi-lingual Embedding Service for RAG

Module này chịu trách nhiệm khởi tạo và cung cấp dịch vụ Embedding (chuyển đổi văn bản thành vector) cho hệ thống Retrieval-Augmented Generation (RAG). Hệ thống được thiết kế để xử lý tối ưu cho cả Tiếng Anh và Tiếng Việt.

## 🚀 Tính năng chính
* **Mã hóa Tiếng Anh:** Sử dụng `all-MiniLM-L6-v2` (384 dimensions) để tối ưu hóa tốc độ và tài nguyên.
* **Mã hóa Tiếng Việt & Đa ngôn ngữ:** Sử dụng `BAAI/bge-m3` (1024 dimensions) - mô hình State-of-the-Art cho khả năng nắm bắt ngữ nghĩa ngữ cảnh tiếng Việt xuất sắc.
* **Tìm kiếm Ngữ nghĩa (Semantic Search):** Tích hợp thuật toán tính khoảng cách Cosine Similarity (dựa trên các phép toán Đại số tuyến tính) để truy xuất tài liệu chính xác nhất.

## 🛠️ Cài đặt (Installation)

Đảm bảo bạn đã cài đặt Python 3.8+. Chạy lệnh sau để cài đặt các thư viện cần thiết:

```bash
pip install sentence-transformers numpy