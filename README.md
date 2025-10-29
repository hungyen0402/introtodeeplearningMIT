# MIT 6.S191: Lời giải và Thực hành Deep Learning
## Portfolio cá nhân của Hoàng Đức Hưng

Chào mừng bạn đến với kho lưu trữ (repository) ghi lại quá trình học tập và hoàn thành các bài lab của tôi từ khóa học uy tín **MIT 6.S191: Introduction to Deep Learning**.

Repo này không chỉ là một bản "fork" đơn thuần. Tôi sử dụng nó làm một portfolio kỹ thuật, nơi tôi tự tay lập trình (code) các giải pháp, hoàn thành các phần `TODO`, và lưu lại kết quả đầu ra của mình. Mục tiêu của tôi là xây dựng sự hiểu biết sâu sắc về các mô hình deep learning từ những nguyên lý cơ bản nhất.

> **Repo gốc của khóa học:** Đây là bản fork từ kho lưu trữ chính thức [MITDeepLearning/introtodeeplearning](https://github.com/MITDeepLearning/introtodeeplearning). Toàn bộ tài liệu gốc và đề bài thuộc bản quyền của đội ngũ giảng viên MIT 6.S191.

---

## 🚀 Các kỹ năng và khái niệm đã thực hành

Qua các bài lab này, tôi đã trực tiếp xây dựng và huấn luyện các mô hình, qua đó nắm vững các kỹ năng:

* **PyTorch (Nền tảng):** Xây dựng các layer mạng nơ-ron tùy chỉnh (custom `nn.Module`), hiểu rõ về tensors, và vòng lặp huấn luyện.
* **Automatic Differentiation:** Vận dụng `torch.autograd` và `loss.backward()` để triển khai thuật toán Gradient Descent, tối ưu hóa các tham số.
* **Mạng Nơ-ron Tích chập (CNNs):** Xây dựng các mô hình CNN để giải quyết bài toán thị giác máy tính (computer vision).
* **Mạng Nơ-ron Hồi quy (RNNs):** Hiểu và xây dựng mạng RNN từ đầu để xử lý dữ liệu tuần tự (sequential data).
* **Mô hình Sinh nhạc (Generative Models):** Huấn luyện một mô hình RNN để tự động sáng tác các bản nhạc mới.
* **Attention & Transformers:** (Sẽ cập nhật) Triển khai cơ chế self-attention, nền tảng của các mô hình ngôn ngữ lớn (LLM) hiện đại.

---

## 📂 Tổng quan các bài Lab đã hoàn thành

Dưới đây là tóm tắt các bài lab tôi đã hoàn thành và những gì tôi đã làm trong mỗi bài.

### 📝 Lab 1: Giới thiệu PyTorch & Sáng tác Nhạc với RNN

Trong bài lab này, tôi đã xây dựng các thành phần cốt lõi của deep learning từ con số không và áp dụng chúng vào một nhiệm vụ sáng tạo.

* **Phần 1: Giới thiệu PyTorch**
    * Tự tay xây dựng một layer `OurDenseLayer` (fully-connected) bằng cách kế thừa `nn.Module`.
    * Thực hành các phép toán tensor và hiểu cách PyTorch quản lý tham số (`nn.Parameter`).
    * Triển khai vòng lặp **Gradient Descent** thủ công để tối ưu hàm loss $L=(x-x_f)^2$, qua đó hiểu rõ cách `loss.backward()` và `x.grad` hoạt động.

* **Phần 2: Sáng tác Nhạc với RNN**
    * Xây dựng mô hình **RNN** tùy chỉnh, bao gồm cả hàm `forward` để xử lý trạng thái ẩn (hidden state) qua từng bước thời gian (time step).
    * Huấn luyện mô hình trên kho dữ liệu nhạc ABC notation.
    * Sử dụng mô hình đã huấn luyện để **sinh ra các bản nhạc mới** bằng cách dự đoán chuỗi nốt nhạc tiếp theo.

### 🖼️ Lab 2: Thị giác Máy tính (Computer Vision)

*(...Đang thực hiện...) - Tôi sẽ cập nhật phần này sau khi hoàn thành.*

### 🤖 Lab 3: Mạng Transformers và Mô hình Ngôn ngữ

*(...Sắp tới...) - Tôi sẽ cập nhật phần này sau khi hoàn thành.*

---

## 🛠️ Công nghệ sử dụng

* **Ngôn ngữ:** Python
* **Thư viện Deep Learning:** PyTorch
* **Môi trường:** Google Colab (sử dụng T4 GPU)
* **Công cụ khác:** NumPy, Matplotlib, Git
