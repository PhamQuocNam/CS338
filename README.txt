** Về SpikeGPT **
- Folder này chứa toàn bộ mã nguồn gốc để định nghĩa và huấn luyện mô hình ngôn ngữ dựa trên SNN, bao gồm cấu trúc mô hình (src/model.py với kiến trúc Spiking RWKV/RFFN), các kịch bản huấn luyện (train.py, run.py), các tối ưu hóa CUDA (cuda/), và bộ nhân cài đặt SNN (Folder spikingjelly) từ paper.
- Lưu ý về trạng thái mã nguồn:
Folder này hiện tại chỉ đang đóng vai trò là một bản gốc được tải về từ GitHub. Trong quá trình thực nghiệm thực tế (ví dụ: đưa lên Kaggle để huấn luyện với bộ dữ liệu mới), các file mã nguồn (như model.py hoặc các file config) sẽ được can thiệp và chỉnh sửa liên tục. Chính là việc tinh chỉnh mã nguồn để bật hoặc tắt cơ chế headqk. Do sự linh hoạt trong quá trình thử nghiệm này, mã nguồn thực tế khi train sẽ sinh ra nhiều phiên bản biến thể cấu trúc mô hình khác nhau, chứ không chỉ cố định ở một cấu trúc duy nhất như tình trạng đang thấy trong Folder hiện tại.

** Về Notebooks **
- Chứa tất cả các notebook của đồ án: 
+ trial: Notebook dùng để đánh giá tập dữ liệu 2000 mẫu do LLM sinh ra để đảm bảo ngữ nghĩa và độ đúng đắn của dữ liệu trước khi data aug để huấn luyện mô hình.
+ demo: Notebook chứa mã nguồn backend của demo (backend chạy trên kaggle để có thể sử dụng cùng lúc 6 mô hình để demo).
+ eval: Notebook dùng để đánh giá điểm của mô hình SpikeGPT sau khi huấn luyện.
+ gpt2-eval: Notebook dùng để đánh giá điểm của mô hình GPT2 sau khi huấn luyện.
+ gpt-medium/small-finetuning: Notebook dùng để huấn luyện GPT2.
+ headqk: Notebook dùng để huấn luyện SpikeGPT.
- Lưu ý: SpikeGPT được huấn luyện nhiều lần và nhiều biến thể (Có headqk, không có) nên được huấn luyện bằng cách gọi nhánh của repo github và huấn luyện trên kaggle, nên cấu trúc mô hình sẽ nằm ở Folder SpikeGPT/

** Về Demo **
- Folder này chứa mã nguồn Frontend của web demo (Hoạt động bằng cách cho Kaggle chạy Backend rồi mở API qua ngrok, sau đó Frontend chỉ gọi API đó để mô hình xử lý, tương tác với cơ sở dữ liệu Supabase rồi trả kết quả về để Frontend hiển thị).
- Mã nguồn Backend là bài báo và paper về việc LLM gọi tool.

** Về Evaluations **
- Folder này chứa toàn bộ kết quả của 6 lần chạy evaluation 5k mẫu dữ liệu bao gồm: Ma trận nhập nhằn, kết quả notebook, biểu đồ kết quả và chi tiết kết quả của từng mẫu.

** Về Documents **
- Folder này chứa .pdf của Slide, Báo cáo, Paper gốc và Nội dung chuẩn bị.

** Về data **
- Agent_data_LLM.json là 2000 mẫu dữ liệu hội thoại one turn được tạo ra bằng Gemini + ChatGPT sau đó được validate lại giữ lại mẫu đạt chuẩn.
- train/valid_tool_data.jsonl lần lượt là 56k/7k6 mẫu sử dụng phương pháp Data Augmentation dựa trên 2000 mẫu dữ liệu gốc để tạo thành.
- test_ood_data là 5k mẫu dữ liệu dùng để đem đi Evaluate mô hình sau khi đã train xong.

** Về Preprocessing **
- Chức năng chính: Chuyển đổi các tệp dữ liệu văn bản thô từ định dạng .jsonl sang định dạng nhị phân (.bin) và tệp chỉ mục (.idx) giúp tối ưu tốc độ khi huấn luyện mô hình ngôn ngữ.
- Thành phần công cụ: preprocess_data.py cùng với các bộ Tokenizer 20B_tokenizer.json và rwkv_vocab để chuẩn hóa dữ liệu đầu vào cho kiến trúc mô hình RWKV và SpikeGPT.