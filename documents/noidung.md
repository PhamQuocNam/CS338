# **Báo Cáo Triển Khai Hệ Thống Chatbot Agentic Trong Quản Lý Thương Mại**

### **1\. Giới thiệu**

Trong bối cảnh bán lẻ và thương mại điện tử hiện đại, các nhà quản lý cửa hàng thường xuyên phải xử lý một lượng lớn dữ liệu động: từ thông tin sản phẩm, tình trạng tồn kho, đến tiến độ đơn hàng và báo cáo doanh thu. Quy trình truyền thống đòi hỏi người quản lý phải thao tác trực tiếp với hệ thống quản trị, hoặc sử dụng các truy vấn cơ sở dữ liệu phức tạp. Điều này không chỉ gây tiêu tốn thời gian mà còn tiềm ẩn nhiều rủi ro sai sót trong quá trình trích xuất số liệu.

Với sự phát triển vượt bậc của các Mô hình Ngôn ngữ Lớn (LLMs) và kiến trúc AI Agentic, việc xây dựng các trợ lý ảo có khả năng tương tác trực tiếp với hệ thống cơ sở dữ liệu thông qua các công cụ (Tool Calling) đã trở nên khả thi. Thay vì điều hướng qua nhiều giao diện hoặc viết mã SQL/GraphQL, người dùng chỉ cần đưa ra yêu cầu bằng ngôn ngữ tự nhiên. Hệ thống sẽ tự động phân tích ngữ nghĩa, gọi các API tương ứng để trích xuất dữ liệu, và tổng hợp thành câu trả lời trực quan, chính xác.

### **2\. Giải pháp: Hệ thống Chatbot Agentic**

Dự án tập trung nghiên cứu và phát triển một hệ thống Chatbot Agentic chuyên biệt phục vụ tác vụ quản lý thương mại. Hệ thống đóng vai trò như một tác tử trung gian thông minh với các năng lực cốt lõi:

* **Truy xuất dữ liệu tự động:** Tương tác trực tiếp với cơ sở dữ liệu và hệ thống quản lý (như danh mục sản phẩm, đơn hàng) thông qua các Tool APIs.  
* **Định tuyến công cụ (Tool Routing):** Tự động suy luận và lựa chọn công cụ (tool) phù hợp nhất dựa trên ý định trong câu hỏi của người dùng.  
* **Tổng hợp và Trình bày:** Xử lý dữ liệu thô trả về từ API, tóm tắt và trình bày lại cho người quản lý dưới định dạng ngôn ngữ tự nhiên, dễ đọc, dễ hiểu.

### **3\. Mục tiêu Dự án**

Việc triển khai hệ thống hướng đến ba mục tiêu trọng tâm:

* **Tối ưu hóa thời gian:** Giảm thiểu đáng kể thời gian thao tác thủ công của người quản lý cửa hàng trên các phần mềm quản trị.  
* **Đơn giản hóa quy trình:** Xóa bỏ rào cản kỹ thuật trong việc truy xuất dữ liệu, cho phép người dùng không có chuyên môn IT vẫn có thể tra cứu số liệu phức tạp.  
* **Nâng cao năng lực ra quyết định:** Cung cấp thông tin theo thời gian thực một cách chính xác, hỗ trợ các quyết định kinh doanh nhanh chóng và hiệu quả.

### **4\. Giới thiệu các Mô hình Ngôn ngữ áp dụng**

Để thực thi tác vụ Agentic và Tool Calling, dự án tiến hành thử nghiệm và so sánh ba kiến trúc mô hình khác nhau, mỗi mô hình mang những đặc điểm kỹ thuật riêng biệt:

#### **4.1. SpikeGPT**

SpikeGPT là một kiến trúc tiên tiến kết hợp giữa Mạng nơ-ron xung (Spiking Neural Networks \- SNNs) và cơ chế ngôn ngữ hiện đại, hoạt động dựa trên các sự kiện xung nhịp để xử lý luồng dữ liệu.

* **Ưu điểm:**  
  * **Hiệu suất năng lượng cực cao:** Do tính chất thưa thớt của SNN, mô hình tiết kiệm đáng kể tài nguyên tính toán và chi phí duy trì hệ thống so với các mô hình Transformer truyền thống.  
  * **Độ trễ thấp:** Phù hợp để triển khai các hệ thống yêu cầu phản hồi nhanh (real-time) hoặc chạy trên các thiết bị/môi trường có giới hạn về phần cứng.  
  * **Xử lý luồng (Streaming):** Kiến trúc có khả năng nạp và xử lý token theo dạng chuỗi thời gian một cách tự nhiên.  
* **Nhược điểm:**  
  * **Độ phức tạp trong huấn luyện:** Cần phương pháp mã hóa dữ liệu đặc thù (như biến đổi dữ liệu sang dạng binary/spike) và việc hội tụ gradient khó khăn hơn.  
  * **Hệ sinh thái:** Các thư viện và tài liệu hỗ trợ chưa phong phú bằng hệ sinh thái của kiến trúc Transformer.

#### **4.2. GPT-2 Small (124M Parameters)**

GPT-2 Small là phiên bản nhẹ nhất trong họ mô hình sinh văn bản của OpenAI, sử dụng kiến trúc Decoder-only Transformer.

* **Ưu điểm:**  
  * **Triển khai nhanh chóng:** Kích thước siêu nhỏ giúp mô hình dễ dàng được tải và tinh chỉnh (fine-tune) trên các máy tính cá nhân hoặc hệ thống đám mây tiêu chuẩn mà không cần GPU đắt tiền.  
  * **Làm Baseline lý tưởng:** Rất phù hợp để thiết lập mức cơ sở (baseline) khi đánh giá các tác vụ NLP cơ bản trước khi chuyển sang mô hình phức tạp hơn.  
* **Nhược điểm:**  
  * **Năng lực suy luận hạn chế:** Khó có thể xử lý các logic Tool Calling phức tạp, đa bước hoặc các câu hỏi chứa nhiều tham số ẩn.  
  * **Dễ bị ảo giác (Hallucination):** Thường xuyên sinh ra các tham số (arguments) không chính xác hoặc gọi sai tên công cụ do dung lượng tham số thấp.

#### **4.3. Fine-tuned GPT-2 Medium (355M Parameters)**

Bản nâng cấp GPT-2 Medium với 355 triệu tham số, được nhóm dự án tiến hành tinh chỉnh (fine-tune) sâu trên bộ dữ liệu Tool Calling đặc thù của ngành thương mại.

* **Ưu điểm:**  
  * **Cân bằng giữa hiệu năng và tài nguyên:** Khả năng nắm bắt cú pháp JSON và lập luận (reasoning) để chọn công cụ tốt hơn hẳn bản Small, nhưng vẫn đủ nhẹ để vận hành mượt mà với chi phí thấp.  
  * **Độ chính xác cao trong miền hẹp:** Nhờ quá trình fine-tune với dữ liệu hội thoại đa lượt, mô hình có khả năng trích xuất các đối số (arguments) từ truy vấn người dùng chính xác hơn nhiều so với việc dùng zero-shot.  
* **Nhược điểm:**  
  * **Giới hạn về Context Window:** Kiến trúc cũ có độ dài ngữ cảnh hạn chế (thường là 1024 token), gây khó khăn khi phải chứa nhiều thông tin từ lịch sử hội thoại hoặc kết quả mô tả API dài.  
  * **Khả năng tổng quát hóa:** Chịu giới hạn về mặt kiến trúc so với các LLM hiện đại, nên hiệu suất có thể giảm sút nhanh chóng khi đối mặt với các cấu trúc lệnh (prompt) nằm ngoài phân phối dữ liệu huấn luyện.

### **5\. Bộ dữ liệu Tool Calling**

Để huấn luyện và tinh chỉnh các mô hình trên (đặc biệt là thiết lập luồng Agent), dự án đã tự xây dựng một bộ dữ liệu hội thoại đa lượt chất lượng cao, mô phỏng sát nhất các tác vụ quản lý nghiệp vụ thực tế.

#### **5.1. Thu thập Dữ liệu & Định dạng**

Bộ dữ liệu bao quát các truy vấn phổ biến trong nghiệp vụ quản lý: kiểm tra tồn kho, tra cứu trạng thái đơn hàng, đối soát doanh thu và thông tin chi tiết sản phẩm. Mỗi mẫu dữ liệu mô phỏng một quy trình hội thoại trọn vẹn giữa hệ thống và người quản lý.

Cấu trúc chuẩn của một mẫu được tổ chức dưới dạng JSON minh bạch về vai trò (role):

* user: Truy vấn hoặc lệnh từ người quản lý.  
* assistant: Phản hồi của Agent (thường chứa chỉ thị gọi API).  
* tool\_calls: Mảng chứa tên công cụ (name) và các đối số tương ứng (arguments) dạng JSON.  
* tool: Dữ liệu thô trả về từ hệ thống sau khi chạy API.

*Ví dụ cấu trúc:*

{  
    "id": "sample\_1801",  
    "description": "Kiểm tra tồn kho cho mã sản phẩm cụ thể",  
    "tool\_used": \["check\_inventory"\],  
    "messages": \[  
        {"role": "user", "content": "Kho còn bao nhiêu sản phẩm P100?"},  
        {  
            "role": "assistant",  
            "content": null,  
            "tool\_calls": \[  
                {  
                    "name": "check\_inventory",   
                    "arguments": {"product\_id": "P100"}  
                }  
            \]  
        }  
    \]  
}

#### **5.2. Quy trình Xây dựng Dữ liệu**

1. **Tạo tập dữ liệu vàng (Gold Standard):** Các chuyên gia tự tay xây dựng một tập hợp các mẫu hội thoại hoàn hảo để làm chuẩn mực.  
2. **Sinh dữ liệu tự động (Synthetic Generation):** Sử dụng các LLM mạnh (như Gemini Pro) làm đầu vào để mô phỏng và mở rộng quy mô bộ dữ liệu dựa trên tập "vàng", tạo ra hàng ngàn mẫu JSON đa dạng.  
3. **Đánh giá chéo (IAA):** Dữ liệu sinh ra được các thành viên trong nhóm rà soát chéo. Áp dụng phương pháp Đồng thuận giữa các người gán nhãn (Inter-Annotator Agreement \- IAA) để lọc bỏ các phản hồi sai lệch, ảo giác.  
4. **Lưu trữ:** Đóng gói toàn bộ tập dữ liệu đã xác thực thành các tệp JSON có cấu trúc để quản lý version và nạp vào pipeline huấn luyện.

#### **5.3. Kiểm chứng và Đảm bảo Chất lượng**

Trước khi đưa vào huấn luyện, toàn bộ file JSON phải vượt qua quy trình kiểm thử tự động nghiêm ngặt:

* **Tính toàn vẹn:** Đảm bảo không có trường dữ liệu nào bị khuyết thiếu (description, tool\_used, messages...).  
* **Tính nhất quán của Tool:** Đối chiếu tool\_used ở cấp độ metadata phải khớp hoàn toàn với các hàm được gọi bên trong mảng tool\_calls của Agent.  
* **Độ chính xác của Argument:** Kiểm tra chéo xem các tham số (ví dụ: product\_id) Agent trích xuất có hoàn toàn trùng khớp với thực thể được người dùng nhắc đến trong câu hỏi gốc hay không.

#### **5.4. Tiền xử lý dữ liệu**

* **Lọc độ dài:** Loại bỏ các mẫu hội thoại có độ dài vượt quá giới hạn (ví dụ: token\_length \> 600) để phù hợp với Context Window của các mô hình nhỏ (như GPT-2).  
* **Tokenization:** Mã hóa văn bản thành các token bằng tokenizer chuyên dụng của từng mô hình tương ứng.  
* **Data Augmentation:** Tăng cường dữ liệu bằng các kỹ thuật xáo trộn thứ tự câu hỏi, thay thế từ đồng nghĩa hoặc đổi tham số giả để chống overfitting và tăng tính đa dạng.  
* **Chuyển đổi đặc thù cho SpikeGPT:** Tiến hành xử lý thêm một bước biến đổi chuỗi dữ liệu sang dạng nhị phân (binary encoding) để tương thích với cơ chế xử lý theo xung (spike) của mạng SNN.

### **6\. Training & Evaluation**

| Model | Valid JSON (%) | Intent Accuracy (%) | Args Exact Match (%) | Precision | Recall | F1 Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SPIKEGPT** |  |  |  |  |  |  |
| SpikeGPT NoHeadQK (epoch 78\) | **100.00** | 86.62 | 0.00 | 0.8781 | 0.8662 | 0.8659 |
| SpikeGPT Scratch (epoch 78\) | 95.98 | 84.64 | 1.78 | 0.8944 | 0.8464 | 0.8655 |
| SpikeGPT Scratch (epoch 220\) | 91.52 | 81.24 | 4.68 | 0.9022 | 0.8124 | 0.8463 |
| SpikeGPT Finetune (epoch 220\) | 99.80 | **94.10** | **13.94** | **0.9459** | **0.9410** | **0.9415** |
| **BASELINE** |  |  |  |  |  |  |
| GPT2\_Small | 99.98 | 96.52 | **68.30** | 0.9671 | 0.9652 | 0.9648 |
| GPT2\_Medium | **100.00** | **96.78** | 67.76 | **0.9696** | **0.9678** | **0.9678** |

### **7\. Demo**

