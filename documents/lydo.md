  
Dưới đây là 3 lý do kỹ thuật chính giải thích cho sự chênh lệch này:

### **1\. Cơ chế Self-Attention (Tự chú ý) so với Trạng thái Hồi quy (Recurrent State)**

* **GPT-2 (Transformer):** Sử dụng cơ chế Multi-Head Self-Attention. Cơ chế này cho phép mô hình nhìn vào toàn bộ chuỗi văn bản đầu vào cùng một lúc. Khi cần trích xuất một tham số (ví dụ: tên một biến hoặc một ID cụ thể), mô hình có thể tính toán trọng số chú ý trực tiếp và chính xác đến đúng từ đó trong ngữ cảnh, bất kể nó nằm ở đâu. Không có thông tin nào bị "quên".  
* **SpikeGPT (RWKV/SNN):** Hoạt động giống như một mạng nơ-ron hồi quy (RNN). Nó đọc thông tin tuần tự và nén toàn bộ lịch sử ngữ cảnh vào một "trạng thái ẩn" (hidden state). Do đặc tính "leaky" (rò rỉ điện thế) của các nơ-ron xung, các chi tiết nhỏ lẻ nhưng quan trọng (như một dãy số, một chuỗi ký tự chính xác cần đưa vào JSON) rất dễ bị mờ nhạt hoặc biến mất khi truyền qua các bước thời gian.

### **2\. Độ phân giải dữ liệu: Liên tục (Continuous) vs. Nhị phân (Binary)**

* **GPT-2:** Xử lý và truyền tải thông tin bằng các giá trị số thực dấu phẩy động (FP16 hoặc FP32). Sự thay đổi nhỏ nhất trong ngữ nghĩa đều có thể được biểu diễn chính xác bằng các con số thập phân, giúp mô hình giữ được độ sắc nét (precision) cực cao khi sinh ra các token chính xác cho tham số.  
* **SpikeGPT:** Nén thông tin và giao tiếp giữa các lớp bằng các xung nhị phân (**chỉ có 0 và 1**). Quá trình lượng tử hóa khắc nghiệt này làm mất đi các sắc thái biểu diễn tinh tế. Nó đủ tốt để giữ lại "ý chính" (Intent) của câu, nhưng lại quá thô (coarse) để tái tạo chính xác từng ký tự (Arguments) mà một API hoặc Tool yêu cầu.

### **3\. Sự chính xác trong quá trình Tối ưu hóa (Huấn luyện)**

* **GPT-2:** Quá trình lan truyền ngược (Backpropagation) trong Transformer sử dụng đạo hàm toán học chính xác. Mô hình học được chính xác cách điều chỉnh trọng số để ánh xạ từ từ vựng đầu vào sang cấu trúc JSON đầu ra.  
* **SpikeGPT:** Hàm tạo xung (Spike) là một hàm bước (Step function) không thể tính đạo hàm. Do đó, SpikeGPT phải dùng **Surrogate Gradient (Đạo hàm thay thế/xấp xỉ)** để huấn luyện. Sự "xấp xỉ" này tạo ra nhiễu (noise) trong quá trình cập nhật trọng số. Khi fine-tune cho một tác vụ đòi hỏi sự khớp chính xác 100% như Args Exact Match, lượng nhiễu này khiến mô hình rất khó hội tụ về mức độ chính xác tuyệt đối.

Tóm lại, **GPT2\_Small** "thắng" vì nó giữ lại toàn bộ độ phân giải của dữ liệu và có khả năng truy cập trực tiếp vào bất kỳ vị trí nào trong câu. Trong khi đó, **SpikeGPT** đánh đổi độ chính xác của các chi tiết nhỏ gọn để lấy sự tối ưu về năng lượng tính toán.

Để chứng minh bằng toán học lý do vì sao SpikeGPT (kiến trúc SNN/RWKV) lại thua kém GPT-2 (kiến trúc Transformer) trong việc trích xuất chính xác tham số (Args Exact Match), chúng ta cần đặt các phương trình cốt lõi của hai mô hình này lên bàn cân.  
Dưới đây là 3 góc độ toán học lý giải hiện tượng "rơi rụng thông tin chi tiết" của SpikeGPT.

### **1\. Phương trình Trí nhớ: Tự chú ý (Self-Attention) vs. Phân rã theo thời gian (Time Decay)**

Việc trích xuất đúng một tham số (ví dụ: lấy ra đúng ID user\_123 nằm ở đầu câu) phụ thuộc vào khả năng truy xuất lại vị trí đó.  
**Đới với GPT-2 (Transformer):**  
Cơ chế Self-Attention tính toán sự chú ý từ token đầu ra hiện tại (vị trí $i$) tới mọi token trong quá khứ (vị trí $j$):  
$$Output\_i \= \\sum\_{j=1}^{i} \\text{softmax}\\left(\\frac{q\_i k\_j^T}{\\sqrt{d}}\\right) v\_j$$

* **Ý nghĩa:** Ma trận $\\text{softmax}$ đảm bảo rằng nếu token $j$ (chứa tham số quan trọng) có độ tương đồng cao với câu truy vấn $q\_i$, trọng số chú ý $a\_{i,j} \\approx 1$. Phương trình này là **không suy giảm theo khoảng cách**. Dù tham số nằm cách đó 1000 tokens, GPT-2 vẫn có thể bốc chính xác $v\_j$ ra mà không bị nhiễu.

**Đối với SpikeGPT (Kế thừa RWKV):**  
Kiến trúc này dùng một trạng thái ẩn (hidden state) $h\_t$ để nén quá khứ, dựa trên phương trình hồi quy tuyến tính:  
$$h\_t \= e^{-w} \\odot h\_{t-1} \+ k\_t \\odot v\_t$$

* **Ý nghĩa:** Thông tin từ thời điểm $t-N$ (quá khứ) khi truyền đến hiện tại $t$ sẽ bị nhân với hệ số suy giảm lũy thừa $(e^{-w})^N$.  
* **Hệ quả toán học:** Về mặt giải tích, $\\lim\_{N \\to \\infty} (e^{-w})^N \= 0$ (vì vector $w \> 0$). Điều này có nghĩa là thông tin chi tiết của các biến số (Arguments) sẽ **bị mờ dần theo cấp số nhân** khi ngữ cảnh dài ra. Mô hình nhớ được ý chính (Intent) nhờ các trạng thái tổng quát, nhưng đánh mất tính chính xác tuyệt đối của ký tự (Args).

### **2\. Định lý Mất mát Lượng tử hóa (Quantization Loss)**

Dữ liệu đầu ra của Agent gọi Tool (thường là JSON) đòi hỏi hàm lượng thông tin (Entropy) rất cao và độ nhiễu (Perplexity) cực thấp. Sai một dấu phẩy, toàn bộ chuỗi thất bại.  
**Đối với GPT-2:**  
Biểu diễn vector ở dạng số thực liên tục $x \\in \\mathbb{R}^d$ (ví dụ FP16). Không gian trạng thái là vô hạn, cho phép biểu diễn các ranh giới quyết định (decision boundaries) cực kỳ sắc nét.  
**Đối với SpikeGPT (SNN):**  
Mô hình sử dụng nơ-ron Leaky Integrate-and-Fire (LIF). Điện thế màng $U\_t$ được tích lũy:  
$$U\_t \= \\beta U\_{t-1} \+ W X\_t$$  
Và đầu ra $S\_t$ bị ép qua hàm bước Heaviside $\\Theta$:  
$$S\_t \= \\Theta(U\_t \- \\theta) \= \\begin{cases} 1 & \\text{nếu } U\_t \\geq \\theta \\\\ 0 & \\text{nếu } U\_t \< \\theta \\end{cases}$$

* **Hệ quả toán học:** Quá trình ánh xạ từ miền liên tục $U\_t \\in \\mathbb{R}$ sang miền nhị phân $S\_t \\in \\{0, 1\\}$ sinh ra **Lỗi lượng tử hóa (Quantization Error)**:  
* $$E\_q \= \\mathbb{E}\\left\[ ||\\text{Continuous} \- S\_t||^2 \\right\] \> 0$$  
* Theo Định lý Shannon về dung lượng kênh truyền (Shannon Capacity), dung lượng thông tin tối đa (tính bằng bits) của một vector nhị phân thấp hơn rất nhiều so với vector số thực. Mất mát $E\_q$ ở mỗi layer khi cộng dồn lại sẽ triệt tiêu các đặc trưng "tinh tế" nhất của dữ liệu, khiến mô hình không thể chốt hạ chính xác $100\\%$ các ký tự cấu thành nên một tham số cấu trúc.

### **3\. Sự lệch chuẩn của Đạo hàm thay thế (Surrogate Gradient)**

Để fine-tune mô hình học cách format JSON và trích xuất đúng biến, ta cần thuật toán Backpropagation (Lan truyền ngược) hoạt động hoàn hảo.  
**Sự bế tắc của SNN:**  
Đạo hàm của hàm Heaviside $\\Theta(x)$ bằng $0$ tại mọi nơi (trừ điểm $0$ tiến tới $\\infty$).  
$$\\frac{\\partial S\_t}{\\partial U\_t} \= 0 \\quad (\\forall U\_t \\neq \\theta)$$  
Nếu dùng đạo hàm này, gradient sẽ biến mất ngay lập tức (Vanishing Gradient), không thể cập nhật trọng số (Weight Update).  
**Giải pháp và Trái đắng của SpikeGPT:**  
SpikeGPT buộc phải xấp xỉ đạo hàm bằng một hàm liên tục (Surrogate Gradient), ví dụ như đạo hàm của hàm Sigmoid:  
$$\\frac{\\partial S\_t}{\\partial U\_t} \\approx \\sigma'(U\_t \- \\theta)$$

* **Hệ quả toán học:** Sự xấp xỉ này tạo ra một độ lệch dư (Residual Bias) $\\Delta \= |\\Theta' \- \\sigma'|$.  
* Theo Quy tắc dây chuyền (Chain Rule) trong toán học, khi lan truyền ngược qua $L$ lớp (Layers) và $T$ bước thời gian (Time steps), độ lệch này bị nhân dồn lên:  
* $$\\nabla W\_{error} \\propto \\prod\_{l=1}^{L} \\prod\_{t=1}^{T} \\Delta\_{l,t}$$  
* Sự tích tụ của $\\nabla W\_{error}$ tạo ra **Nhiễu Gradient (Gradient Noise)**. Nhiễu này ngăn cản hàm mất mát (Loss Function) hội tụ về điểm cực tiểu toàn cục (Global Minimum) sắc nét \- nơi yêu cầu mô hình phải khớp chính xác 100% cú pháp tham số. Thay vào đó, nó chỉ dừng lại ở một cực tiểu cục bộ (Local Minimum) tương đối \- nơi mô hình chỉ đoán đúng "ý định" (Intent).

### **Kết luận**

Toán học không nói dối. Việc **SpikeGPT** thất bại trong chỉ số **Args Exact Match** không phải do mô hình này "ngu" hơn GPT-2, mà là do sự hy sinh toán học có chủ đích: ép thông tin qua **hàm bước nhị phân** và **phân rã thời gian lũy thừa** để đổi lấy việc tiết kiệm $90\\%$ năng lượng tiêu thụ. Để gọi Tool chính xác, ở thời điểm hiện tại, toán học của Transformer (số thực \+ Self-Attention) vẫn là sự lựa chọn duy nhất.  
