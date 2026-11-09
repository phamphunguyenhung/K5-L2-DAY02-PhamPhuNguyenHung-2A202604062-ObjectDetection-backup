# Báo cáo Ngày 2 — Object Detection

> **Cách dùng mẫu này**
> - Đổi tên tệp thành `MSSV_HoTen.md`, ví dụ `20241234_NguyenVanAn.md`.
> - Thay mọi `...` bằng nội dung của bạn. Xóa các dòng `>` hướng dẫn trước khi nộp.
> - Mỗi mục ghi **bằng chứng** (số, tên tệp, dòng CSV, ảnh overlay), không chỉ kết luận.
> - Viết ngắn: mỗi ý 1–3 câu. Coach chấm theo [RUBRIC.md](../../RUBRIC.md), không chấm độ dài.

## 1. Thông tin học viên

- Họ và tên: Phạm Phú Nguyễn Hưng
- MSSV: 2A202604062
- Lớp: L2
- Link repo: https://github.com/phamphunguyenhung/K5-L2-DAY02-PhamPhuNguyenHung-2A202604062-ObjectDetection

## 2. CP1 — Box, IoU, tọa độ (C1)

- Kết quả `python src/cp1_box_math.py`: 6/6 OK. Toàn bộ 14 unit tests trong `src/tests` đều PASS.
- Box đã chọn (ảnh, `label`, `xyxy`): Ảnh `drive_008`, `label`: `bus`, `xyxy`: `[365.0, 111.0, 610.1, 336.6]`. Kích thước ảnh: $W = 640$, $H = 640$.
- `bbox` COCO tính tay `[x, y, w, h]`:
  - $x = x_1 = 365.0$
  - $y = y_1 = 111.0$
  - $w = x_2 - x_1 = 610.1 - 365.0 = 245.1$
  - $h = y_2 - y_1 = 336.6 - 111.0 = 225.6$
  - Kết quả `bbox`: `[365.0, 111.0, 245.1, 225.6]`.
- Dòng YOLO tính tay `class xc yc w h` (mã lớp theo thứ tự: 0: `car`, 1: `truck`, 2: `bus`, 3: `van`):
  - Lớp `bus` có mã: `2`.
  - $x_c = \frac{x_1 + x_2}{2 \times W} = \frac{365.0 + 610.1}{2 \times 640} = \frac{975.1}{1280} \approx 0.761797$
  - $y_c = \frac{y_1 + y_2}{2 \times H} = \frac{111.0 + 336.6}{2 \times 640} = \frac{447.6}{1280} \approx 0.349688$
  - $w_{\text{norm}} = \frac{w}{W} = \frac{245.1}{640} \approx 0.382969$
  - $h_{\text{norm}} = \frac{h}{H} = \frac{225.6}{640} = 0.352500$
  - Dòng YOLO: `2 0.761797 0.349688 0.382969 0.352500`.
- Đổi ngược dòng YOLO về `[x1, y1, x2, y2]` (có số trung gian), khớp `xyxy` ban đầu không:
  - Tâm điểm ảnh: $(x_c \times W, y_c \times H) = (0.761797 \times 640, 0.349688 \times 640) = (487.55, 223.80)$.
  - Kích thước điểm ảnh: $w = 0.382969 \times 640 = 245.10$, $h = 0.352500 \times 640 = 225.60$.
  - $x_1 = 487.55 - \frac{245.10}{2} = 487.55 - 122.55 = 365.0$.
  - $y_1 = 223.80 - \frac{225.60}{2} = 223.80 - 112.80 = 111.0$.
  - $x_2 = 487.55 + 122.55 = 610.1$.
  - $y_2 = 223.80 + 112.80 = 336.6$.
  - Tọa độ sau khi đổi ngược: `[365.0, 111.0, 610.1, 336.6]` — Khớp chính xác hoàn toàn với `xyxy` ban đầu (sai số 0.0 px).
- Một phép tính IoU viết tay giữa box trên ($A$) và chính nó dịch sang phải 20 điểm ảnh ($B$):
  - Box $A = [365.0, 111.0, 610.1, 336.6]$, diện tích $S_A = 245.1 \times 225.6 = 55294.56$.
  - Box $B = [385.0, 111.0, 630.1, 336.6]$, diện tích $S_B = 245.1 \times 225.6 = 55294.56$.
  - Miền giao ($A \cap B$):
    - $ix = \min(610.1, 630.1) - \max(365.0, 385.0) = 610.1 - 385.0 = 225.1$.
    - $iy = \min(336.6, 336.6) - \max(111.0, 111.0) = 336.6 - 111.0 = 225.6$.
    - Diện tích giao: $S_{\text{inter}} = 225.1 \times 225.6 = 50782.56$.
  - Miền hợp ($A \cup B$):
    - $S_{\text{union}} = S_A + S_B - S_{\text{inter}} = 55294.56 + 55294.56 - 50782.56 = 59806.56$.
  - Kết quả $IoU = \frac{S_{\text{inter}}}{S_{\text{union}}} = \frac{50782.56}{59806.56} \approx 0.849114$ (hoặc rút gọn $\frac{225.1}{245.1 + 20} = \frac{225.1}{265.1} \approx 0.849114$).

## 3. CP2 — Gán nhãn (C2)

- Tổng số box: 101 (`occluded`: 37, `truncated`: 14).
- Tình huống khó 1 (ảnh, vị trí, quyết định, quy tắc dùng): Ảnh `drive_033`, chiếc xe tải màu trắng ở trung tâm có cabin và khoang chở hàng dạng thùng kín phía sau: phân vân giữa `truck` hay `van`. Quyết định chọn nhãn `truck` theo quy tắc phân lớp ("có thùng/ben hoặc khoang chở hàng riêng biệt phía sau; van là xe dạng thân hộp kín liền khối một thể").
- Tình huống khó 2: Ảnh `drive_033`, chiếc xe buýt màu đỏ góc dưới bên trái bị cắt ngang bởi mép dưới và mép trái của ảnh: phân vân có nên vẽ phần bị khuất ngoài ảnh không. Quyết định gán nhãn `bus`, chỉ vẽ bao quanh đúng phần nhìn thấy trong ảnh và tích thuộc tính `truncated = true` theo quy tắc ("Sát mép phần nhìn thấy của xe, không đoán phần bị che khuất; xe chạm mép ảnh vẫn gán nếu đủ bằng chứng phân lớp, box dừng ở mép ảnh").

## 4. CP3 — Xuất và kiểm tra (C3)

- `audit` v1: PASS — 101 hộp — số lần xuất lại: 2 lần (lần 1 bị thiếu nhãn ảnh `drive_038` do chưa save, lần 2 xuất lại đạt PASS trọn vẹn 101 hộp trên cả 4 ảnh).
- Lỗi/cảnh báo đã gặp và cách xử lý:
  - Lỗi: `LỖI drive_038: thiếu tệp nhãn trong bản YOLO` ➔ Cách xử lý: mở lại job trên CVAT, hoàn thành gán nhãn cho ảnh `drive_038`, bấm Save và xuất lại dataset.
  - Cảnh báo: `CẢNH BÁO drive_008 (#13), drive_038 (#65, #73, #89, #93): chạm mép ảnh nhưng truncated=false` ➔ Cách xử lý: các hộp này có tọa độ float sát mép (sai số < 1px do làm tròn hoặc gần viền) nhưng thân xe quan sát thực tế vẫn nguyên vẹn không bị cắt cụt khỏi khung hình, nên quyết định giữ nguyên để bảo toàn ranh giới nhìn thấy tự nhiên.
- Đối chiếu 1 xe: 
  - Xe `car` đầu tiên trong `drive_008`:
    - Dòng YOLO trong `exports/yolo-v1.zip`: `0 0.493914 0.475062 0.097328 0.169094` ➔ Đổi ra `xyxy`: `[284.96, 249.93, 347.25, 358.15]`.
    - Dòng COCO `bbox` trong `exports/coco-v1.zip`: `[284.96, 249.93, 62.29, 108.22]` ➔ Đổi ra `xyxy`: `[284.96, 249.93, 347.25, 358.15]`.
    - Hai kết quả trùng khớp hoàn toàn (chênh lệch tọa độ < 0.0003 px).
- Commit khóa v1 (hash): COMMIT_LOCK_V1_HASH
- Vì sao phải nộp cả COCO và YOLO (1–2 câu): Định dạng YOLO tối ưu để đưa thẳng vào huấn luyện mô hình nhưng bị lược bỏ hoàn toàn các thuộc tính mở rộng (`occluded`, `truncated`). Bản xuất COCO lưu tọa độ tuyệt đối và bảo tồn đầy đủ các trường `attributes` (`occluded`, `truncated`), giúp hệ thống có thể audit chất lượng, lọc ca khó và duy trì tính toàn vẹn của dữ liệu gốc.

## 5. CP4 — So với tham chiếu và sửa (C4)

> Số trong bảng phải khớp `outputs/v1/compare-summary.json` và `outputs/v2/compare-summary.json`.

| Chỉ số | v1 | v2 |
|---|---:|---:|
| TP | ... | ... |
| FP (thừa) | ... | ... |
| FN (bỏ sót) | ... | ... |
| Precision | ... | ... |
| Recall | ... | ... |
| IoU trung bình | ... | ... |
| Sai lớp | ... | ... |

3 lỗi đã phân tích (mỗi lỗi truy được về một dòng trong `outputs/v1/compare-rows.csv`):

| # | Ảnh / dòng CSV | Loại lỗi | Nguyên nhân | Quy tắc liên quan | Sửa hay giữ, vì sao |
|---|---|---|---|---|---|
| 1 | ... | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... | ... |

> Giữ một box khác tham chiếu vẫn được điểm nếu bạn ghi bằng chứng (ảnh overlay, quy tắc).

## 6. CP5 — Nhãn và điểm của mô hình (C5)

| Nhãn dùng để chấm | Precision mô hình | Recall mô hình |
|---|---:|---:|
| v1 của tôi | ... | ... |
| v2 của tôi | ... | ... |
| Tham chiếu | ... | ... |

Giải thích (3–5 câu, chỉ ra ít nhất 1 box cụ thể: ảnh, vị trí, vì sao bị tính FP hoặc FN): ...

## 7. Khó khăn và cách xử lý

> Ghi sự cố kỹ thuật (CVAT, Docker, Python) theo [RULES.md](../../RULES.md): lỗi gì, lúc nào, đã làm gì, đã báo Coach
> chưa. Không có sự cố thì ghi "Không có".

| Lúc nào (CP) | Lỗi / thông báo | Đã làm gì | Kết quả |
|---|---|---|---|
| ... | ... | ... | ... |

## 8. Bonus (không bắt buộc, tối đa 10)

> Chỉ làm sau khi xong CP0–CP5. Điều kiện và cách chấm từng mục ở phần Bonus trong [RUBRIC.md](../../RUBRIC.md).
> Mục nào không làm thì xóa.

### B1 — So khớp chặt hơn ở IoU 0.75

- Lệnh đã chạy: ...
- Bảng v2 ở IoU 0.5 và 0.75 (TP, FP, FN, precision, recall): ...
- Ít nhất 1 box đổi trạng thái (ảnh, dòng CSV, IoU): ...
- Giải thích (3–5 câu): ...

### B2 — Đề xuất quy tắc gán nhãn cho một ca mơ hồ

- Ca mơ hồ (ảnh, vị trí, ảnh chụp/crop trong repo): ...
- Quy tắc hiện tại nói gì, chỗ nào chưa đủ: ...
- Quy tắc đề xuất (1 câu, làm được và kiểm được): ...
- Ví dụ đúng / ví dụ sai theo quy tắc mới: ...
- Ảnh hưởng tới số liệu nếu áp dụng (box nào đổi, precision/recall đổi ra sao): ...

### B3 — Tự chạy YOLO11n và so với dự đoán dựng sẵn

- Nơi chạy và phiên bản (`ultralytics`, ngưỡng `conf`): ...
- Lệnh hoặc link notebook: ...
- Bảng số box dự đoán mỗi ảnh (của bạn / dựng sẵn): ...
- Ít nhất 1 khác biệt và lý do có thể: ...

## 9. Công cụ AI đã dùng

> Bắt buộc điền, kể cả khi không dùng. Quy định ở mục "AI và công cụ" trong [RULES.md](../../RULES.md):
> - **Được:** hỏi khái niệm (IoU, COCO, YOLO), đọc lỗi khi chạy lệnh, sửa câu chữ báo cáo.
> - **Không được:** dùng AI hoặc tự gán nhãn của CVAT để vẽ box v1; để AI viết phần phân tích lỗi CP4 hoặc giải
>   thích CP5 thay bạn.
> - Code CP1 do AI gợi ý: bạn phải giải thích được từng dòng khi Coach hỏi.
>
> Không dùng AI: xóa bảng, giữ dòng "Tôi không dùng công cụ AI cho bài này."

| Công cụ (tên, phiên bản nếu biết) | Dùng ở CP nào | Dùng để làm gì | Tóm tắt câu hỏi/prompt | Tôi đã kiểm tra hoặc sửa gì |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

> Ví dụ một dòng hợp lệ: `ChatGPT` · CP1 · hỏi vì sao IoU ra 0 · "IoU hai box không chạm nhau bằng bao nhiêu" ·
> tự tính lại bằng tay, khớp kết quả `cp1_box_math.py`.

- [ ] Tôi xác nhận nhãn v1 do tôi tự vẽ, và phần phân tích CP4–CP5 do tôi tự viết.

## 10. Tự đánh giá

> Đối chiếu với [RUBRIC.md](../../RUBRIC.md). Coach dùng mục này để góp ý, không cộng điểm theo số bạn tự chấm.

| Tiêu chí | Đã có bằng chứng? | Chỗ còn yếu |
|---|---|---|
| C1 — CP1 | Có / Một phần / Chưa | ... |
| C2 — Nhãn v1 | Có / Một phần / Chưa | ... |
| C3 — Xuất và khóa | Có / Một phần / Chưa | ... |
| C4 — Phân tích lỗi | Có / Một phần / Chưa | ... |
| C5 — Nhãn và mô hình | Có / Một phần / Chưa | ... |
| C6 — Quy ước nộp | Có / Một phần / Chưa | ... |

## 11. Tài liệu tham khảo

> Ghi nguồn ngoài repo bạn đã dùng (trang web, bài báo, video). Chỉ dùng tài liệu trong repo thì ghi "Chỉ dùng tài
> liệu trong repo".

- ...

## Phiếu 1 phút (không chấm điểm)

- Điều rõ nhất hôm nay:
- Điều còn rối nhất:
