"""Qualitative review notes for the fixed, reproducibly selected 51-error audit sample.
Notes are hypotheses and observations, never replacement NOVA labels.
"""
from pathlib import Path
import pandas as pd
import json, argparse

ROOT=Path(__file__).resolve().parent
R=ROOT/'outputs/nova_tv3_details'
if not R.exists(): R=ROOT
ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=R)
R=ap.parse_args().input
# group, observed input evidence, hypothesis requiring confirmation
NOTES={
'vietnam:8':('cue','Danh sách tiếng Pháp có E1442, modifié, épaississant và sel. Mô hình vẫn đoán 3 thay vì nhãn 4.','Cạnh tranh giữa các tín hiệu thành phần; không phải mô hình hoàn toàn không nhận biết chất làm đặc.'),
'vietnam:43':('vocab','Văn bản chứa INS330, INS211, INS951, INS952(iv), INS950, INS954(i), INS955; mô hình đoán 3.','Cách viết mã INS và nhiều token không có trong từ vựng có thể làm biểu diễn yếu; salt đẩy điểm về 3.'),
'vietnam:59':('cue','Bơ Hạt Phỉ có oil, cocoa, milk powder và hazelnut flavor; nhãn 3 nhưng dự đoán 2.','Quan hệ từ/cụm từ đã học không đủ mô tả ngữ cảnh hỗn hợp; cần đối chiếu thêm nhãn sản phẩm.'),
'vietnam:66':('shortcut','Vitax Regulavit liệt kê bốn loại hoa/lá với phần trăm 34, 32, 27, 7. Các số 27/34/32 ủng hộ dự đoán 3 so với 1.','Mô hình có thể dùng số phần trăm làm dấu hiệu thay thế không bền vững. Cần thí nghiệm bỏ/chuẩn hóa số trên dev.'),
'vietnam:78':('cue','Mozarella cheese có cellulose, salt, enzyme; cellulose ủng hộ nhãn 4 nhưng cellulose added và salt ủng hộ dự đoán 3.','Tương tác unigram/bigram và hệ số chặn có thể đảo chiều quyết định dù đã nhận ra từ cellulose.'),
'vietnam:83':('label_review','Pia Cake mang nhãn 3, có emulsifying agent, synthetic green food colour và synthetic flavor; mô hình đoán 4.','Cần xác minh nhãn dữ liệu cùng danh sách thành phần gốc; chưa thể kết luận dự đoán 4 là nhãn chuẩn.'),
'vietnam:85':('cue','Annatto chỉ có hai token Annatto frø; annatto đẩy điểm về 4 so với nhãn 1.','Từ đa ngữ cảnh có thể bị liên hệ với sản phẩm chứa chất màu; đầu vào rất ngắn thiếu ngữ cảnh.'),
'vietnam:90':('cue','Coconut Cream ghi Coconut extract 95%, water, guar; guar ủng hộ 4 nhưng mô hình đoán 1.','Tín hiệu coconut extract và hệ số chặn lấn át tín hiệu guar; không thể quy cho OOV vì tỷ lệ OOV bằng 0.'),
'vietnam:98':('vocab','Curcumin & Honey Royal Jelly Tablets có thành phần tiếng Việt, lỗi chính tả curcurmim và phần trăm rời.','Từ vựng tiếng Việt và nhiễu nhập liệu có thể làm mất ngữ nghĩa; tên dạng tablets không được đưa vào mô hình.'),
'vietnam:127':('context','Tên Unsalted Creamy Butter nhưng đầu vào mô hình chỉ là Pasteurised cream (separated from cow’s milk).','Thành phần gần với nguyên liệu sữa; thiếu tên/loại sản phẩm hoặc quy trình có thể làm nhầm 2 thành 1.'),
'vietnam:142':('label_review','Coconut CocoXim có Natural coconut flavor, Antioxidant (300); nhãn 3 nhưng dự đoán 4.','Cần kiểm tra nhãn tham chiếu và cách khai báo hương liệu, không tự đổi nhãn theo một từ.'),
'vietnam:144':('label_review','Dao Kombucha có synthetic peach flavor và acidity regulator; nhãn 3, dự đoán 4.','Cần đối chiếu nguồn nhãn và thông tin thành phần; đây là mẫu ưu tiên rà soát nhãn.'),
'vietnam:154':('cue','Bovine Collagen có hydrolyzed collagen, glycine, lysine, vitamins; toàn bộ token unigram đều có trong từ vựng nhưng dự đoán 1.','Tỷ lệ OOV không giải thích lỗi này; cần xem trọng số, hệ số chặn và tính đại diện của nhóm thực phẩm bổ sung.'),
'vietnam:162':('cue','Lotus flavour green tea ghi Green tea (99%), flavour; flavour đẩy điểm về 4 nhưng kết quả cuối là 1.','Tín hiệu nguyên liệu chính và hệ số chặn có thể lấn át hương liệu trong văn bản ngắn.'),
'vietnam:166':('vocab','Gân gà nóng sốt có nhiều cụm tiếng Việt, INS160a(i), INS211, INS600(ii), cùng các chuỗi bất thường như muối in và đường kinh.','Kết hợp OOV, cách ghi mã phụ gia và nhiễu nhập liệu; không thể tách riêng tác động từng yếu tố từ một mẫu.'),
'test:5356':('context','Bio Saure Sahne có văn bản Creme fraiche, nhãn 1 và dự đoán 2.','Chuỗi rất ngắn và giao nhau với train; cần kiểm tra nhãn của các sản phẩm cùng văn bản và ngữ cảnh sản phẩm.'),
'test:17124':('context','Tên Huile d’olive vierge nhưng thành phần chỉ ghi Olives; nhãn 1, dự đoán 2.','Tên sản phẩm gợi ý dầu nhưng mô hình chỉ dùng thành phần; nguồn nhãn và thông tin đầu vào có thể không đồng nhất.'),
'test:11725':('label_review','Sherry Vinegar có đúng chuỗi Sherry vinegar, nhãn 1 nhưng dự đoán 2; văn bản đã xuất hiện trong train.','Cần đối chiếu nhãn của những mẫu cùng thành phần và nguồn nhãn gốc.'),
'test:17047':('format','Trường ingredients_text là YOGURT CON FRUTA DE FRESA Vaca Blanca Fresa, giống mô tả sản phẩm hơn danh sách thành phần.','Thông tin đầu vào không đủ chi tiết; cần kiểm tra nhập sai trường hoặc thiếu danh sách đầy đủ.'),
'test:930':('uncertain','Black forest mix liệt kê các loại quả; tất cả token có trong từ vựng. Mô hình đoán 3 thay nhãn 1.','Các liên hệ từ đã học có thể không phù hợp hỗn hợp này; chưa đủ bằng chứng kết luận nguyên nhân gốc.'),
'test:8306':('context','Chicorée liquide có chuỗi ngắn Extrait de chicorée liquide, không có thông tin quy trình.','Dạng chiết xuất và ngữ cảnh chế biến không được biểu diễn đầy đủ; cần xem nhóm mẫu tương tự.'),
'test:10966':('label_review','Reichi café có arôme, nhiều chiết xuất và citrate de magnésium; nhãn 1, dự đoán 4.','Ưu tiên kiểm tra nhãn và tính đầy đủ của metadata, chưa có nhãn độc lập để sửa.'),
'test:4778':('label_review','Thé des Étoiles có arôme marron và extrait naturel; nhãn 1, dự đoán 4.','Có xung đột giữa liên hệ từ mô hình đã học và nhãn dữ liệu; cần đối chiếu nguồn nhãn.'),
'test:3765':('label_review','Süssstoff Tabletten có Natriumcyclamat, Natriumsaccharin và các chất điều chỉnh; nhãn 1, dự đoán 4.','Cần xác minh nhãn gốc và loại sản phẩm, không dùng nhãn mô hình làm đáp án thay thế.'),
'test:9789':('context','Tên Pure huile de sésame nhưng thành phần chỉ ghi roasted sesame seeds; nhãn 2, dự đoán 1.','Đầu vào chỉ nêu nguyên liệu, thiếu thông tin sản phẩm cuối; thử thêm tên/loại sản phẩm trên dev.'),
'test:7579':('context','Weidemilch Süßrahmbutter có thành phần chỉ Milch; nhãn 2, dự đoán 1.','Mô hình không biết sản phẩm cuối là butter vì không nhận tên; kiểm tra ngữ cảnh và nhãn tương ứng.'),
'test:1413':('context','Australian Unsalted Butter có Cream (Milk), Water; nhãn 2, dự đoán 1.','Thành phần nguyên liệu không mô tả rõ quá trình tạo bơ; thiếu thông tin ngoài văn bản.'),
'test:41':('format','Granulated Cane Sugar ghi Sugar (Sucrose - Cas-00057-50-1); có mã hóa học và số tách rời.','Mã CAS và số tạo thêm đặc trưng không ổn định; cần thí nghiệm chuẩn hóa số/mã có kiểm soát.'),
'test:2021':('context','Apple Cider Vinegar Honey & Ginger ghi vinegar, honey và ginger extract; nhãn 2, dự đoán 3.','Hỗn hợp nhiều thành phần làm ranh giới 2–3 khó nhận biết chỉ bằng từ; cần nguồn nhãn độc lập.'),
'test:14849':('context','Zwiebelschmalz ghi mỡ, tóp, hành, táo, muối và Gewürzextrakte; nhãn 2, dự đoán 3.','Ngữ cảnh hỗn hợp và lớp 2 ít mẫu có thể góp phần; chưa đủ chứng minh nguyên nhân.'),
'test:4040':('format','Blüten Honig chứa văn bản nhãn dài, HTML &quot;, bảng dinh dưỡng, địa chỉ, ngày và mã lô.','Trường thành phần có nhiễu giống OCR/chép toàn nhãn; các token ngoài thành phần tham gia quyết định.'),
'test:4161':('label_review','Organic garlic extra virgin olive oil có natural flavor; nhãn 2, dự đoán 4.','Cần kiểm tra nguồn nhãn, thành phần và ngữ cảnh hương liệu trước khi quy lỗi hoàn toàn cho mô hình.'),
'test:8871':('cue','Cukier 1 kg chỉ có Cukier biały.; OOV bằng 0, nhưng dự đoán 4 thay nhãn 2.','Từ đã có trong từ vựng vẫn có thể học trọng số sai ngữ cảnh; kiểm tra số mẫu và nhãn cho cụm này.'),
'test:21004':('vocab','Málnalekvár có văn bản tiếng Hungary, mã phần trăm và nhiều token ngoài từ vựng; nhãn 3, dự đoán 1.','Biểu diễn đa ngôn ngữ chưa đủ và số có thể làm mất cân bằng tín hiệu; cần so sánh các mẫu cùng ngôn ngữ.'),
'test:22113':('context','Coleslaw chỉ có Cabbage, carrots.; nhãn 3, dự đoán 1; văn bản đã thấy trong train.','Thiếu thông tin chế biến hoặc thiếu thành phần khác; cần đối chiếu sản phẩm gốc và nhãn các mẫu cùng chuỗi.'),
'test:13978':('cue','Đồ uống dừa có eau, sucre, pulpe; sucre ủng hộ 3 nhưng mô hình đoán 1.','Tín hiệu nguyên liệu quả và hệ số chặn có thể lấn át tín hiệu đường; không phải lỗi OOV.'),
'test:9259':('context','Семки XXL ghi Sonnenblumenkerne, Sonnenblumenöl; nhãn 3, dự đoán 2.','Hai thành phần ngắn không mô tả đầy đủ sản phẩm/quy trình; lớp 2 và 3 có từ vựng giao nhau.'),
'test:17843':('context','Tên sản phẩm thiếu, thành phần chỉ açúcar; nhãn 3, dự đoán 2; văn bản đã gặp trong train.','Thiếu ngữ cảnh và có thể nhãn không nhất quán; cần đối chiếu nhóm cùng văn bản.'),
'test:19736':('context','Miel et gelée royale ghi 96% mật ong, 4% gelée royale; nhãn 3, dự đoán 2.','Thành phần chính mật ong có thể lấn át thông tin hỗn hợp; cần thêm ngữ cảnh và ví dụ tương tự.'),
'test:6037':('label_review','Bánh có diphosphates, carbonates và bột nở; nhãn 3, dự đoán 4.','Cần xác minh nguồn nhãn và quy tắc tham chiếu; không kết luận chỉ dựa trên tên phụ gia.'),
'test:19971':('label_review','Jambon có dextrose, arômes naturels và nitrite; nhãn 3, dự đoán 4.','Nhãn và các tín hiệu mô hình sử dụng cần được kiểm tra độc lập; dextrose/arômes đẩy mạnh về 4.'),
'test:3630':('label_review','Fitness-Mischung có Emulgator và Raps-Lecithin; nhãn 3, dự đoán 4.','Cần đối chiếu nhãn gốc và toàn bộ thành phần; chưa xác định mẫu này là lỗi mô hình hay vấn đề nhãn.'),
'test:9400':('cue','Đồ uống táo–củ dền có Kohlensäure, nhãn 4 nhưng dự đoán 1; OOV không bằng 0.','Tín hiệu nước quả và ngữ cảnh đồ uống cạnh tranh; cần kiểm tra nhãn và mẫu tương tự, chưa đủ xác định nguyên nhân.'),
'test:15801':('cue','Infusions Après-repas có mint aromas nhưng nhiều tên thảo mộc; nhãn 4, dự đoán 1.','Tín hiệu thảo mộc và hệ số chặn có thể lấn át aromas; mô hình vẫn nhận ra từ aromas.'),
'test:526':('format','Maná Livianas sabor Vainilla chỉ ghi enriched trigo harina (*), không giống danh sách thành phần đầy đủ.','Trường văn bản thiếu dữ liệu; kiểm tra bản gốc trước khi cải thiện thuật toán.'),
'test:4805':('context','butterschmalz có Butter, Stickstoff; nhãn 4, dự đoán 2.','Cần đối chiếu ngữ cảnh đóng gói/chế biến và nguồn nhãn; văn bản rất ngắn không đủ kết luận.'),
'test:16325':('format','Aceton balsamico có lỗi chính tả và phần bảng dinh dưỡng tiếng Đức lẫn trong thành phần.','Nhiễu chép nhãn và biến thể từ làm biểu diễn suy giảm; cần chuẩn hóa và kiểm tra nguồn.'),
'test:2580':('cue','Dessert Vegetaux có gomme de xanthane nhưng dự đoán 2 thay nhãn 4; OOV bằng 0.','Các cụm dầu/đường và hệ số chặn có thể lấn át tín hiệu chất làm đặc; cần xem biên quyết định.'),
'test:7895':('vocab','Ragú bolognese có văn bản tiếng Czech dài, aroma và kvasničný extrakt; nhãn 4, dự đoán 3.','Đa ngôn ngữ và OOV có thể góp phần, nhưng các token hương liệu đã biết vẫn chưa đủ đổi quyết định.'),
'test:18447':('cue','Thai Green Curry Paste có Modified Tapioca Starch và Salt; nhãn 4, dự đoán 3.','Tín hiệu salt cạnh tranh với modified; mô hình đã nhận ra cả hai nhưng tổng điểm vẫn nghiêng về 3.'),
'test:9501':('cue','Granola Butter có mono- and diglycerides và salt; nhãn 4, dự đoán 3, OOV bằng 0.','Tín hiệu chất phụ gia không thắng tổng điểm của các đặc trưng còn lại và hệ số chặn.'),
}
GROUPS={
'context':('Thiếu ngữ cảnh sản phẩm/quy trình','Kiểm tra sản phẩm gốc; thử thêm tên/nhóm sản phẩm trên dev và kiểm soát nguy cơ học lại cách gán nhãn.'),
'vocab':('Từ vựng, đa ngôn ngữ hoặc mã viết khác','Kiểm tra token OOV; thử char n-gram, chuẩn hóa INS/E-number và lỗi chính tả trên dev, không tinh chỉnh theo test.'),
'format':('Chất lượng hoặc định dạng văn bản','Kiểm tra trường thành phần gốc; loại nội dung bảng dinh dưỡng/địa chỉ khi có căn cứ, chuẩn hóa có kiểm soát.'),
'cue':('Cạnh tranh tín hiệu trong mô hình','Đối chiếu đóng góp token và hệ số chặn; tăng ví dụ phân biệt, thử đặc trưng theo ngữ cảnh trên dev.'),
'label_review':('Nhãn cần đối chiếu độc lập','Đối chiếu nguồn sản phẩm, metadata gán nhãn và tiêu chí tham chiếu; chưa tự đổi nhãn theo dự đoán.'),
'shortcut':('Dấu hiệu số có thể không bền vững','Thử giữ/bỏ/chuẩn hóa số phần trăm trên dev và so sánh bằng phép chia theo nhóm độc lập.'),
'uncertain':('Chưa đủ bằng chứng nguyên nhân gốc','Đọc thêm mẫu cùng loại và nhãn đối chứng trước khi chọn biện pháp cải thiện.')}

df=pd.read_csv(R/'error_review_sample.csv',keep_default_na=False)
assert len(df)==51
notes=[]
for _,x in df.iterrows():
    key=f'{x.dataset}:{x.prediction_row_0based}'
    group,obs,hyp=NOTES[key]
    notes.append({'group_code':group,'primary_review_group':GROUPS[group][0],
                  'observed_evidence':obs,'hypothesis_to_verify':hyp,'proposed_followup':GROUPS[group][1],
                  'review_status':'Đã đọc văn bản và đối chiếu đóng góp mô hình; chưa xác minh nhãn độc lập'})
assert len(NOTES)==len(df)
out=pd.concat([df.reset_index(drop=True),pd.DataFrame(notes)],axis=1)
out.to_csv(R/'error_review_51_annotated.csv',index=False,encoding='utf-8-sig')
g=out.groupby(['dataset','group_code','primary_review_group']).size().reset_index(name='sample_count')
g.to_csv(R/'review_group_counts.csv',index=False,encoding='utf-8-sig')
(R/'review_notes.json').write_text(json.dumps(NOTES,ensure_ascii=False,indent=2),encoding='utf-8')
print(g.to_string(index=False))
