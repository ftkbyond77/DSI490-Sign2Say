Face-Lab/ 


ในส่วนี้เป็นการ สร้างตัว Face Recognition สำหรับการ Classifier ตัว Expression (สีหน้าอารมณ์) โดยใช้ข้อมูลจาก FER2013 โดยคุณสามารถดูได้ที่  FER2013-Data/ (ฉันโหลดผ่าน Kaggle มาเก็บไว้ที่นั่น)


Context: เนื่องด้วย Project เรา (Main Forward Part) เป็นการ Focus บนตัว Sign -> Speech โดยมีการประยุกต์ใช้สีหน้าเข้ามาด้วย ว่าอารมณ์ของเขาในตอนนั้นเป็นอย่างไร (ตอนทำท่า)


แต่เนื่องจากตัวของ MediaPipe หรือ Landmark Extractor มีการจับจุดของตัวหน้าอยู๋แล้ว แต่อาจไม่ได้ Classifier โดยตรง ฉันจึงสนใจนำ Face Recognition Model เข้ามา Classifier เสริมด้วย ซึ่ง Plan ตอนนี้อาจจะเป็นการพยายามรวม 2 ส่วนนี้ด้วยกัน โดยในท้ายที่สุด ตอนที่เราใช้จริงแบบ Real-time มันสามารถทำนายอารมณ์ผ่านสีหน้าได้อย่างแม่นยำ และ รวมถึงไม่ขัดต่อตัว Landmark Extractor ด้วย (ไม่ทำให้ Sign Model ทำนายผิด เพราะ Sign Language มีการประยุกต์สีหน้าด้วยเช่นกัน เพราะฉะนั้นจึงต้องออกแบบให้ชัดเจน)



---

ไอเดียการพัฒนาจาก GPT (ใช้เป็นแนวทางเพิ่มเติมหรือต่อยอดได้ (ปรับให้เข้ากับสถานการณ์ของ data ที่มีและ Project เรามากที่สุด))



ผมแนะนำให้พัฒนา Face Model เป็น Independent Facial Expression Recognition (FER) Branch ที่ทำงานคู่กับ Sign Model โดยแยกหน้าที่กันชัดเจน ไม่ใช้การนำ Face Model เข้าไปเป็นส่วนหนึ่งของ Sign Model ตั้งแต่ต้น


ประเด็นหลัก
- สีหน้ามีทั้งความหมายทางอารมณ์และความหมายทางไวยากรณ์ของภาษามือ เราจึงต้องไม่ให้ Face Classifier มีอำนาจเปลี่ยนแปลงข้อมูลที่ Sign Model ใช้ทำนายท่าทาง
    - ตัวอย่างเช่น คนทำภาษามืออาจยกคิ้ว อ้าปาก หรือขมวดคิ้วเพื่อแสดงโครงสร้างทางภาษา ไม่ได้หมายความว่ากำลังโกรธ ประหลาดใจ หรือมีอารมณ์ตามที่ FER Model จำแนกเสมอไป



ดังนั้นผมจะวางแนวทางให้มี 2 เป้าหมายที่แยกกัน:
1. Sign Recognition: อ่านท่ามือ ทิศทางการเคลื่อนไหว ตำแหน่งร่างกาย และ Facial Grammar เพื่อทำนายภาษามือ
2. Facial Expression Recognition: วิเคราะห์ภาพใบหน้าเพื่อประมาณการแสดงออกทางสีหน้า โดยไม่เปลี่ยนข้อมูลที่ Sign Model ใช้


ผลลัพธ์ทั้งสองส่วนค่อยนำมาประกอบกันในขั้นตอนหลังการทำนาย เช่น การสร้าง Speech หรือการเลือก Prosody ไม่ใช่เอา Emotion ไปเป็นเงื่อนไขในการทำนาย Sign ตั้งแต่ต้น


Architecture หลักของ FER คือ


Face Branch


Face Detection / Corp -> Independent FER Model -> Expression Prediction (Class / Confidence / Quality) 


then Late Fusion / Downstream Integration


-> Context Integration Layer
    - Timestamp alignment + confidence + uncertainty
    - Sign output remains unchanged


-> Speech / Output Layer
    - Sign meaning -> Text / Speech
    - Optional expression-aware prosody



หลักการ: Face Branch อ่านภาพใบหน้าแยกจาก Sign Branch และไม่มีเส้นทางย้อนกลับไปแก้ไข Landmark หรือ Sign Prediction



กฎสำคัญของ Architecture
ส่วน	หน้าที่	สิ่งที่ห้ามทำ
Sign Branch	ทำนายภาษามือ	ห้ามใช้ FER output มาแก้ Sign prediction
Face Branch	จำแนก Facial Expression	ห้ามแก้ไข Sign landmarks
Fusion Layer	รวมผลลัพธ์ตามเวลาและความเชื่อมั่น	ห้ามเปลี่ยน Gloss หรือ Sign Class
Speech Layer	ใช้ผลลัพธ์ประกอบการสร้างเสียง	ห้ามตีความ Expression เป็นความหมายของภาษามือโดยตรง




ส่วนของโมเดล อยากให้ทดลอง 2 ตัวคือ


1. MobileNetV3
2. EfficientNet-B0



MobileNetV3-Large pretrained → Fine-tune FER2013 → Temporal Smoothing



อย่างไรก็ตาม ให้ Agent ตรวจสอบผลจริงของ MobileNetV3-Large เทียบกับ EfficientNet-B0 โดยใช้ Validation Set และเลือกเพียงตัวที่เหมาะสมที่สุดก่อนเริ่ม Training หลัก ไม่ควรเทรนโมเดลใหญ่หลายตัวโดยไม่จำเป็น



โดยทดลอง แล้วหยิบตัวที่ Performance สูงสุดมาใช้จากทั้ง 2 ตัว รวมถึงเขียน Evaluation Metrics ให้ฉันในด้านต่างๆด้วย Accuracy, Latency, .. .. . อื่นๆ




3. FER2013 เพียงพอหรือไม่?
FER2013 เป็นจุดเริ่มต้นที่ดี แต่มีข้อจำกัดที่ต้องเข้าใจ

โดยทั่วไปมี 7 Classes:
Class	ความหมาย
Angry	โกรธ
Disgust	รังเกียจ
Fear	กลัว
Happy	มีความสุข
Sad	เศร้า
Surprise	ประหลาดใจ
Neutral	สีหน้ากลาง

ปัญหาคือ FER2013 เป็น Dataset ภาพใบหน้าแบบ Static Image เป็นหลัก ขณะที่โปรเจกต์ของคุณเป็น Video Stream ของคนที่กำลังทำภาษามือ

ความแตกต่างมีหลายด้าน:
- ภาพ FER2013 มีความละเอียดค่อนข้างต่ำ
- ภาพจริงอาจมีการเคลื่อนไหว
- ใบหน้าอาจหันไปด้านข้าง
- มืออาจบังใบหน้า
- มีการเปลี่ยนแปลงสีหน้าอย่างรวดเร็ว
- Facial Grammar อาจถูกตีความผิดเป็น Emotion
- Distribution ของภาพจาก Dataset อาจต่างจาก Webcam จริง
ดังนั้นผมแนะนำให้ใช้ FER2013 เป็น Initial Training Dataset แต่ต้องทดสอบกับ Video ที่ใกล้เคียงกับสถานการณ์ใช้งานจริงด้วย

อีกประเด็นหนึ่ง: คำว่า Emotion Recognition อาจแรงเกินกว่าความสามารถของ Dataset นี้ ในระบบควรเรียกผลลัพธ์ว่า Predicted Facial Expression หรือ Expression Category เพราะสีหน้าที่มองเห็นไม่ได้ยืนยันสภาวะอารมณ์ภายในของบุคคล




เนื่องจากเป็นภาพนิ่ง คุณอาจจำเป็นต้องพลิกแพลง เพิ่มความสามารถให้รองรับตัว Video หรือ Webcam แบบ Real-time ได้ด้วย



4. วิธีป้องกัน Face Model ไม่ให้ Conflict กับ Sign Model
นี่คือส่วนที่สำคัญที่สุดของการออกแบบ
4.1 ห้ามแก้ไข Shared Landmarks
สมมติ Sign Model ของคุณใช้ Landmark จาก Face, Hands และ Body อยู่แล้ว

Face Model ไม่ควรเข้าไปแก้ไข:
- Face Landmark Coordinates
- Head Pose
- Facial Keypoints
- Hand Coordinates
- Body Coordinates
- Landmark Normalization
- Sign Model Input Tensor
ให้ Face Model ใช้ภาพ RGB ที่ Crop จากใบหน้าแยกต่างหาก



Original Video Frame
        |
        +-------------------------+
        |                         |
        v                         v
Existing Landmark Pipeline    Face Crop Pipeline
        |                         |
        v                         v
Existing Sign Model          FER Model
        |                         |
        v                         v
Sign Prediction             Expression Prediction



4.2 ห้ามใช้ Emotion เป็น Sign Classification Feature โดยตรง
สมมติ Sign Model ทำนายว่า:
sign_prediction = "QUESTION"



แต่ Face Model ทำนายว่า:
expression_prediction = "HAPPY"



ระบบไม่ควรนำ HAPPY ไปแก้ไข QUESTION หรือเลือก Sign Class ใหม่

เพราะ Expression อาจเป็นอารมณ์จริง หรืออาจเป็น Facial Grammar ก็ได้

ให้เก็บผลลัพธ์แยกกัน:
result = {    "sign": {        "prediction": "QUESTION",        "confidence": 0.94    },    "face": {        "expression": "HAPPY",        "confidence": 0.76    }}



4.3 Temporal Alignment
Sign และ Expression อาจเกิดขึ้นไม่พร้อมกัน หรือมีช่วงเวลาที่ความหมายเปลี่ยนไป

จึงต้องมี Timestamp กำกับ

ตัวอย่าง:
{    "timestamp": 2.40,    "sign_prediction": "QUESTION",    "expression": "SURPRISE",    "face_confidence": 0.81}



ไม่ควรนำ Expression จากเฟรมเดียวไปกำหนดให้กับประโยคภาษามือทั้งประโยค

ควรใช้ Window หรือ Segment-level Aggregation ให้สอดคล้องกับช่วงเวลาของ Sign
4.4 Confidence และ Uncertainty
Face Model ต้องส่งค่าความเชื่อมั่นออกมาด้วย

ตัวอย่าง:
{    "expression": "HAPPY",    "confidence": 0.82,    "face_quality": 0.94,    "valid": True}



ถ้าความเชื่อมั่นต่ำ หรือใบหน้าถูกบัง:
{    "expression": "UNKNOWN",    "confidence": 0.0,    "face_quality": 0.18,    "valid": False}



อย่าบังคับให้ Model ต้องทำนายหนึ่งใน 7 Classes ทุกครั้ง เพราะการไม่ทำนายเมื่อข้อมูลไม่เพียงพอเป็นพฤติกรรมที่ดีกว่าการทำนายผิดอย่างมั่นใจ
4.5 Temporal Smoothing
ไม่ควรแสดงผล Emotion ใหม่ทุกเฟรมทันที เพราะจะทำให้ผลลัพธ์กระพริบไปมา เช่น:
Happy → Neutral → Sad → Happy → Angry


ให้ใช้:
- Exponential Moving Average (EMA)
- Moving Average
- Temporal Majority Voting
- Hysteresis
- Minimum Duration
- Confidence Threshold
เริ่มจาก EMA + Hysteresis ก่อนก็เพียงพอ

ข้อควรระวัง: การ Smooth ต้องทำเฉพาะผลลัพธ์ Face Branch ไม่ใช่ไป Smooth Landmark ที่ Sign Model ใช้อยู่โดยไม่ผ่านการตรวจสอบ



5. Metrics และ Evaluation ที่ต้องมี
ผมแนะนำให้แยก Evaluation เป็น 3 ระดับ เพราะ Accuracy ของ Face Model อย่างเดียวไม่สามารถพิสูจน์ได้ว่าเอาไปใช้ร่วมกับ Sign Model แล้วจะไม่เกิดปัญหา
ระดับ A — Face Model Performance
Metric	จุดประสงค์	ความสำคัญ
Accuracy	วัดความถูกต้องโดยรวม	สูง
Macro-F1	วัดผลทุก Class อย่างสมดุล	สูงมาก
Weighted-F1	คำนึงถึงจำนวนตัวอย่างแต่ละ Class	สูง
UAR / Balanced Accuracy	ตรวจสอบผลเมื่อ Class ไม่สมดุล	สูงมาก
Per-class Recall	ดูว่า Class ใดถูกพลาดบ่อย	สูงมาก
Confusion Matrix	ตรวจสอบการจำแนกผิดระหว่างอารมณ์	สูงมาก
ECE	วัดความ Calibration ของ Confidence	สูง
Inference Latency	วัดความเร็ว	สูงมาก
FPS	ประเมินความสามารถ Real-time	สูงมาก

ผมให้น้ำหนักกับ Macro-F1 + UAR + Per-class Recall มากกว่า Accuracy เพียงอย่างเดียว

ตัวอย่างเช่น ถ้า Neutral มีข้อมูลมากกว่า Class อื่นมาก โมเดลอาจทำนาย Neutral เก่งจน Accuracy สูง แต่กลับจำแนก Fear, Disgust หรือ Sad ได้ไม่ดี
ระดับ B — Non-Conflict Evaluation
นี่คือชุดทดสอบที่ผมอยากให้ Agent สร้างขึ้นโดยเฉพาะ
Test	วิธีทดสอบ	สิ่งที่ต้องการพิสูจน์
Sign Baseline	ทดสอบ Sign Model เดี่ยว	ได้ค่าพื้นฐาน
Dual Branch	เปิด Face Model พร้อม Sign Model	Sign ไม่แย่ลง
Face Failure	ทำให้ Face Model ทำนายผิดหรือไม่มั่นใจ	Sign ยังทำงานได้เหมือนเดิม
Face Occlusion	บังใบหน้า	Sign ไม่เสียความแม่นยำ
Face Disable	ปิด Face Branch	Sign output ไม่เปลี่ยน
Expression Variation	เปลี่ยนสีหน้าแต่ทำ Sign เดิม	Sign ยังคงถูกต้อง
Facial Grammar	ทดสอบ Sign ที่ใช้สีหน้าเป็นองค์ประกอบทางภาษา	Sign Model ยังอ่านสีหน้าได้
Timing Misalignment	เลื่อน Timestamp ของ Face Branch	Sign Prediction ไม่เปลี่ยน

ตัวชี้วัดสำคัญ:
- Sign Accuracy Difference
- Sign Macro-F1 Difference
- Sign WER / CER หากเป็น Sequence Prediction
- Sign Segment Detection Recall
- Sign Prediction Consistency
- Sign Latency Difference
ตัวอย่างเกณฑ์เบื้องต้น:
\[
\Delta F1_{sign}
=
F1_{integrated}-F1_{baseline}
\]

เป้าหมายคือ:
\[
\Delta F1_{sign} \geq -0.005
\]

หรือยอมให้ลดลงไม่เกิน 0.5 Percentage Points ในการประเมินแบบเดียวกัน แต่สำหรับ Architecture ที่แยก Branch จริง ๆ ควรตั้งเป้าให้ Sign Output เหมือนเดิมทุกประการเมื่อเปิดหรือปิด Face Branch
ระดับ C — Integrated Real-time Evaluation
ต้องวัดระบบทั้งหมดด้วย ไม่ใช่แค่ Face Model
Metric	จุดประสงค์
End-to-End Latency	เวลาตั้งแต่รับ Frame จนได้ผลลัพธ์
Face Branch Latency	เวลาเฉพาะ Face Model
Sign Branch Latency	เวลาเฉพาะ Sign Model
FPS	จำนวนเฟรมที่ประมวลผลได้ต่อวินาที
GPU Memory	การใช้ VRAM รวม
CPU Memory	การใช้ RAM
Face Detection Failure Rate	อัตราที่ตรวจจับใบหน้าไม่ได้
Expression Stability	ความนิ่งของ Prediction ตามเวลา
Unknown Rate	สัดส่วนที่ Face Model ปฏิเสธการทำนาย


6. Training Strategy ที่ผมแนะนำ
สำหรับเครื่อง RTX 2050 4GB ผมจะไม่เริ่มจากการ Train CNN ตั้งแต่ศูนย์

ใช้แนวทางนี้:
Recommended Training Pipeline
1. Face Detection and Cropping
   
   ใช้ Face Detector ที่มีอยู่แล้วในระบบ ถ้ามี และ Crop เฉพาะใบหน้าจากภาพต้นฉบับ
2. Pretrained MobileNetV3-Large
   
   โหลด ImageNet pretrained weights แล้วเปลี่ยน Classification Head เป็น 7 Classes ของ FER2013
3. Transfer Learning
   
   เริ่ม Freeze Backbone บางส่วนเพื่อฝึก Classification Head จากนั้น Unfreeze บาง Layer เพื่อ Fine-tune โดยใช้ Learning Rate ต่ำ
4. Class Imbalance Handling
   
   วิเคราะห์จำนวนข้อมูลแต่ละ Class ก่อนเลือก Weighted Cross Entropy หรือ Focal Loss ไม่ใช้ทั้งสองอย่างโดยไม่มีเหตุผล
5. Validation and Calibration
   
   เลือก Checkpoint ด้วย Macro-F1 และตรวจ Calibration ของ Confidence
6. Temporal Smoothing
   
   เพิ่ม EMA + Hysteresis ใน Inference ไม่จำเป็นต้อง Train LSTM/Transformer เพิ่มจาก FER2013 ตั้งแต่แรก
7. Real-world Testing
   
   ทดสอบกับวิดีโอภาษามือจริงและตรวจสอบ Non-Conflict กับ Sign Model

ทำไมยังไม่ใช้ LSTM หรือ Transformer?

เพราะ FER2013 เป็นข้อมูลภาพนิ่งเป็นหลัก การเพิ่ม Temporal Model โดยไม่มี Video Dataset ที่มี Label ตามเวลา ไม่ได้ทำให้โมเดลเข้าใจการเปลี่ยนแปลงของสีหน้าได้ดีขึ้นโดยอัตโนมัติ

เริ่มด้วย CNN + Temporal Smoothing จะประหยัดเวลาและง่ายต่อการ Debug มากกว่า

ถ้าในอนาคตมี Video Dataset ที่เหมาะสม ค่อยพิจารณา TCN หรือ GRU เป็น Temporal Head


---

# SUPER PROMPT: FACE-LAB — HIGH-PERFORMANCE FACIAL EXPRESSION RECOGNITION WITH NON-CONFLICT SIGN LANGUAGE INTEGRATION

**Project:** Face-Lab  
**Primary Objective:** Develop a lightweight, accurate, real-time Facial Expression Recognition model  
**Dataset:** FER2013  
**Base Model Strategy:** Pretrained Deep Learning + Transfer Learning  
**Primary Candidate:** MobileNetV3-Large  
**Alternative Candidates:** EfficientNet-B0, ResNet18  
**Hardware:** NVIDIA RTX 2050 4GB VRAM  
**System RAM:** 16GB  
**Development Priority:** High Accuracy, Low Training Cost, Low Inference Latency, Non-Conflict Integration  
**Existing System:** A Sign Language Recognition Model already exists and must be inspected by the coding agent.

---

# PART 0 — ROLE AND MISSION

## 0.1 Your Role

Act as a Senior Computer Vision Engineer, Deep Learning Research Engineer, Facial Expression Recognition Specialist, and ML Systems Architect.

You are responsible for designing and implementing a production-oriented Facial Expression Recognition subsystem.

Your responsibilities include:

1. Inspecting the existing project.
2. Inspecting the existing Sign Language Model.
3. Understanding the current video pipeline.
4. Understanding the existing face detection and landmark extraction pipeline.
5. Investigating FER2013.
6. Selecting an appropriate pretrained architecture.
7. Training and optimizing the FER model.
8. Evaluating classification performance.
9. Evaluating real-time performance.
10. Designing an independent Face Branch.
11. Preventing conflicts with Sign Recognition.
12. Integrating the two branches safely.
13. Implementing confidence-based prediction.
14. Implementing temporal smoothing.
15. Creating reproducible experiments.
16. Producing technical documentation.

Do not modify unrelated project components.

## 0.2 Primary Mission

Develop an accurate and computationally efficient Facial Expression Recognition model using FER2013.

The model must be able to classify facial expressions from video frames.

The target expression categories are:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

Verify the actual dataset labels before implementation.

The model must operate independently from the existing Sign Recognition Model.

The final system should be capable of running both models simultaneously on the same input video stream.

## 0.3 Critical System Requirement

The most important requirement is:

**Facial Expression Recognition must never degrade, modify, or interfere with the existing Sign Language Recognition pipeline.**

Facial expressions may contain two different types of information:

1. Emotional or expressive facial characteristics.
2. Linguistic facial information used by Sign Language.

These two categories must not be treated as equivalent.

The Sign Model may depend on facial movements, eyebrows, mouth shapes, eye movements, head pose, and other facial landmarks as part of sign-language grammar.

The Face Model must not overwrite or alter these features.

## 0.4 Core Design Principle

Use an independent dual-branch architecture.

Do not merge the Face Model into the Sign Model during the initial development stage.

Do not retrain the existing Sign Model unless there is a technically justified reason and explicit approval.

Preserve the existing Sign Model as the primary linguistic recognition system.

The Face Model is an auxiliary perception component.

---

# PART 1 — PROJECT INSPECTION

## 1.1 Workspace

The development workspace is:

`Face-Lab/`

The FER2013 dataset is available at:

`FER2013-Data/`

The existing Sign Recognition Model is assumed to exist elsewhere in the project.

You must locate and inspect it autonomously.

Do not assume a fixed directory structure.

## 1.2 Existing Project Discovery

Inspect:

1. Project root.
2. Face-Lab directory.
3. FER2013-Data directory.
4. Existing Sign Model directory.
5. Existing video inference pipeline.
6. Existing face detection implementation.
7. Existing landmark extraction implementation.
8. Existing model interfaces.
9. Existing preprocessing functions.
10. Existing configuration files.
11. Existing inference scripts.
12. Existing evaluation reports.
13. Existing GPU dependencies.

Preserve the original files.

Do not overwrite the existing Sign Model.

## 1.3 Sign Model Investigation

Before implementing the Face Model, identify how the current Sign Recognition system works.

Determine:

- Model architecture.
- Input modality.
- Input tensor shape.
- Input frame rate.
- Sequence length.
- Landmark representation.
- Face landmark usage.
- Body landmark usage.
- Hand landmark usage.
- Coordinate normalization.
- Temporal preprocessing.
- Prediction interface.
- Output structure.
- Inference latency.
- GPU and CPU utilization.

Investigate whether the existing Sign Model uses facial information.

Specifically determine whether it uses:

- Eyebrow coordinates.
- Eye landmarks.
- Mouth landmarks.
- Jaw landmarks.
- Head orientation.
- Facial movement.
- Facial expression-related features.

Do not remove any of these features merely because a separate Face Model is introduced.

## 1.4 Existing Sign Pipeline Compatibility

Identify the exact location where the Face Branch can operate independently.

The preferred integration point is the original video frame before or alongside landmark extraction.

The Face Branch should consume a separate image crop.

The Sign Branch must continue receiving its original input.

Document the existing pipeline before making modifications.

## 1.5 Hardware Inspection

Inspect:

- GPU model.
- VRAM.
- CUDA version.
- PyTorch version.
- Available system RAM.
- CPU specifications.
- Available storage.
- Installed image-processing libraries.

The primary local GPU is NVIDIA RTX 2050 with 4GB VRAM.

Optimize the training and inference pipeline for this hardware.

---

# PART 2 — DATASET INVESTIGATION: FER2013

## 2.1 Dataset Source

Use the locally available dataset:

`FER2013-Data/`

Do not automatically download another copy.

Inspect the current files and dataset format.

The dataset may contain:

- CSV files.
- Image directories.
- Train, validation, and test directories.
- Label metadata.

Determine the actual structure before implementing the loader.

## 2.2 Dataset Verification

Verify:

1. Number of samples.
2. Number of classes.
3. Image resolution.
4. Color channels.
5. Pixel value range.
6. Dataset split.
7. Class distribution.
8. Missing files.
9. Corrupted images.
10. Duplicate samples.
11. Label consistency.

Do not assume the dataset structure.

## 2.3 FER2013 Categories

The expected seven categories are:

- Angry.
- Disgust.
- Fear.
- Happy.
- Sad.
- Surprise.
- Neutral.

Map the actual dataset labels to consistent internal class IDs.

Create an explicit label mapping file.

## 2.4 Dataset Profiling

Produce:

`reports/dataset_profile.md`

Include:

- Total samples.
- Samples per class.
- Image resolution.
- Image format.
- Color channel distribution.
- Dataset split.
- Missing data.
- Corrupted samples.
- Class imbalance.
- Duplicate detection results.

Generate a class distribution chart.

## 2.5 Dataset Integrity

Check for:

- Exact duplicate images.
- Near-duplicate images.
- Incorrect labels.
- Invalid pixel values.
- Unexpected image dimensions.

Do not silently discard large numbers of samples.

Record every exclusion reason.

## 2.6 Dataset Splitting

If official train, validation, and test splits exist, preserve them.

Do not combine the official test set with training data.

If no validation set exists, create a stratified validation split from training data.

Do not use the test set for hyperparameter selection.

## 2.7 Identity Leakage Consideration

FER2013 may not provide reliable identity-level grouping.

Investigate whether identity leakage can be assessed.

Do not claim a person-disjoint split unless identity information or defensible identity grouping is available.

Document this limitation.

---

# PART 3 — MODEL SELECTION

## 3.1 Model Selection Objective

Select the smallest model that provides strong classification performance and acceptable real-time inference.

Prioritize:

1. Macro-F1.
2. UAR / Balanced Accuracy.
3. Per-class Recall.
4. Calibration.
5. Inference Latency.
6. Model Size.
7. GPU Memory.

Do not select a model based only on overall Accuracy.

## 3.2 Primary Candidate: MobileNetV3-Large

Start with:

`MobileNetV3-Large`

Use pretrained ImageNet weights.

Replace the final classification layer with a seven-class output layer.

Investigate:

- Input resolution.
- Feature dimension.
- Backbone structure.
- Classification head.
- Pretrained weight compatibility.

MobileNetV3-Large is the primary candidate because it provides a good balance between accuracy and computational efficiency.

## 3.3 Alternative Candidate: EfficientNet-B0

Evaluate EfficientNet-B0 as an alternative when its expected performance advantage justifies the additional experiment.

Use pretrained ImageNet weights.

Compare:

- Macro-F1.
- UAR.
- Per-class Recall.
- Latency.
- Model size.
- Memory consumption.

Do not automatically train both architectures to completion.

## 3.4 Alternative Candidate: ResNet18

ResNet18 may be used as a lightweight baseline.

Its purpose is to provide a reference architecture and help determine whether a more complex backbone provides meaningful improvements.

## 3.5 Architecture Decision

Perform an architecture feasibility assessment before training.

Choose one primary model.

Avoid unnecessary experiments.

If a short pilot is needed, keep it computationally inexpensive and use the same data split.

Do not conduct several full training runs without approval.

## 3.6 Pretrained Model Requirements

Use publicly available pretrained weights with verifiable provenance.

Record:

- Model name.
- Weight source.
- Weight version.
- License.
- Input normalization.
- Original training dataset.
- Expected input format.

Do not assume ImageNet preprocessing is automatically appropriate for FER2013.

---

# PART 4 — IMAGE PREPROCESSING

## 4.1 Input Resolution

FER2013 commonly contains low-resolution grayscale images.

Inspect the actual image dimensions.

Evaluate a suitable model input resolution.

Potential options:

- 48 × 48.
- 96 × 96.
- 128 × 128.
- 224 × 224.

Do not enlarge images unnecessarily.

Choose a resolution based on measured performance and inference cost.

## 4.2 Grayscale and RGB

Determine whether the pretrained backbone expects three-channel RGB input.

If FER2013 images are grayscale, evaluate:

1. Replicating the grayscale channel three times.
2. Modifying the first convolution to accept one channel.

Prefer the simplest approach that preserves pretrained weight compatibility.

Do not convert grayscale data to artificial color images.

## 4.3 Normalization

Use preprocessing compatible with the selected pretrained weights.

Compare the effects of:

- Dataset-specific mean and standard deviation.
- Pretrained ImageNet normalization.
- Input scaling.

Record the selected method.

## 4.4 Training Augmentation

Use moderate augmentation.

Potential transformations:

- Small rotation.
- Horizontal flip.
- Small translation.
- Random crop.
- Mild brightness adjustment.
- Mild contrast adjustment.
- Mild Gaussian noise.
- Small affine transformation.

Avoid transformations that change the semantic expression.

Do not use aggressive rotation, deformation, or cropping.

## 4.5 Validation and Test Preprocessing

Do not use random augmentation during validation or testing.

Use deterministic preprocessing.

Ensure all splits use consistent input dimensions and normalization.

---

# PART 5 — TRAINING STRATEGY

## 5.1 Training Philosophy

Use Transfer Learning rather than training a CNN from scratch.

Start from pretrained weights.

The objective is to maximize performance while minimizing training time.

## 5.2 Two-Stage Fine-Tuning

Use a practical two-stage strategy within one planned training workflow.

Stage A:

- Freeze most of the backbone.
- Train the classification head.
- Confirm stable convergence.

Stage B:

- Unfreeze selected upper backbone layers.
- Use a lower learning rate.
- Fine-tune the model.

Do not automatically unfreeze every layer.

## 5.3 Optimizer

Start with AdamW unless the selected implementation has a justified alternative.

Investigate:

- Learning rate.
- Weight decay.
- Betas.
- Gradient clipping.

Use separate learning rates for the backbone and classification head if appropriate.

## 5.4 Learning Rate

Use a conservative backbone learning rate.

Use a higher learning rate for the newly initialized classification head.

Evaluate:

- Warmup.
- Cosine decay.
- ReduceLROnPlateau.

Do not use a learning-rate schedule without validation.

## 5.5 Class Imbalance

Calculate the class distribution before choosing the loss function.

Potential approaches:

- Weighted Cross Entropy.
- Focal Loss.
- Class-aware sampling.
- Balanced sampling.

Choose one primary approach based on the actual distribution.

Do not combine all imbalance methods indiscriminately.

Avoid oversampling so aggressively that the model memorizes minority samples.

## 5.6 Training Configuration

Use a configuration file.

Suggested initial configuration:

```yaml
model:
  name: mobilenet_v3_large
  pretrained: true
  num_classes: 7

training:
  seed: 42
  batch_size: 32
  epochs: 20
  optimizer: adamw
  backbone_lr: 0.00001
  head_lr: 0.0001
  weight_decay: 0.0001
  mixed_precision: true
  early_stopping: true

evaluation:
  monitor: macro_f1
  save_best_checkpoint: true
```

These are initial candidates, not mandatory final values.

Adapt batch size to the actual GPU memory.

## 5.7 Mixed Precision

Use FP16 where supported and stable.

Use BF16 only if the GPU and framework support it efficiently.

Do not force an unsupported precision mode.

## 5.8 Memory Optimization

Investigate:

- Batch size.
- Gradient accumulation.
- Mixed precision.
- Efficient data loading.
- Pin memory.
- Worker count.
- Image caching.

Avoid loading the entire dataset into GPU memory.

## 5.9 Early Stopping

Monitor validation Macro-F1.

Save the best checkpoint.

Do not automatically use the last checkpoint.

Stop when validation performance consistently deteriorates or no longer improves.

## 5.10 Reproducibility

Set random seeds.

Record:

- Python version.
- PyTorch version.
- CUDA version.
- GPU.
- Dataset version.
- Dataset split.
- Training configuration.
- Random seed.
- Model checkpoint.
- Training duration.

---

# PART 6 — FACIAL EXPRESSION MODEL OUTPUT

## 6.1 Output Structure

The Face Model must return a structured result.

Example:

```python
{
    "expression": "happy",
    "confidence": 0.91,
    "probabilities": {
        "angry": 0.01,
        "disgust": 0.00,
        "fear": 0.01,
        "happy": 0.91,
        "sad": 0.02,
        "surprise": 0.02,
        "neutral": 0.03
    },
    "face_quality": 0.95,
    "valid": True,
    "timestamp": 2.40
}
```

Use actual model probabilities.

Do not fabricate confidence values.

## 6.2 Unknown Class

The inference system must support an `unknown` or `uncertain` state.

Use this when:

- No face is detected.
- The face is occluded.
- The face is too small.
- Image quality is poor.
- Prediction confidence is insufficient.
- The prediction is unstable.

Do not force a seven-class prediction when the input is unreliable.

## 6.3 Confidence Calibration

Evaluate probability calibration.

Potential methods:

- Temperature Scaling.
- Reliability Diagrams.
- Expected Calibration Error.
- Brier Score.

Choose thresholds based on validation data.

Do not assume a softmax score of 0.9 means the model is correct 90% of the time.

## 6.4 Face Quality

If possible, calculate a face-quality score using:

- Face detection confidence.
- Face crop size.
- Blur.
- Occlusion.
- Brightness.
- Pose angle.

Do not confuse face detector confidence with expression classification confidence.

Keep these values separate.

---

# PART 7 — TEMPORAL EXPRESSION PROCESSING

## 7.1 Static Classifier First

The primary FER model should initially be a frame-level image classifier.

Do not add a large temporal architecture without appropriate sequential training data.

## 7.2 Temporal Smoothing

Implement lightweight temporal smoothing after the frame-level classifier.

Candidate methods:

- Exponential Moving Average.
- Moving Average.
- Majority Voting.
- Hysteresis.
- Minimum Prediction Duration.

Start with EMA and Hysteresis.

## 7.3 Temporal Stability

Avoid unstable transitions such as:

```text
Happy → Sad → Neutral → Angry → Happy
```

when the visible expression is approximately unchanged.

## 7.4 Hysteresis

Require a new class to exceed a calibrated confidence threshold and persist for a minimum number of observations before changing the displayed state.

Allow rapid changes when strong evidence exists.

Avoid introducing excessive lag.

## 7.5 Temporal Output

Return:

- Frame timestamp.
- Raw class.
- Raw confidence.
- Smoothed class.
- Smoothed confidence.
- Validity.
- Face-quality score.

Keep raw and smoothed predictions separate for evaluation.

## 7.6 Future Temporal Models

Only consider GRU, TCN, or temporal Transformers if a suitable video dataset with reliable temporal labels becomes available.

Evaluate whether temporal learning provides measurable gains beyond lightweight smoothing.

Do not assume that a temporal model is inherently superior.

---

# PART 8 — NON-CONFLICT ARCHITECTURE

## 8.1 Absolute Independence

The Face Branch must operate independently from the existing Sign Branch.

The original Sign Model must remain functional when the Face Model is:

- Disabled.
- Unavailable.
- Slow.
- Incorrect.
- Uncertain.
- Crashed.

The Sign Model must never depend on Face Model output to generate its primary prediction.

## 8.2 Separate Input Pipelines

Preferred architecture:

```text
                  ORIGINAL VIDEO
                         |
              +----------+----------+
              |                     |
              v                     v
       SIGN INPUT PIPELINE    FACE IMAGE PIPELINE
              |                     |
              v                     v
       EXISTING LANDMARK       FACE DETECTION
          EXTRACTION                 |
              |                     v
              v                 FACE CROP
       EXISTING SIGN                |
           MODEL                    v
              |                FACE CLASSIFIER
              v                     |
       SIGN PREDICTION               v
                                EXPRESSION
                                  OUTPUT
              |                     |
              +----------+----------+
                         |
                         v
                 OUTPUT INTEGRATION
                         |
                         v
                 DOWNSTREAM SYSTEM
```

The two pipelines must remain independent.

## 8.3 Do Not Modify Sign Inputs

Never:

- Replace face landmarks with FER predictions.
- Remove face landmarks.
- Alter facial coordinates.
- Modify hand coordinates.
- Modify body coordinates.
- Change Sign Model normalization.
- Change Sign Model sequence length.
- Change Sign Model input frame rate.
- Change Sign Model temporal alignment.
- Insert expression labels into Sign Model inputs without approval.

## 8.4 Do Not Modify Sign Predictions

The Face Branch must never:

- Override a Sign Prediction.
- Change a Gloss.
- Change a Sign Class.
- Delete a Sign Segment.
- Change Sign Confidence.
- Select another Sign Candidate.
- Trigger a Sign Correction.

The Face Branch may contribute auxiliary information to downstream output generation only.

## 8.5 Shared Face Detector

If both branches use the same face detector, evaluate whether the detector can be shared safely.

A shared detector may reduce computation.

However:

- The Sign Branch must preserve its existing facial landmarks.
- The Face Branch must use a separate crop.
- Detector failure must not remove Sign input.
- Face crop resizing must not modify the original frame.
- Face tracking must not alter Sign timestamps.

Prefer reusing an existing detection result only if it does not change existing behavior.

## 8.6 Separate Data Structures

Use separate data objects.

Example:

```python
sign_result = {
    "prediction": "EXISTING_SIGN_OUTPUT",
    "confidence": 0.94,
    "timestamp": 2.40
}

face_result = {
    "expression": "happy",
    "confidence": 0.82,
    "valid": True,
    "timestamp": 2.40
}
```

Do not merge the two into one ambiguous classification label.

## 8.7 Late Fusion Only

The initial integration must use late fusion.

Face information may be attached to the downstream representation after Sign Recognition has completed.

It must not influence the Sign Recognition decision.

## 8.8 Optional Speech Integration

The downstream Speech system may use the expression result to select an appropriate speaking style or prosody.

However, expression must be treated as auxiliary information.

Do not automatically equate facial expression with the intended emotion of the entire utterance.

The system should preserve the Sign Model's recognized linguistic meaning.

## 8.9 Graceful Degradation

If Face Recognition fails:

- Continue Sign Recognition.
- Return the Sign output normally.
- Set expression to unknown.
- Log the Face Branch failure.
- Do not stop the video pipeline.

If Sign Recognition fails:

- The Face Branch may still return an expression result if operational.
- Do not fabricate a Sign output.

---

# PART 9 — SIGN NON-INTERFERENCE EVALUATION

## 9.1 Critical Requirement

Non-conflict must be measured experimentally.

Do not claim that the branches are independent merely because the architecture diagram shows separate boxes.

## 9.2 Baseline Evaluation

Evaluate the original Sign Model without the Face Branch.

Record:

- Accuracy.
- Macro-F1.
- Sequence-level metrics.
- WER or CER where applicable.
- Segment detection metrics.
- Inference latency.
- GPU memory.
- CPU memory.

Use a fixed evaluation dataset.

## 9.3 Integrated Evaluation

Run the same videos through:

1. Sign Model alone.
2. Sign Model with Face Branch enabled.

Compare the outputs.

The Sign Model must receive identical input data in both conditions.

## 9.4 Output Invariance Test

For a strictly independent implementation, the Sign Model's output should be identical when the Face Branch is enabled or disabled, provided the Sign input and execution settings are unchanged.

Test:

- Same frame sequence.
- Same timestamps.
- Same landmarks.
- Same preprocessing.
- Same Sign Model weights.
- Same inference configuration.

Compare raw outputs.

Where deterministic inference is supported, compare output tensors or class sequences directly.

If nondeterministic operations exist, compare with an appropriate numerical tolerance and document the source of variation.

## 9.5 Face Failure Test

Simulate:

- No face detected.
- Face crop failure.
- Low face confidence.
- Face classifier exception.
- Face inference timeout.
- Face GPU memory error.

The Sign Branch must continue working.

## 9.6 Face Occlusion Test

Test videos where the face is partially occluded.

Examples:

- Hand in front of the mouth.
- Hand near the eyes.
- Hand near the forehead.
- Face partially outside the frame.

Evaluate:

- Face detection failure.
- Expression prediction reliability.
- Sign prediction stability.

Do not remove the original facial landmarks from the Sign Model.

## 9.7 Facial Grammar Test

Create a targeted evaluation set containing signs that use facial expressions as linguistic components.

Examples should be identified from the existing Sign Model's supported vocabulary and the project's actual data.

Evaluate:

- Signs with raised eyebrows.
- Signs with lowered eyebrows.
- Signs involving mouth shape.
- Signs involving head movement.
- Signs involving eye movement.
- Signs involving combined facial and manual movements.

The purpose is to ensure that the separate FER system does not cause linguistic facial information to be lost or misinterpreted.

## 9.8 Expression Variation Test

Use the same sign sequence under different facial expressions where possible.

The Sign Model must preserve the intended sign prediction.

The Face Model may return different expression labels.

Do not require the Face Model and Sign Model to agree.

## 9.9 Timestamp Misalignment Test

Introduce controlled timestamp offsets into the Face Branch.

Verify that:

- Sign predictions remain unchanged.
- Expression timestamps remain identifiable.
- Downstream fusion does not incorrectly assign an expression to an entire sign sequence.

## 9.10 Non-Interference Metrics

Calculate:

```text
Sign F1 Difference =
Integrated Sign F1 - Baseline Sign F1
```

Also calculate:

- Sign Accuracy Difference.
- Sign Macro-F1 Difference.
- Sign WER Difference.
- Sign CER Difference.
- Sign Segment Recall Difference.
- Sign Prediction Agreement.
- Sign Inference Latency Difference.

The preferred target is:

- No change in Sign Model input.
- No change in Sign Model weights.
- No change in Sign Model preprocessing.
- 100% prediction agreement under deterministic identical-input tests.
- No measurable Sign quality degradation in integrated evaluation.

If numerical variation is caused by shared GPU scheduling or nondeterminism, investigate it separately.

Do not incorrectly attribute hardware timing variation to model interference.

---

# PART 10 — FACIAL EXPRESSION EVALUATION

## 10.1 Primary Metrics

Calculate:

1. Accuracy.
2. Macro-F1.
3. Weighted-F1.
4. Balanced Accuracy.
5. Unweighted Average Recall.
6. Per-class Precision.
7. Per-class Recall.
8. Per-class F1.
9. Confusion Matrix.

Macro-F1 and UAR must be treated as primary classification metrics.

## 10.2 Class-Level Evaluation

Inspect the performance of every class separately.

Identify commonly confused categories.

Examples:

- Fear vs Surprise.
- Sad vs Neutral.
- Angry vs Disgust.
- Happy vs Surprise.

Do not assume that confusion patterns are identical across datasets.

## 10.3 Calibration Metrics

Evaluate:

- Expected Calibration Error.
- Brier Score.
- Reliability Diagram.
- Confidence Distribution.

Calibrate thresholds on validation data.

## 10.4 Robustness Evaluation

Test:

- Low lighting.
- Bright lighting.
- Blur.
- Partial occlusion.
- Head rotation.
- Different face sizes.
- Different camera distances.
- Different skin tones.
- Different age groups where represented appropriately in the evaluation data.
- Different camera qualities.

Do not make demographic performance claims without sufficient representative evaluation data.

## 10.5 Real-World Video Evaluation

FER2013 is not sufficient to prove real-time video performance.

Create or collect a small separate video evaluation set.

Annotate visible facial expression categories with appropriate uncertainty.

Use frame-level or segment-level annotation.

Avoid assuming that every frame has a clearly identifiable expression.

Evaluate on actual webcam or project video input.

## 10.6 Temporal Metrics

Evaluate:

- Prediction stability.
- Transition delay.
- False transition rate.
- Prediction flicker rate.
- Average expression segment duration.
- Unknown prediction rate.

Compare raw predictions with temporally smoothed predictions.

Ensure smoothing does not hide genuine expression changes.

## 10.7 Performance Metrics

Measure:

- Face detection latency.
- Face crop latency.
- Model inference latency.
- Total Face Branch latency.
- FPS.
- GPU memory.
- CPU memory.
- Model size.
- Cold-start time.
- Warm inference time.

Measure both standalone and integrated execution.

---

# PART 11 — REAL-TIME INFERENCE

## 11.1 Real-Time Requirement

The Face Branch must be suitable for real-time video processing.

Do not assume that processing every frame is necessary.

Investigate frame sampling.

For example, the Face Branch may run at a lower frame rate than the Sign Branch if this does not compromise expression recognition.

The Sign Branch must retain its original frame rate and sequence processing behavior.

## 11.2 Frame Sampling

Evaluate:

- Every frame.
- Every 2 frames.
- Every 3 frames.
- Adaptive inference.

Select a sampling rate based on measured expression stability and latency.

## 11.3 Face Tracking

If face tracking is used, maintain stable crop identity.

Do not allow tracking failure to affect Sign landmark tracking.

## 11.4 Asynchronous Execution

Investigate whether the two branches can run asynchronously.

Potential design:

- Sign inference worker.
- Face inference worker.
- Timestamp-based result queue.
- Downstream integration worker.

Avoid blocking the Sign Branch while waiting for Face inference.

## 11.5 Resource Isolation

Measure whether concurrent inference increases:

- GPU contention.
- Memory pressure.
- CPU overhead.
- End-to-end latency.

If necessary, use:

- Lower Face inference frequency.
- Smaller batch size.
- CPU inference.
- Separate scheduling.
- Lightweight quantization.

Do not sacrifice Sign performance for minor Face improvements.

## 11.6 Quantization

Investigate:

- FP16.
- Dynamic INT8.
- Static INT8.
- ONNX Runtime.
- TensorRT if compatible.

Quantization is optional.

Evaluate accuracy before and after quantization.

Do not deploy a quantized model without checking per-class performance.

---

# PART 12 — INTEGRATION INTERFACE

## 12.1 Separate Inference Modules

Implement a dedicated Face inference module.

Suggested interface:

```python
class FaceExpressionRecognizer:

    def load_model(self):
        ...

    def predict(self, frame):
        ...

    def predict_batch(self, frames):
        ...

    def reset_temporal_state(self):
        ...
```

Adapt to the existing project's coding conventions.

## 12.2 Sign Interface Preservation

Do not change the existing Sign Model's public interface unless absolutely necessary.

If integration requires a wrapper, the wrapper must preserve existing behavior.

## 12.3 Integrated Output

Suggested structure:

```python
{
    "timestamp": 2.40,

    "sign": {
        "prediction": "EXISTING_SIGN_OUTPUT",
        "confidence": 0.94
    },

    "face": {
        "expression": "happy",
        "confidence": 0.82,
        "valid": True,
        "quality": 0.95
    },

    "integration": {
        "mode": "independent_late_fusion",
        "sign_modified": False
    }
}
```

Do not use placeholder predictions in the actual implementation.

## 12.4 API Separation

Prefer separate internal interfaces:

```text
SignRecognizer.predict(...)
FaceExpressionRecognizer.predict(...)
```

Then integrate their outputs through a downstream coordinator.

The coordinator must not modify Sign Model results.

## 12.5 Failure Isolation

Implement exception handling around the Face Branch.

A Face inference exception must not terminate Sign inference.

Log the error with a timestamp.

Return an unknown expression result.

---

# PART 13 — PROJECT STRUCTURE

Create a self-contained Face-Lab implementation.

Suggested structure:

```text
Face-Lab/
│
├── configs/
│   ├── model.yaml
│   ├── dataset.yaml
│   ├── training.yaml
│   ├── inference.yaml
│   └── integration.yaml
│
├── data/
│   ├── manifests/
│   ├── processed/
│   └── evaluation/
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   │
│   ├── models/
│   │   ├── face_classifier.py
│   │   └── model_factory.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── trainer.py
│   │   └── losses.py
│   │
│   ├── inference/
│   │   ├── face_detector.py
│   │   ├── face_cropper.py
│   │   ├── recognizer.py
│   │   └── temporal_smoother.py
│   │
│   ├── evaluation/
│   │   ├── classification_metrics.py
│   │   ├── robustness.py
│   │   ├── calibration.py
│   │   └── non_interference.py
│   │
│   ├── integration/
│   │   ├── face_branch.py
│   │   ├── sign_adapter.py
│   │   └── output_coordinator.py
│   │
│   └── utils/
│
├── checkpoints/
│   ├── best_model.pth
│   └── final_model.pth
│
├── reports/
│   ├── dataset_profile.md
│   ├── architecture_decision.md
│   ├── training_report.md
│   ├── evaluation_report.md
│   ├── non_interference_report.md
│   └── production_readiness.md
│
├── experiments/
│   ├── logs/
│   ├── metrics/
│   └── samples/
│
├── scripts/
│
├── tests/
│   ├── test_face_model.py
│   ├── test_sign_invariance.py
│   ├── test_failure_isolation.py
│   └── test_integration.py
│
├── requirements.txt
└── README.md
```

Adjust the structure to match the existing project.

Do not duplicate the existing Sign Model implementation.

---

# PART 14 — SOFTWARE TESTING

## 14.1 Unit Tests

Test:

- Dataset loading.
- Image preprocessing.
- Label mapping.
- Model initialization.
- Model forward pass.
- Checkpoint loading.
- Face crop extraction.
- Confidence calculation.
- Temporal smoothing.

## 14.2 Integration Tests

Test:

- Face Branch standalone.
- Sign Branch standalone.
- Both branches running simultaneously.
- Face Branch disabled.
- Face Branch failure.
- Sign Branch failure.
- Missing face.
- Invalid frame.
- Timestamp synchronization.

## 14.3 Regression Tests

Run the original Sign Model before and after integration.

Ensure its input preprocessing and output behavior remain unchanged.

## 14.4 Reproducibility Tests

Verify that:

- The model checkpoint can be reloaded.
- The same input produces reproducible output where deterministic execution is supported.
- Configuration files are sufficient to reproduce inference.
- Class mappings are consistent.

---

# PART 15 — REQUIRED DELIVERABLES

## 15.1 Research

Produce:

- `reports/dataset_profile.md`
- `reports/architecture_decision.md`
- `reports/training_report.md`
- `reports/evaluation_report.md`
- `reports/non_interference_report.md`
- `reports/production_readiness.md`

## 15.2 Model

Produce:

- Best checkpoint.
- Model configuration.
- Label mapping.
- Preprocessing configuration.
- Inference implementation.

## 15.3 Evaluation

Produce:

- Confusion Matrix.
- Classification Report.
- Accuracy.
- Macro-F1.
- Weighted-F1.
- UAR.
- Balanced Accuracy.
- Calibration metrics.
- Latency.
- FPS.
- Memory measurements.

## 15.4 Non-Conflict

Produce:

- Sign-only baseline metrics.
- Integrated metrics.
- Output invariance test.
- Face failure isolation test.
- Face occlusion test.
- Facial grammar test.
- Timestamp alignment test.

## 15.5 Production

Produce:

- Independent Face API.
- Integration adapter.
- Temporal smoother.
- Confidence handling.
- Unknown state.
- Error handling.
- README.
- Example video inference.

---

# PART 16 — ACCEPTANCE CRITERIA

## 16.1 Face Classification

The model must demonstrate measurable performance on a held-out test set.

Prioritize Macro-F1, UAR, and per-class recall.

Do not claim success based on Accuracy alone.

## 16.2 Real-Time Performance

Demonstrate actual inference performance on the target hardware.

Report FPS and latency separately for:

- Face detection.
- Face classification.
- Full Face Branch.
- Integrated Sign + Face execution.

## 16.3 Non-Conflict

This is a mandatory acceptance condition.

The Face Branch must not alter:

- Sign Model weights.
- Sign Model inputs.
- Sign Model preprocessing.
- Sign Model output decisions.

The Sign Model must continue operating when the Face Branch fails.

## 16.4 Output Reliability

The Face Model must support:

- Confidence.
- Unknown state.
- Face-quality information.
- Timestamp.
- Temporal smoothing.

## 16.5 Production Readiness

The model must be independently loadable and testable.

It must not require the Sign Model to be loaded to perform standalone expression inference.

---

# PART 17 — EXECUTION PLAN

## Phase 1: Inspect Existing System

Inspect the project and the Sign Model.

Do not train yet.

Deliver an architecture and compatibility report.

## Phase 2: Profile FER2013

Inspect and validate the dataset.

Create the dataset report.

## Phase 3: Select Architecture

Compare candidate architectures at a lightweight level.

Choose the most suitable primary model.

## Phase 4: Implement Data Pipeline

Implement preprocessing, augmentation, dataset splitting, and class balancing.

## Phase 5: Train FER Model

Fine-tune the selected pretrained architecture.

Use validation-based checkpoint selection.

Avoid unnecessary full training runs.

## Phase 6: Evaluate FER Model

Calculate classification, calibration, and efficiency metrics.

## Phase 7: Implement Temporal Processing

Add lightweight temporal smoothing.

Measure latency and stability.

## Phase 8: Implement Independent Face Branch

Use separate frame cropping and independent model inference.

Preserve the original Sign pipeline.

## Phase 9: Non-Interference Testing

Run the complete test suite.

Do not proceed to final integration if Sign behavior has changed unexpectedly.

## Phase 10: Production Packaging

Prepare the inference interface and documentation.

---

# PART 18 — IMPORTANT WARNINGS

1. Do not call this project Face Identity Recognition. It is Facial Expression Recognition.
2. Do not assume facial expression equals internal emotion.
3. Do not assume FER2013 represents real-time sign-language video.
4. Do not remove facial landmarks from the Sign Model.
5. Do not overwrite the existing Sign Model.
6. Do not use Face predictions to modify Sign predictions.
7. Do not merge emotion labels into Sign classification inputs.
8. Do not assume a high Accuracy means balanced performance.
9. Do not ignore minority classes.
10. Do not use the test set for hyperparameter tuning.
11. Do not introduce a large temporal model without sequential training data.
12. Do not use excessive image augmentation.
13. Do not force an emotion prediction when face quality is poor.
14. Do not treat confidence as calibrated probability without testing.
15. Do not block Sign inference while waiting for Face inference.
16. Do not change the Sign Model frame rate.
17. Do not change the Sign Model sequence length.
18. Do not alter existing landmark normalization.
19. Do not assume that two branches are independent without testing.
20. Do not claim non-conflict without regression tests.
21. Do not claim real-time performance without measuring it.
22. Do not claim improved sign recognition because of adding a Face Model.
23. Do not fabricate evaluation results.
24. Do not implement unsupported controls.
25. Do not modify unrelated components.

---

# PART 19 — FINAL ENGINEERING INSTRUCTIONS

Work autonomously.

Inspect the existing project before implementation.

The Sign Model already exists. Your responsibility is to discover how it works and design a compatible independent Face Branch.

Do not ask the user to provide architecture information that can be discovered from the existing source code and checkpoints.

Prioritize:

1. High classification performance.
2. Efficient training.
3. Low inference latency.
4. Stable expression predictions.
5. Independent model execution.
6. Zero intentional modification to the Sign pipeline.
7. Reliable downstream integration.

Choose the simplest architecture that meets the performance requirements.

Prefer pretrained MobileNetV3-Large as the starting point.

Use EfficientNet-B0 or ResNet18 only if a meaningful performance advantage is demonstrated.

Do not introduce complex temporal architectures without suitable temporal training data.

The most important requirement is non-interference.

The final implementation must demonstrate that the Face Model can be enabled, disabled, or fail without changing the existing Sign Model's prediction behavior under identical Sign inputs.

**FINAL OBJECTIVE: Build a lightweight, high-performance, real-time Facial Expression Recognition subsystem using FER2013, and integrate it as an independent auxiliary branch alongside the existing Sign Language Recognition system, preserving all facial information required for Sign Language grammar and preventing any degradation or interference with Sign Recognition.**

---




จากด้านบนเป็นไอเดียหรือแนวทางการในการพัฒนาเบื้องต้น คุณสามารถ Edit, Adjust หรือเพิ่มเติมได้ โดยอิงจาก ข้อมูลและโมเดล รวมถึง Overall Project เราเป็นหลัก (ยังไงก็ได้ให้ Performance ดีขึ้น และไม่เกิด Conflict ขึ้น)



และ File Structure ของตัว Face-Lab/ จริงๆไม่ต้องซับซ้อนมาก พยายามทำให้ Clean สุด (ถ้าอิงตามด้านบนจะเยอะเกินไป)



โดยโมเดลที่อยากให้ทดสอบ มี 2 ตัวคือ

- MobileNet
- EfficientNet 


ตามที่บอก และดู Metrics ที่ได้พร้อมหยิบตัวที่ดีที่สุดมาใช้ (หยิบตัว Performance สูงมาใช้ก่อน เบื้องต้น)


ทำ figures/ ที่จะรวบรวมตัว architecture plot, visualize ไว้สำหรับตัว architecture (สถาปัตยากรรมนอกจากตัวโมเดลเองแล้ว มีส่วนที่เอาไปต่อด้วย)


และอยากให้สร้าง Face-lab.ipynb สำหรับ การลองเทสโมเดลทำนายตัว FER2013, ทำนายบน วิด๊โอคลิป (ผ่าน data_test/tean-drive/Jack) และ ลองเปิดกล้อง web cam แบบ real-time ให้ฉันทดสอบว่า มันสามารถ จับแบบ real-time และถูกต้องได้หรือไม่


ส่วน cell ด้านบนๆ ดึง architecture หลักที่ต้องใช้อธิบายและเกี่ยวข้องมาจาก figures/ มาใส่ไว้ด้วย


และ cell ล่างๆใช้เป็น evaluation จาก lab ทดลอง metrics ที่เกี่ยวข้องจาก 2 โมเดลที่ได้เลือกไป โดยเฉพาะ accuracy, latency (inference, real-time, etc.)


สำหรับการเขียน markdown, comment ใน .ipynb นี้ไม่ต้องเยอะ และเอาสรุปสำคัญๆพอ (เดี๋ยวจะรกเกินไป)



การเทรนหรือ Fine-Tuning, Transfer Learning, etc.


สำหรับ lab Face ฉันให้ใช้ Local เพียวๆ (Local GPU) ผ่าน 490-Sign หรือ hugging (จาก conda env) เพื่อ cost saving



เขียนรายงานหลังเสร็จ (เขียนทุกอย่าง ตั้งแต่ออกแบบ สร้าง ทดลอง ผลลัพธ์ อื่นๆที่เกี่ยวข้อง) เป็น face-report-v1.pdf



ตัวโมเดลที่ดีที่สุด ให้ไปเก็บที่ face-model-____ (ชื่อโมเดลที่ใช้)-v1/  ใน path models/ 


models/ จะเป็นที่เรียกโมเดลหลักสำหรับการทดลองบน root หลัก



แล้วหลังเสร็จ อยากให้ทดลองบนหน้าเว็บด้วย (รันตัว docker compose ขึ้นมาแล้วทดสอบจริง เข้ากับ Sign Model และตัวอื่นๆ เพื่อดูว่าบนเว็บไม่เกิด conflict หรือ error กับส่วนอื่น) (เน้นการทดสอบจริง ว่าไม่มีอะไรผิด รวมถึงการแสดงผล UI ด้วยเช่นกัน ไม่ควรมี conflict, error, หรือข้อมูลไม่ตรง หรือไม่แสดงผล)



รอบนี้เป็นการ Focus ตัว Face เป็นหลัก และ การต่อเข้ากับระบบหลัก (Sign รวมถึง TTS ที่จำนำ Text และ Emotion ไปใช้ต่อ (เช็คให้ดีว่า TTS รับอะไรเป็น Input))




---



=== Dev Constraint ===


===== Maximize Cost =====
$0 Only dont use Cloud Computing 

Using Only Local GPU (RTX 2050 4GB) และระวังเรื่อง OOM ด้วย



การเขียนโค้ดหลีกเลี่ยงการทำ ไฟล์_ตามด้วย version

เช่น main_v1.py คุณทำ main.py ไปเลย เวลามี version ใหม่แค่เขียนทับพอ จะได้ไม่เสียเวลาในการกลับมา clear workspace ไมงั้น code workspace ฉันมันจะรกขึ้นเรื่อยๆตาม version (ฉันจริงจังเรื่องนี้)



หลังจากเสร็จทุกอย่างสิ่งที่คุณต้องเช็คเพิ่มเติมมีดังนี้
- Code Workspace (บน local ฉัน): clear ทุกอย่างที่ไม่เกี่ยวข้องกับ main หลักออกไป หรือ script ที่คุณสร้างขึ้นแล้วมันไมไ่ด้ใช้ แล้วลบออกให้หมด ไม่ต้องเหลือทิ้งไว้
- Background Process: เช็คให้ดีว่าคุณรันอะไรทิ้งไว้หรือไม่ จะอะไรก็แล้วแต่ ควร clear ออกให้หมด เพื่อไม่ให้กิน memory บนเครื่องของฉัน




หลังจาก build อะไรเสร็จ อยากให้คุณลองทดสอบเองแบบจริงๆด้วย ว่าควรต้องปรับปรุงอะไรต่อ ไม่ใช่รอให้ฉันทดสอบเองอย่างเดียว


(อย่าลืมเช็คเรื่อง Docker, docker compose ด้านความพร้อม (บน root หลักของระบบว่าไม่เกิด conflict))


การ optimize ควรดีขึ้น และไม่แย่ลงจากของเก่าล่าสุด และสามารถปิด GAP หรือ จัดการปัญหาด้านบนได้ หากยังไม่ได้ optimize ทำไปเรื่อยๆจนกว่าจะดีแบบ The Best



เขียนรายงานบนช่องแชท ทุก 25%, 50%, 75%, 100% ว่าถึงไหนแล้ว เหลืออะไร ที่ต้องทำ เหลืออีกเท่าไร จาก overall เพื่อให้ฉันสามารถติดตามได้



(รอบนี้ทำงานบน main branch)


Face-Lab/ เป็น path/folder หลักสำหรับการพัฒนาตัว Face Model เป็นหลัก เพื่อไม่ให้ workspace รกเกินและกระทบกับ root หลักของระบบ ทุกอย่างจึงทำบนนี้ เพียงแต่การเรียกใช้จริงผ่าน Main System เราเรียกผ่าน models/


---


เน้นย้ำอีกครั้ง 3 เรื่อง

1. Face Model
2. Conflict Sign Model
3. Overall System  
    - ไม่ conflict กับ Sign Model
    - ต่อเข้ากับ TTS เดิมได้ โยไม่ติดขัดหรือ Error


เพราะฉะนั้น นอกจากทดสอบโมเดล Face ปกติ แล้วคุณยังต้องทดสอบกับ Overall System อีกด้วย


---

Go Build It.
