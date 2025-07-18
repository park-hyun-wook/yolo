# 🧠 YOLOv8 (You Only Look Once Version 8)

### ✅ YOLOv8이란?
- **실시간 객체 탐지 분야**에서 최신 기술을 선도하는 **딥러닝 기반**의 비전 모델
- 2023년 1월 Ultralytics에서 공식 출시되었으며, **이전 버전들보다 향상된 정확도와 속도**를 자랑한다.
- 이미지 속 객체의 위치와 종류를 단 한 번의 신경망 연산으로 예측하는 **단일 단계(single stage) 탐지기**로, 영상 및 이미지 분석, 자동화 등 다양한 분야에서 폭넓게 활용

<img width="1920" height="720" alt="image" src="https://github.com/user-attachments/assets/00ac0338-8b61-4ff7-bc92-66e4699acb14" />

#### 💻 YOLOv8 간략 설명 영상
https://www.youtube.com/watch?v=Na0HvJ4hkk0&t=7s

### ✅ YOLOv8 주요 기능
- **고급 백본 및 넥 아키텍처**: YOLOv8 최첨단 백본 및 넥 아키텍처를 채택하여 특징 추출 및 객체 감지 성능이 향상되었습니다.
- **앵커 프리 스플릿 Ultralytics 헤드**: YOLOv8 앵커 프리 스플릿 Ultralytics 헤드를 채택하여 앵커 기반 접근 방식에 비해 더 나은 정확도와 효율적인 탐지 프로세스에 기여합니다.
- **최적화된 정확도-속도 트레이드오프**: 정확도와 속도 간의 최적의 균형을 유지하는 데 중점을 둔 YOLOv8 은 다양한 애플리케이션 영역의 실시간 물체 감지 작업에 적합합니다.
- **다양한 사전 학습 모델**: YOLOv8 에서는 다양한 작업 및 성능 요구 사항을 충족하는 다양한 사전 학습 모델을 제공하므로 특정 사용 사례에 적합한 모델을 쉽게 찾을 수 있습니다.


### ✅ 지원되는 작업 및 모드
- YOLOv8 시리즈는 컴퓨터 비전의 특정 작업에 특화된 다양한 모델을 제공합니다.
- 이러한 모델은 객체 감지부터 인스턴스 분할, 포즈/키포인트 감지, 방향성 객체 감지 및 분류와 같은 보다 복잡한 작업까지 다양한 요구 사항을 충족하도록 설계되었습니다.
- YOLOv8 시리즈의 각 변형은 각 작업에 최적화되어 있어 높은 성능과 정확성을 보장합니다.
- 또한 이러한 모델은 추론, 검증, 교육, 내보내기 등 다양한 운영 모드와 호환되므로 배포 및 개발의 여러 단계에서 쉽게 사용할 수 있습니다.

| 모델 | 파일 이름 | 작업 |
|---|-----|----|
|YOLOv8 | `yolov8n.pt` `yolov8s.pt` `yolov8m.pt` `yolov8l.pt` `yolov8x.pt` | [탐지](https://docs.ultralytics.com/ko/tasks/detect/) | 
| YOLOv8-seg |	`yolov8n-seg.pt` `yolov8s-seg.pt` `yolov8m-seg.pt` `yolov8l-seg.pt` `yolov8x-seg.pt` |	[인스턴스 세분화](https://docs.ultralytics.com/ko/tasks/segment/) |
| YOLOv8-pose |	`yolov8n-pose.pt` `yolov8s-pose.pt` `yolov8m-pose.pt` `yolov8l-pose.pt` `yolov8x-pose.pt` `yolov8x-pose-p6.pt` |	[포즈/키포인트](https://docs.ultralytics.com/ko/tasks/pose/) |
| YOLOv8-obb |	`yolov8n-obb.pt` `yolov8s-obb.pt` `yolov8m-obb.pt` `yolov8l-obb.pt` `yolov8x-obb.pt` |	[방향 탐지](https://docs.ultralytics.com/ko/tasks/obb/)|
| YOLOv8-cls |	`yolov8n-cls.pt` `yolov8s-cls.pt` `yolov8m-cls.pt` `yolov8l-cls.pt` `yolov8x-cls.pt` |	[분류](https://docs.ultralytics.com/ko/tasks/classify/) |

- 위 표는 YOLOv8 모델 변형에 대한 개요를 제공하며, 특정 작업에서의 적용 가능성과 추론, 검증, 훈련, 내보내기 등 다양한 작동 모드와의 호환성을 강조합니다.
- [컴퓨터 비전](https://www.ultralytics.com/glossary/computer-vision-cv)의 다양한 애플리케이션에 적합한 YOLOv8 시리즈의 다목적성과 견고함을 보여줍니다.

<img width="813" height="374" alt="스크린샷 2025-07-16 11 55 47" src="https://github.com/user-attachments/assets/ebd47694-c2e9-4b48-8136-ab282b345645" />

### ✅ 성능
#### 📝 탐지
| 모델 |	크기 (픽셀) |	mAPval 50-95 |	속도CPU ONNX(ms)	| 속도 A100 TensorRT(ms) |	매개변수(M) |	FLOPs(B)|
|-----|-------|-----|------|-------|------|-----|
| YOLOv8n |	640 |	37.3 |	80.4 |	0.99 |	3.2 |	8.7 |
| YOLOv8s |	640 |	44.9 |	128.4 |	1.20 |	11.2 |	28.6 |
| YOLOv8m |	640 |	50.2 |	234.7 |	1.83 |	25.9 |	78.9 |
| YOLOv8l |	640 |	52.9 |	375.2 |	2.39 |	43.7 |	165.2 |
| YOLOv8x |	640 |	53.9 |	479.1 |	3.53 |	68.2 |	257.8 |


### ✅ YOLO 알고리즘의 발전 배경
- YOLO 시리즈는 2015년 처음 등장한 이후, 빠른 속도와 높은 정확도를 바탕으로 객체 탐지 분야의 표준으로 자리매김해 왔습니다.
- 각 버전에서 주요 기능과 성능 면의 진보가 이루어졌으며, YOLOv8은 이러한 발전을 집대성해 최신 기술과 실용성을 결합하고 있습니다.

### ✅ 주요 특징 및 혁신점
- **실시간 처리 성능**: GPU 한 대만으로도 최첨단 속도와 정확도를 구현할 수 있습니다.
- **다양한 작업 지원**: 객체 탐지뿐 아니라 세분화(Segmentation), 분류(Classification), 키포인트 검출 등 다양한 컴퓨터 비전 작업에 활용 가능합니다
- **앵커 프리(Anchor-Free) 방식**: 기존의 앵커 기반 방식보다 하이퍼파라미터가 적어져 적용이 간편하고 다양한 객체의 비율과 크기에 더 유연하게 대응합니다
- **개선된 백본·넥 구조**: 고도화된 합성곱 신경망(CNN) 백본, 그리고 정보 통합을 위한 PANet 등 최신 구조 도입으로 다양한 크기의 객체 탐지가 한층 정교해졌습니다
- **다양한 크기 모델 지원**: YOLOv8은 ultra-tiny부터 large, extra-large까지 다양한 크기와 성능의 모델을 제공합니다. 사용자는 자신의 상황(모바일, PC, 서버)에 맞게 모델을 선택할 수 있습니다
- **학습 및 배포의 용이성**: 유연한 Python 기반 구조와 풍부한 오픈소스 리소스로 누구나 쉽게 모델을 학습시키고 실제 서비스에 배포할 수 있습니다

### ✅ 실제 코드 예시
```
# Ultralytics YOLOv8 공식 패키지 설치
# pip install ultralytics

from ultralytics import YOLO

# 모델 불러오기
model = YOLO('yolov8n.pt')  # 'n'은 nano, 's', 'm', 'l', 'x' 등 선택 가능

# 예측(추론)
results = model('test_image.jpg')

# 결과 시각화
results.show()

# 직접 학습시키고 싶을 땐 (COCO 등 포맷에 맞춘 커스텀 데이터셋)
model.train(data='dataset.yaml', epochs=50)
```

- model('이미지경로') 형태로 매우 간단하게 추론 가능
- 학습 진행시 커스텀 데이터셋에 맞춰 yaml 설정 파일 작성 필요

### ✅ YOLOv8 사용 예시
- 아래 예는 YOLOv8 객체 감지를 위한 모델감지용 예제입니다.
- PyTorch 사전 교육 `*.pt` 모델 및 구성 `*.yaml` 파일을 `YOLO()` 클래스를 사용하여 python 에서 모델 인스턴스를 생성합니다:
```
from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the YOLOv8n model on the 'bus.jpg' image
results = model("path/to/bus.jpg")
```

- YOLOv8 모델 또는 이 리포지토리의 다른 소프트웨어를 작업에 사용하는 경우 다음 형식을 사용하여 인용해 주세요:
```
@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
```

### ✅ YOLOv8의 상세 기술 구조
1. **입력과 전처리**(Input & Preprocessing)
- 이미지는 Tensor 형태(NCHW)로 변환됩니다.
- 크기 조정(Resize), 정규화(Normalization), 데이터 증강(Augmentation) 적용
2. **백본**(Backbone)
- Feature Map을 추출하는 핵심 신경망(예: CSPDarknet, EfficientNet 등)
- 입력 이미지에서 객체의 다양한 특징(Edge, Color, Texture)을 계층별로 뽑아냅니다.
3. **넥**(Neck)
- Feature Pyramid Network(FPN), PANet 등 구조로 여러 해상도의 정보 융합
- 대, 중, 소 객체 모두 잘 탐지할 수 있도록 중요한 역할 수행
4. **헤드**(Head)
- 앵커-프리(anchor-free) 구조: 박스 예측을 위한 하이퍼파라미터 최소화
- 네트워크 마지막에서 각 위치마다 객체의 종류(Class), 위치(Bounding Box), **신뢰도(Confidence Score)**를 바로 출력


### ✅ 기술적 구조 요약

| 구성 요소   | 설명                                                         |
|-------------|--------------------------------------------------------------|
| 백본(Backbone) | 고급 CNN 구조로 멀티스케일(다층) 특징들을 추출             |
| 넥(Neck)       | PANet 또는 FPN 구조를 적용, 다양한 크기 특징을 효과적으로 결합 |
| 헤드(Head)     | 앵커 프리 방식으로 Bounding Box, 클래스 확률, 신뢰도 예측    |
| 최적화(Optimization) | 데이터 증강, 개선된 손실 함수, 비지도 학습 등 최신 기술 적용 |


### ✅ 대표적 성능 지표
- **MAP**(mean Average Precision): 객체 탐지 모델의 대표적 평가 지표
- **FPS**(Frame per Second): 1초에 처리 가능한 이미지 수
- **탐지 클래스 수**: 탐지 가능한 개체 종류 수 (COCO 등 데이터셋 사용 시 약 80개)

| 모델 크기 | 	파라미터 수	| 속도(FPS) |	MAP |
|------|--------|-----------|-----------|
| YOLOv8-nano  |	약 3M |	150~200 |	40~45 |
| YOLOv8-large |	약 68M	 | 35~60	  | 54~56 |

※ 실제 결과는 하드웨어, 데이터에 따라 달라질 수 있다.

### ✅ YOLOv8 모델을 어떻게 교육하나요?
- YOLOv8 모델 훈련은 Python 또는 CLI 을 사용하여 수행할 수 있습니다. 
- 다음은 10개의 에포크에 대한 COCO8 데이터 세트에서 COCO가 사전 훈련한 YOLOv8 모델을 사용하여 모델을 훈련하는 예제입니다:
```
from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

### ✅ YOLOv8 모델을 벤치마킹하여 성능을 확인할 수 있나요?
- 예, YOLOv8 모델은 다양한 내보내기 형식에 걸쳐 속도와 정확도 측면에서 성능을 벤치마킹할 수 있습니다.
- PyTorch , ONNX, TensorRT 등을 사용하여 벤치마킹할 수 있습니다. 다음은 Python 및 CLI 을 사용하여 벤치마킹하는 예제 명령어입니다:
```
from ultralytics.utils.benchmarks import benchmark

# Benchmark on GPU
benchmark(model="yolov8n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)
```

### ✅ 활용 사례 및 응용 분야

- **자율주행**: 차량, 보행자, 도로 장애물 등을 실시간으로 탐지하여 경로 계획에 활용
- **보안 및 감시**: CCTV 장면 내 이상행동, 침입자 실시간 감지
- **의료 영상 분석**: X-ray, MRI 등에서 종양 또는 이상 징후 탐지
- **소매 및 재고 관리**: 매장 내 상품 자동 인식, 재고 모니터링
- **스포츠 분석**: 경기장 내 선수 및 공 추적
- **스마트 농업**: 작물 건강 상태 감지, 농축산 자동화
- **증강현실(AR)**: 실시간 객체 인식을 통한 사용자 경험 강화

### ✅ 모델의 장점

- 높은 정확도와 실시간 처리 성능 조화
- 다양한 디바이스(엣지, 서버)에서 손쉽게 적용 가능
- 코드 오픈소스화 및 커뮤니티 중심의 빠른 기술 발전

### ✅ 한계 및 고려점

- 대용량 모델은 고성능 하드웨어 필요
- 완벽한 일반화에는 대규모 데이터셋 학습이 중요
- 특정 상황(예: 군중 속 미세한 객체)에서는 추가 튜닝 필요

### ✅ 결론

- YOLOv8은 객체 탐지 알고리즘의 최신 버전으로, 다양한 업무와 산업 전반에 실질적인 변화를 가져오는 실시간 인공지능 비전 기술입니다. 
- 강력한 처리 성능, 다양한 작업 지원, 실용적인 오픈소스 생태계의 특징을 바탕으로 앞으로도 많은 혁신 사례가 기대되는 모델입니다.
