# app-for-video

YOLO26x 기반 Object Detection / Instance Segmentation 예제 코드에 비디오 입출력과
커스텀 시각화를 적용한 작업 내역입니다.

## 공통 준비

```bash
source venv/bin/activate
```

- `venv`에는 `opencv-python`(GUI 지원 포함, headless 아님)이 설치되어 있어야 합니다.
  없다면: `pip install opencv-python`
- 두 예제 모두 `--video`(`-v`)로 비디오 입력을 받고, `--save`(`-s`)를 주면 결과를
  `artifacts/python_example/.../output.mp4` (또는 `--save-dir`로 지정한 경로)에
  비디오 파일로 저장합니다. (이 부분은 공통 러너 `common/runner`에 이미 구현되어
  있던 기능입니다.)

---

## 1. Object Detection — `object_detection/yolo26x`

수정 파일: `common/visualizers/detection_visualizer.py`

### 변경 내용

- **좌/우 반전 라벨 배치**: 화면 중앙(`width/2`)을 기준으로 BBOX 중심이 왼쪽이면
  라벨을 BBOX **좌측 상단**에, 오른쪽이면 **우측 상단**(라벨 우측 끝을 BBOX의
  `x2`에 맞춤)에 표시합니다. 화면 중앙 쪽으로 라벨이 몰려 겹치는 것을 방지합니다.
- **라벨 배경 여백 정리**: 텍스트 아래쪽에 불필요하게 남던 여백을 제거(배경
  사각형 하단을 baseline 기준으로 맞춤), 좌우로는 `_LABEL_PAD_X = 6`px 패딩을 줘서
  텍스트가 배경에 너무 붙지 않게 함.
- **파스텔 색상 팔레트**: 클래스별 색상을 `np.random.uniform` 완전 랜덤 대신
  HSV 기반(저채도·고명도) 파스텔 톤으로 생성, 클래스 수만큼 색상환에 고르게 분산.
- **confidence 제거**: 라벨에 `class_name: 0.xx` 대신 클래스 이름만 표시.

### 실행 예시

```bash
python object_detection/yolo26x/yolo26x_sync.py -v <video.mp4> -s
```

---

## 2. Instance Segmentation — `instance_segmentation/yolo26x_seg`

수정 파일: `common/visualizers/instance_seg_visualizer.py`,
`factory/yolo26x_seg_factory.py`, `common/runner/args.py`, `config.json`,
`yolo26x_seg_sync.py`, `yolo26x_seg_async.py`

### 변경 내용

- **마스크 전용 출력**: `show_boxes=False`로 BBOX/클래스명 텍스트를 없애고
  마스크만 표시 (`assets/yolo26-seg-out_v2.mp4` 참고 스타일).
- **파스텔 + 형광 팔레트**: 골든앵글(137.508°) 기반으로 hue를 균등 분산시키고,
  인덱스 짝/홀에 따라 채도를 교대(형광 고채도 / 파스텔 저채도)로 30색 팔레트 생성.
  트랙 ID가 팔레트 길이를 넘어가며 wraparound돼도 인접 색끼리 안 겹치도록 설계.
- **트래킹 기반 색상 고정**: 기존에 구현되어 있던 `common/trackers/iou_tracker.py`
  (IoU 기반 SORT-lite)를 이용해 `track_id`별로 색을 고정 — 같은 객체는 프레임이
  바뀌어도 색이 유지됨 (`enable_tracking=True`가 기본값).
- **깜빡임(flicker) 방지 — hold_frames**: confidence가 threshold 근처에서 오르내려
  프레임마다 마스크가 사라졌다 나타났다 하는 문제를 완화하기 위해, 트랙이 몇
  프레임(기본 `hold_frames=2`) 동안 검출을 못 받아도 마지막 마스크를 그대로 유지해서
  그리다가 그래도 안 돌아오면 지우는 로직 추가. (`enable_tracking=False`면 자동으로
  0으로 꺼짐 — 어떤 마스크가 "같은 물체"인지 알 수 없으므로.)
- **`score_threshold` 조정**: `config.json` 0.4 → 0.3으로 낮춰 경계선상 검출의
  깜빡임을 줄임.
- **`--no-tracking` 옵션 추가**: 트래킹을 끄고 싶을 때 사용. 색이 검출 순서
  기준으로 매겨져 프레임마다 바뀔 수 있으며, hold_frames도 함께 꺼짐.
  (`*_cpp_postprocess.py` 변형은 원래부터 별도 시각화 경로라 트래킹 대상 아님.)

### 실행 예시

```bash
# 기본 (트래킹 + flicker-hold 적용)
python instance_segmentation/yolo26x_seg/yolo26x_seg_sync.py -v <video.mp4> -s
python instance_segmentation/yolo26x_seg/yolo26x_seg_async.py -v <video.mp4> -s

# 트래킹 끄기
python instance_segmentation/yolo26x_seg/yolo26x_seg_sync.py -v <video.mp4> -s --no-tracking
```

### 검증

- 실제 모델(`assets/yolo26-x-seg_640x640.dxnn`)로 `assets/CL_gemini_video_watermark_removed_2.mp4`
  240프레임 전체를 async 파이프라인으로 돌려 확인.
- 프레임당 "채색 픽셀 수" 기준 프레임 간 변화량(깜빡임 정도)이 hold_frames 적용
  전 평균 17,820px → 적용 후 10,399px로 약 42% 감소.
- 같은 의자/사람의 색이 여러 프레임에 걸쳐 유지되는 것을 직접 프레임 비교로 확인.
