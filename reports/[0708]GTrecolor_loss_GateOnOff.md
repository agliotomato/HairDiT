# GT 스케치 컬러 가져오기

### 결론

HairDiT  
: RGB sketch 색을 stroke label로 사용, 같은 label 위치 중 matte 안쪽 머리색에서 평균/랜덤 선택

SketchHairSalon(SHS)  
: RGB 3ch sketch를 입력하지만 공식 경로에서는 grayscale intensity로 stroke label 생성, 같은 label 위치의 원본 이미지 색에서 평균/랜덤 선택

## 1. Training

### HairDiT

1. sketch 입력
   - RGB 3ch, 8-bit image
   - 각 channel 값 범위 0~255
   - dataloader 이후 float tensor `[0,1]`
2. stroke label 생성
   - float tensor를 uint8 `[0,255]`로 임시 변환 -> **해당 부분 굳이 임시변환 하지말고, 맨 처음 들어온 RGB 3ch, 8-bit image쓰기**
   - 변환한 값을 5-bit로 양자화
   - 같은 quantized RGB를 같은 stroke group으로 분류
3. 각 stroke group 위치에서 사용할 머리색 결정
   - 같은 stroke 위치의 `img * matte` 픽셀 중 RGB 합이 0.05 이상인 픽셀만 사용
   - 사용할 픽셀이 10개 미만인 경우 기존 sketch 색 유지 -> **사용할 픽셀 10개 미만일 경우, 기존 sketch 색을 유지하는게 맞는 방향인지 확인 필요**
   - 사용할 픽셀이 충분한 경우 1/3 확률로 랜덤 픽셀색, 2/3 확률로 평균색으로 대표색 결정(SketchHairSalon과 같은 방식)
   - 결정된 대표색으로 같은 stroke group 전체 recolor -> **해당 부분에서 최소 검정을 완전 0 이 아니라, (7, 7, 7) 정도로 clamp**
4. train 중 매 iteration에 recolor
5. 모델 입력
   - condition: recolor sketch
   - target: matte 기반 hair region
   - mask: matte


핵심 : 같은 stroke 위치의 matte 안쪽 머리색만 사용, RGB sketch 색은 stroke 구분용 label

### SketchHairSalon

1. sketch 입력
   - RGB 3ch, 8-bit image 입력
   - 내부적으로 grayscale 1ch로 변환/읽기
   - pixel intensity 범위 0~255
   - 여기서 변환된 1ch sketch는 stroke label map 역할
2. stroke label 생성
   - intensity 0은 배경 취급하여 제외
   - 같은 intensity 값을 같은 stroke group으로 분류
3. 각 stroke group 위치에서 사용할 머리색 결정
   - 같은 stroke 위치의 원본 이미지 픽셀 전체를 사용
   - 해당 픽셀 집합에서 1/3 랜덤 픽셀색, 2/3 평균색 중 하나로 대표색 결정
   - 결정된 대표색으로 같은 grayscale stroke group 전체 recolor
   - matte는 색 선택 기준보다 colored stroke canvas 역할
4. S2I train color-coding 단계에서 적용
5. 모델 입력
   - 3ch color-coded sketch
   - matte canvas 위 colored stroke
   - GT image
   - matte
   - noise

---

## 2. Inference

### HairDiT

GT recolor 옵션

1. GT 이미지와 matte, sketch 입력
2. stroke label 생성
   - RGB sketch를 uint8 `[0,255]`로 임시 변환
   - 5-bit 양자화 후 같은 quantized RGB를 같은 stroke group으로 분류
3. 각 stroke group 위치에서 사용할 GT 머리색 결정
   - 같은 stroke 위치의 `GT image * matte` 픽셀 중 RGB 합이 거의 0이 아닌 픽셀만 사용
   - 사용할 픽셀이 10개 미만인 경우 원 sketch 색 유지
   - 사용할 픽셀이 충분한 경우 평균색으로 대표색 결정
   - 결정된 평균색으로 같은 stroke group 전체 recolor
   - train과 달리 랜덤 픽셀색 선택 없음


### SketchHairSalon


1. sketch 입력
   - RGB 3ch, 8-bit image 입력 가정
   - 공식 test 경로에서는 내부적으로 grayscale 1ch로 변환/읽기
   - pixel intensity 범위 0~255
   - 여기서의 1ch sketch는 색 입력이 아니라 stroke label map 역할
2. stroke label 생성
   - intensity 0은 배경으로 제외
   - 같은 intensity 값을 같은 stroke group으로 분류
3. 각 stroke group 위치에서 사용할 머리색 결정
   - 같은 stroke 위치의 원본 이미지 픽셀 전체를 사용
   - 해당 픽셀 집합의 평균색으로 대표색 결정
   - 결정된 평균색으로 다시 color-coding
   - 최종 S2I 입력은 3ch color-coded sketch
   - 랜덤 픽셀색 선택 없음


---

# HairDiT, SHS loss 차이

### HairDiT
* joint는 phase1, 2를 그대로 진행하되, 데이터만 바꿔서(unbraid 3000장, braid 1000장, braid augment 2000장 => 총 6000장) 학습

- unbraid / Phase 1 pretrain
  - 40 epoch
  - Flow matching loss 중심
  - LPIPS는 전체 step의 30% 이후 활성화
  - edge loss 없음

```text
Loss = 1.0 * Flow
     + 0.1 * LPIPS(after warmup)
```

- braid / Phase 2 finetune
  - Phase 1 checkpoint에서 이어서 학습
  - 40 epoch
  - Flow matching loss 유지
  - LPIPS 즉시 활성화
  - sketch edge alignment loss 추가

```text
Loss = 1.0 * Flow
     + 0.1 * LPIPS
     + 0.05 * Edge
```

- loss 의미
  - Flow: matte latent 영역에서 velocity prediction MSE
  - LPIPS: decoded prediction과 GT hair region의 perceptual 차이
  - Edge: sketch stroke 위치와 생성 hair edge 정렬

### SketchHairSalon
* joint에 대한 학습 방법 설명 없음

- unbraid S2I
  - 200 epoch
  - 기본 S2I loss

```text
G loss = GAN
       + 100 * L1(fake_B, real_B)
       + 100 * VGG_perceptual(fake_B, real_B)
```

- braid S2I
  - 400 epoch
  - 기본 S2I loss + braid shape/smoothing loss

```text
G loss = GAN
       + 100 * L1(fake_B, real_B)
       + 100 * VGG_perceptual(fake_B, real_B)
       + 100 * smoothing_shape_loss
```

- shape loss
  - matte 영역 안에서 fake hair와 real hair를 smoothing한 뒤 L1 비교

- loss 의미
  - GAN: 생성 이미지가 real image처럼 보이도록 학습
  - L1: 생성 이미지와 GT image의 픽셀 단위 차이 감소
  - VGG perceptual: 생성 이미지와 GT image의 perceptual feature 차이 감소
  - smoothing shape: braid의 부드러운 형태와 구조 정렬



# Gate ON/OFF

### Stable diffusion에서 CLIP On/Off
stable diffusion에서는 학습할 때 확률적으로(10%) 캡션을 드롭해서 의도적으로 빈문자열("")을 학습함  
=> CLIP을 on/off하는 것이 아닌, 빈 문자열로 생성할 수 있는 능력을 학습한 것

### HairDiT Gate에서 ON/OFF
현재 gate 수식 : gated residual = residual * [a * matte_token + (1 - a) * 1]

HairDiT Gate에서 ON/OFF를 하려면, stable diffusion에서 빈 문자열에 대한 걸 학습한것처럼  학습 중 gate_alpha=0(off), gate_alpha=1(full on), 그리고 중간값을 함께 샘플링해 모델이 gated residual과 ungated residual 분포를 모두 보게 해야 함.