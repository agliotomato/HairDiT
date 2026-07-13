## matte 제외 기존 코드와 논문과의 불일치

### 1. L_flow 정규화 방식

#### 논문
`L_flow = sum(m̄ · error) / sum(m̄)`
- 헤어 영역 픽셀 수로 정규화
- → 헤어 영역의 평균 오차


#### 실제 코드
`L_flow = sum(m̄ · error) / (B × C × H × W)`
- 전체 텐서 크기로 정규화
- → 전체 평균 오차


**기호**
- `m̄` : 64×64 hair matte (헤어=1, 배경=0)
- `error` : `(v_pred - v_target)^2`
- `B` : Batch size
- `C` : Channel 수 (16)
- `H`, `W` : Feature map 크기 (64×64)


---
### 2. L_edge 구현 방식

#### 논문

* 스케치의 실제 RGB 값을 가중치로 사용

  * 스케치의 RGB 크기와 채널별 값이 loss에 반영됨
  * 각 채널의 loss가 최종 `L1 norm`에서 누적됨
* VAE로 복원한 RGB 이미지에 gradient 연산을 적용
* 식 전체에 `L1 norm` 적용

`L_edge = sum(abs(per_element_loss))`

* `s * m`이 0이 아닌 영역, 즉 matte 내부의 스케치 영역이 loss에 기여
* 별도의 평균 정규화 없이 모든 위치와 채널의 절댓값을 합산

**결과**

* 수식상 gradient 크기가 `1`일 때 loss가 최소
* gradient가 `1`을 초과하면 `abs(1 - gradient)`가 다시 증가하므로 loss도 증가할 수 있음


#### 실제 코드

* 스케치 RGB 채널 중 최댓값이 `0.1`보다 큰지만 확인하여 `0/1 stroke mask`로 변환

  * threshold를 넘은 stroke는 RGB 값이나 색상과 관계없이 동일한 가중치 `1`을 가짐
* VAE 복원 이미지의 RGB 3채널을 평균하여 1채널 grayscale로 변환
* grayscale 이미지에 Sobel 연산을 적용하여 edge 계산
* Sobel gradient를 `[0, 1]` 범위로 clamp
* 전체 공간 크기 `B * H * W`로 평균

`L_edge = sum(per_pixel_loss) / (B * H * W)`

* stroke가 없는 위치의 loss는 `0`이지만, 해당 위치도 평균의 분모 `B * H * W`에는 포함됨

**결과**

* edge가 `1` 이상이면 clamp 후 모두 `1`이 되어 loss가 `0`
* 충분히 강한 edge끼리는 차이를 구분하지 않음
* stroke 영역이 전체 이미지에서 차지하는 비율이 작을수록 최종 loss도 작아짐

-------

## 결정해야 할 사항

### joint 학습의 epoch·loss 구성

**기존 커리큘럼 학습** — unbraid → braid 순차, 각 40 epoch

* phase1 (unbraid) : `Loss = 1.0·Flow + 0.1·LPIPS`  — LPIPS는 학습 30% 이후 활성
* phase2 (braid)   : `Loss = 1.0·Flow + 0.1·LPIPS + 0.05·Edge`  — LPIPS 즉시 활성, edge loss 포함

**joint 방식 현황**
* 데이터 구분 없이 braid+unbraid 합쳐서(braid 데이터는 데이터 증강) phase1, 2돌림
* 이전에 joint를 어떤 epoch·loss 기준으로 돌렸는지 **기록이 남아있지 않음**

**선택 필요 (둘 중 택1)**
* **(A) 고정 epoch** : 커리큘럼 학습처럼 phase1 40 + phase2 40으로 고정해두고 학습
* **(B) 적응적 중단** : 학습을 중간중간 확인 → phase1이 충분히 학습되면 phase2로 넘어가고, phase2도 결과가 좋을 때 중단

-------------------

## 수정한 사항
### 스케치 컬러 (dark-stroke floor)

스케치 컬러는 `[0,1]` float. 대표색이 너무 어두우면 배경 `(0,0,0)`과 구분되지 않으므로,
조건을 만족할 때만 최소색(floor)으로 끌어올린다. 

#### 기존 코드

* 최소색 = `28/255` → `(28,28,28)`
* 조건: 대표색의 max 채널 `< 28/255`
* 적용: 조건 만족 시 색 전체를 `(28,28,28)`로 **통째 교체**

`sampled_color = (28,28,28)/255   if  max(sampled_color) < 28/255`


#### 수정 코드

* 최소색 = `[7, 7, 10] / 255`
* 조건: 대표색의 max 채널 `< 10/255` (floor의 max 채널)
* 적용: `torch.maximum(sampled_color, [7,7,10]/255)` — **채널별 상향**

`sampled_color = max(sampled_color, [7,7,10]/255)   if  max(sampled_color) < 10/255`

**차이**

| 항목 | 기존 | 수정 |
|---|---|---|
| floor 값 | `(28,28,28)` | `(7,7,10)` (더 검정에 가까움, B만 10) |
| 조건 임계값 | `max < 28/255` | `max < 10/255` (floor보다 밝으면 미적용) |
| 적용 방식 | 통째 교체 | 채널별 `torch.maximum` |

**결과**

* 어느 채널도 어두워지지 않고, 근-검정 색의 미세 구조가 보존됨
* 예: `(9,9,9) → (9,9,10)` (기존 방식은 `(7,7,10)`로 R,G가 깎임), `(0,0,0) → (7,7,10)`
* train / infer / preprocess에 동일 규칙 적용 (대표색 선택만 train은 stochastic, infer는 mean)


---------

### matte+sketch 입력

ControlNet에 들어가는 조건 입력을 논문(PDF §3.3, Eq. 1–6)에 맞춰 **17채널 → 32채널**로 확장했다.
matte를 하나의 feature로만 넣던 것을, **① 헤어 영역 보정 신호**와 **② 명시적 위치(anchor) 신호**
두 갈래로 분리한 것이 핵심이다.

**기존 (17채널)**

* 스케치 latent에 matte 보정(16채널)을 더하고, matte를 1채널로 축소해 이어붙임
* matte 보정의 세기를 조절하는 값이 없고, 위치 정보가 1채널로만 들어감

**수정 (32채널)** — 두 신호를 채널로 이어붙여 구성

1. **스케치 latent (16채널)** : 색이 칠해진 스케치를 (학습하지 않는) VAE로 인코딩 — 색·구조가 이미 담김
2. **matte 보정 (16채널)** : matte를 작은 CNN에 통과시켜 헤어 영역에만 작동하는 보정 신호를 만들고
   스케치 latent에 더함. 이 보정은 처음엔 0에서 시작해 필요한 만큼만 학습됨
3. **위치 anchor (16채널)** : matte의 모양·위치를 담는 신호 (아래 pixel unshuffle 과정)
4. 최종 조건 = (스케치 latent + λ·matte 보정) ⊕ (위치 anchor) = **32채널**  [Eq. 6]

**learnable scale λ 란?**

* 논문 **Eq. 3**: `z_cond = z_sketch + λ · B_matte`. λ는 *"matte bias의 세기를 보정하는 learnable
  residual scale"* 로 정의됨 (PDF §3.3.1, "VAE-Sketch Base and MatteCNN Bias")
* 즉 **matte 보정을 스케치 latent에 얼마나 강하게 더할지를 학습으로 정하는 스칼라 값 하나**
* matte 보정 자체가 0에서 출발하므로, λ와 함께 필요한 만큼만 서서히 커져 원래 스케치 표현을 급격히 깨지 않음

**matte를 latent 크기로 줄이는 과정 (Eq. 4–5, 7–8)**

matte(512×512)를 단순 축소하지 않고, 8×8 국소 점유(occupancy)를 보존하도록 아래를 거친다.

* **PixelUnshuffle** : 8×8 픽셀 블록을 채널 방향으로 접음 → 위치 anchor는 이걸 16채널로 투영 (Eq. 4–5)
* **Channel averaging** : 접힌 블록을 평균 → 64×64 latent matte (Eq. 7)
* **AvgPool 2×2 → flatten** : 32×32로 줄인 뒤 펼쳐 1024개 token으로 (Eq. 8)
* 이 matte token으로 ControlNet 잔차를 헤어 영역에 게이팅 : `r̂ = r ⊙ [a·matte_tok + (1−a)]` (a=게이팅 강도, Eq. 9)

**결과**

* matte가 (i) 영역 보정, (ii) 위치 anchor 두 신호로 나뉘어 각각 기여
* 채널 구조가 바뀌어 기존 17채널 체크포인트는 로드 불가 (32채널 전용, 신규 학습 필요)
