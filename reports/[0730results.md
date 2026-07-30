# [2026-07-26] 학습 결과 및 분석


## 요약

결과1. 
결과2. 
여젼히 노이지함. 남은 차이는 데이터셋 차이 뿐

## 결과 사진

> seed42, `data/paper`+`data/unbraid_new` 샘플 기준. epoch5/10/15/20/25/30/35/40 비교.

### gt sketch

| 파일명 | img | sketch | epoch5 | epoch10 | epoch15 | epoch20 | epoch25 | epoch30 | epoch35 | epoch40 |
|---|---|---|---|---|---|---|---|---|---|---|
| CM_1067 | <img src="../data/paper/img/CM_1067.png" width="70"> | <img src="../data/paper/sketch_gt/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch5/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch10/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch15/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch20/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch25/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch30/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch35/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch40/CM_1067.png" width="70"> |
| CM_1082 | <img src="../data/paper/img/CM_1082.png" width="70"> | <img src="../data/paper/sketch_gt/CM_1082.png" width="70"> | <img src="../outputs/0730/epoch5/CM_1082.png" width="70"> | <img src="../outputs/0730/epoch10/CM_1082.png" width="70"> | <img src="../outputs/0730/epoch15/CM_1082.png" width="70"> | <img src="../outputs/0730/epoch20/CM_1082.png" width="70"> | <img src="../outputs/0730/epoch25/CM_1082.png" width="70"> | <img src="../outputs/0730/epoch30/CM_1082.png" width="70"> | <img src="../outputs/0730/epoch35/CM_1082.png" width="70"> | <img src="../outputs/0730/epoch40/CM_1082.png" width="70"> |
| CM_1068 | <img src="../data/paper/img/CM_1068.png" width="70"> | <img src="../data/paper/sketch_gt/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch5/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch10/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch15/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch20/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch25/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch30/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch35/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch40/CM_1068.png" width="70"> |
| CM_1172 | <img src="../data/paper/img/CM_1172.png" width="70"> | <img src="../data/paper/sketch_gt/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch5/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch10/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch15/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch20/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch25/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch30/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch35/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch40/CM_1172.png" width="70"> |


### Colorful sketch

| 파일명 | img | sketch | epoch5 | epoch10 | epoch15 | epoch20 | epoch25 | epoch30 | epoch35 | epoch40 |
|---|---|---|---|---|---|---|---|---|---|---|
| CM_1067 | <img src="../data/paper/img/CM_1067.png" width="70"> | <img src="../data/paper/sketch_gt/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch5_color/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch10_color/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch15_color/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch20_color/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch25_color/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch30_color/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch35_color/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch40_color/CM_1067.png" width="70"> |
| CM_1068 | <img src="../data/paper/img/CM_1068.png" width="70"> | <img src="../data/paper/sketch/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch5_color/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch10_color/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch15_color/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch20_color/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch25_color/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch30_color/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch35_color/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch40_color/CM_1068.png" width="70"> |
| CM_1172 | <img src="../data/paper/img/CM_1172.png" width="70"> | <img src="../data/paper/sketch/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch5_color/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch10_color/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch15_color/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch20_color/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch25_color/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch30_color/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch35_color/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch40_color/CM_1172.png" width="70"> |
| braid_2625 | <img src="../data/paper/img/braid_2625.png" width="70"> | <img src="../data/paper/sketch/braid_2625.png" width="70"> | <img src="../outputs/0730/epoch5_color/braid_2625.png" width="70"> | <img src="../outputs/0730/epoch10_color/braid_2625.png" width="70"> | <img src="../outputs/0730/epoch15_color/braid_2625.png" width="70"> | <img src="../outputs/0730/epoch20_color/braid_2625.png" width="70"> | <img src="../outputs/0730/epoch25_color/braid_2625.png" width="70"> | <img src="../outputs/0730/epoch30_color/braid_2625.png" width="70"> | <img src="../outputs/0730/epoch35_color/braid_2625.png" width="70"> | <img src="../outputs/0730/epoch40_color/braid_2625.png" width="70"> |
| CM_1084_1 | <img src="../data/unbraid_new/img/CM_1084_1.png" width="70"> | <img src="../data/unbraid_new/sketch/CM_1084_1.png" width="70"> | <img src="../outputs/0730/epoch5_color/CM_1084_1.png" width="70"> | <img src="../outputs/0730/epoch10_color/CM_1084_1.png" width="70"> | <img src="../outputs/0730/epoch15_color/CM_1084_1.png" width="70"> | <img src="../outputs/0730/epoch20_color/CM_1084_1.png" width="70"> | <img src="../outputs/0730/epoch25_color/CM_1084_1.png" width="70"> | <img src="../outputs/0730/epoch30_color/CM_1084_1.png" width="70"> | <img src="../outputs/0730/epoch35_color/CM_1084_1.png" width="70"> | <img src="../outputs/0730/epoch40_color/CM_1084_1.png" width="70"> |


## 내용정리



## 1. 이슈 1

**내용**


(증거 이미지 표)

**원인분석**


**해결방안**
1. 
2. 
3. 
...

## 2. 이슈 2

**내용**
이전 학습(0720)은 phase2에서만 푸석함이 나타났는데, 이번(0725)은 phase1부터 나타남

(증거 이미지 표)


**원인분석**


**해결방안**



