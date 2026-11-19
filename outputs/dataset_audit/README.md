# 교차 split 유사 이미지 추가 검사

검사일: 2026-11-19 (Asia/Seoul). 원본 이미지·라벨·분할과 진행 중인 학습은 변경하지 않았다.

**eraser에서 거의 같은 배치·시점의 촬영 장면이 train–valid와 train–test에 걸쳐 있는 사례를 확인했다.** 파일이 완전히 같은 중복은 없지만, 현재 split의 평가 성능을 새로운 장면에 대한 일반화 성능으로 해석할 때 이 한계를 고려해야 한다. 이 검사로 성능 편향의 크기를 계산할 수는 없다.

## 검사 범위와 후보 수

687장(train 498, valid 116, test 73)의 교차 split **102,590쌍**을 비교했다. 다음 조건을 모두 만족하면 시각 검토 후보로 표시했다.

- pHash64 Hamming 거리 ≤ 6: 32×32 grayscale을 DCT-II 변환하고 저주파 8×8 계수를 중앙값으로 이진화했다.
- dHash128 Hamming 거리 ≤ 12: 가로 64비트와 세로 64비트의 픽셀 밝기 비교를 결합했다.
- 32×32 RGB 썸네일의 평균 절대차 ≤ 12: 각 채널의 0–255 값 기준이다.

이는 사전에 고정한 휴리스틱 조건이며 중복 판정의 정답 기준은 아니다.

| 항목 | 결과 |
|---|---:|
| 완전히 같은 파일 SHA-256의 교차 split 쌍 | 0 |
| `.rf.<hash>`만 제거한 동일 원본 파일명의 교차 split 그룹 | 0 |
| 세 유사도 조건을 만족한 후보 쌍 | 494 |
| 후보 train–valid / train–test / valid–test | 239 / 210 / 45 |
| 후보에 참여한 고유 train / valid / test 이미지 | 136 / 54 / 30 |
| train 이미지와 후보 쌍을 가진 test 이미지 | 29 |
| eraser의 동일 초 단위 파일명 그룹 중 여러 split에 걸친 그룹 | 52 |

494개 후보는 모두 eraser 출처 이미지이다. 파일명 시각은 인증된 촬영 시각이 아니며, 같은 초 단위 이름만으로 연속 프레임이나 중복을 확정하지 않았다. 다른 클래스에서 이 조건의 후보가 없다는 사실도 장면 독립성을 보장하지 않는다.

## 시각 검토: 총 18쌍

가장 유사한 후보 중 이미지가 겹치지 않는 **12쌍**을 확인했다. train–valid 6쌍, train–test 6쌍이며 사용된 이미지는 train 12장·valid 6장·test 6장이다. 모두 초록 격자 매트, 배경 물체, 지우개의 배치·시점이 거의 같다. 파일명이 같거나 같은 바이트라는 의미는 아니다.

- [상위 후보 01–06](near_duplicate_candidates_01.jpg)
- [상위 후보 07–12](near_duplicate_candidates_02.jpg)

별도로 임계값 경계에 가까운 **6쌍**을 검토했다. 같은 배경이지만 지우개 회전·위치나 시점의 차이가 분명하므로, 이들은 거의 같은 프레임의 중복으로 확정하지 않았다. 두 표본을 합친 36장 사이에 재사용한 이미지는 없다. 이 의도적 표본은 전체 후보의 중복 비율을 추정하기 위한 무작위 표본이 아니다.

- [임계값 경계 후보 13–18](near_duplicate_threshold/near_duplicate_candidates_01.jpg)
- [각 쌍의 시각 판정·설명·전체 파일명](near_duplicate_visual_review.json)

예를 들어 상위 후보 02는 다음 train–test 쌍이다. 두 지우개의 위치·각도와 매트·주변 도구가 거의 일치한다.

```text
data/train/images/eraser_20250904152030_jpg.rf.HKbMpaeub9kd2h8dNSIc.jpg
data/test/images/eraser_20250904152031 (1)_jpg.rf.EOPmaZ4De7G5JadOTZ2n.jpg
```

따라서 **494개는 유사도 후보 수이고, 12개는 직접 검토한 거의 동일한 장면·시점의 쌍 수**이다. 이 둘을 모두 확정 중복 수로 바꾸어 보고하지 않는다.

## 보존 확인과 해석 한계

새로 계산한 이미지 687장의 SHA-256과 라벨 687개의 SHA-256은 준비된 원본 전송 ZIP의 `dataset_manifest.json`과 모두 일치했다. 이미지 읽기 전후의 파일 크기·수정 시각도 동일했다. 원본을 이동·삭제·재분할하지 않았다.

현재 진행 중인 학습은 기존 데이터에 대한 baseline으로 유지한다. 이후 별도 개선 실험을 수행한다면 촬영 세션·장면 단위로 먼저 묶어 split을 정하고, 새 장면의 독립 test 세트로 다시 평가하는 방법을 고려할 수 있다. 현재 결과와 개선 실험을 섞지 않아야 한다.

해시는 자르기·회전·큰 시점 변화가 있는 동일 장면을 놓칠 수 있으며, 단순한 배경과 비슷한 물체는 거짓 후보를 만들 수 있다. 전체 494쌍이나 모든 장면을 시각 검토한 것은 아니다.

## 재현

저장소 루트에서 Pillow와 NumPy가 설치된 환경으로 실행한다. 학습 패키지를 import하거나 GPU를 사용하지 않는다.

```text
python scripts/audit_near_duplicates.py --output-dir outputs/dataset_audit_rerun
```

기존 감사 산출물이 있으면 덮어쓰지 않고 중단한다. 새 출력 폴더를 사용해야 한다. 자동 검사는 수치 JSON·이미지 해시·상위 12쌍과 경계 6쌍의 연락시트를 재생성한다. `near_duplicate_visual_review.json`은 이번 시각 검토 기록이며 자동으로 중복을 확정하는 기능은 아니다.

- [자동 검사 JSON](near_duplicate_audit.json)
- [전체 이미지 해시](near_duplicate_image_hashes.json)
- [경계 후보 선정 기록](near_duplicate_threshold/selected_pairs.json)
- [검사 스크립트](../../scripts/audit_near_duplicates.py)
- [원본 데이터 출처·라이선스](../../data/SOURCES.md)
