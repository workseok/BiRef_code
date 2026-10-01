# BiRef_code — BiRefNet 기반 로컬 배경 분리 파이프라인

[BiRefNet](https://github.com/ZhengPeng7/BiRefNet)(MIT 라이선스 코드)을 사용해
카메라·캡처카드·영상 파일에서 사람/피사체 마스크를 뽑고, 새 배경과 합성하는
로컬 파이프라인입니다. 클라우드 API 없이 로컬에서만 동작합니다.

> ⚠️ 본 저장소는 `matanyone2`, `Part1`, `Part2` 저장소와 **완전히 독립된**
> 코드베이스입니다. 해당 저장소의 코드를 복사하거나 import하지 않습니다.

> 🚧 현재 진행 상황: 1단계(설치)·2단계(가중치 라이선스 확인) 완료.
> `realtime_matte.py`, `composite.py`, 설치·실행 절차는 다음 단계에서 추가됩니다.

---

## 1. 가중치 라이선스 확인 결과 (먼저 읽어주세요)

### 결론 요약

- **BiRefNet 코드**: MIT 라이선스 → 상업적 이용 가능.
- **사전학습 가중치(.pth)**: 저자는 저장소 전체를 MIT로 배포하지만, 가중치는
  각자 다른 데이터셋으로 학습되었고 **그 데이터셋 중 일부는 명시적으로 상업적
  이용을 금지**합니다.
- 특히 기본값으로 쓰는 **general 가중치는 DIS5K로 학습**되었는데, DIS5K 이용약관
  ([원문 PDF](https://github.com/xuebinqin/DIS/blob/main/DIS5K-Dataset-Terms-of-Use.pdf))
  2조는 다음과 같이 적고 있습니다.

  > "The Dataset is available for non-commercial use in research or educational
  > purpose. … Without permission from the original authors, commercial use of
  > this dataset is prohibited even after copying, editing, processing or any
  > operations of this database."

- "데이터셋으로 학습한 모델 가중치"가 이 조항의 "processing"에 해당하는지는
  **법적으로 확정된 해석이 없습니다.** 따라서 general 가중치의 "완전히 제약 없는
  상업적 이용"은 **이 저장소에서 보장할 수 없습니다.** 상업 서비스에 투입하기 전
  법무 검토 또는 데이터셋 저자(DIS5K: Xuebin Qin)의 서면 허락을 받는 것을
  권장합니다.
- 일부 블로그/검색 요약에 "DIS5K는 MIT"라는 서술이 있지만, 이는 DIS **코드**
  저장소(Apache-2.0)와 **데이터셋 약관**을 혼동한 것입니다. 위 PDF가 1차 자료입니다.

### 가중치별 학습 데이터셋 표

출처: BiRefNet 공식 README의 *Model Zoo* 표 + [GitHub Release v1](https://github.com/ZhengPeng7/BiRefNet/releases/tag/v1)
의 파일 목록 (확인일: 2026-10-01, 저장소 커밋 `ebcc0bc`).
Model Zoo 표는 Google Drive 링크만 있고 파일명이 없으므로, **파일명 ↔ 표 행
대응은 이름·백본·용도로 추정한 것**입니다.

| Release 파일명 | 용도 / 백본 | 학습 데이터셋 (공식 README 기재) | 상업 이용 판단 |
| --- | --- | --- | --- |
| **`BiRefNet-general-epoch_244.pth`** ⭐ 기본값 | general / Swin-L | DIS5K-TR, DIS-TEs, DUTS-TR_TE, HRSOD-TR_TE, UHRSD-TR_TE, HRS10K-TR_TE, TR-P3M-10k, TE-P3M-500-NP, TE-P3M-500-P, TR-humans | ❌ 리스크 높음 (DIS5K 비상업, DUTS 등 불명확) |
| `BiRefNet-general-bb_swin_v1_tiny-epoch_232.pth` | general / Swin-T (경량) | 위와 동일 | ❌ 리스크 높음 |
| `BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420.pth` | general / Swin-L | DIS5K-TR, DIS-TEs | ❌ DIS5K 비상업 |
| `BiRefNet_HR-general-epoch_130.pth` | general 2048² / Swin-L | AIM-500, DIS-TR, DIS-TEs, HIM2K, PPM-100, HRS10K, Human-2k, P3M, AM-2k, HRSOD, UHRSD, Distinctions-646, BG-20k, TR-humans | ❌ DIS5K 포함 |
| `BiRefNet-matting-epoch_100.pth` | general matting / Swin-L | P3M-10k, TR-humans, AM-2k, AIM-500, Human-2k(+BG-20k), Distinctions-646(+BG-20k), HIM2K, PPM-100 | ⚠️ 불명확 (D646 등 요청 기반 배포·약관 미확인) |
| **`BiRefNet-portrait-epoch_150.pth`** | portrait matting / Swin-L | P3M-10k, humans | ⚠️ 상대적으로 낮음 (아래 참고) |
| `BiRefNet-DIS-epoch_590.pth` | 논문 벤치마크(DIS) | DIS5K-TR | ❌ DIS5K 비상업 |
| `BiRefNet-COD-epoch_125.pth` | 논문 벤치마크(COD) | COD10K-TR, CAMO-TR | ⚠️ 약관 미확인 |
| `BiRefNet-HRSOD_DHU-epoch_115.pth` | 논문 벤치마크(HRSOD) | DUTS-TR, HRSOD-TR, UHRSD-TR | ⚠️ DUTS 불명확 |
| `BiRefNet_lite-general-2K-epoch_232.pth`, `BiRefNet-general-resolution_512x512-fp16-epoch_216.pth`, `BiRefNet_dynamic-*`, `BiRefNet_HR-matting-*`, `BiRefNet_lite-matting-*`, `BiRefNet_lite-anime-*` | 파생 변형 | **공식 README 표에 미기재** | ❓ 판단 불가 |

### 데이터셋별 라이선스 확인 결과

| 데이터셋 | 확인된 이용 조건 | 확인 수준 |
| --- | --- | --- |
| DIS5K (DIS-TR/TE/VD) | 비상업적 연구·교육 목적만 허용, 가공 후에도 상업 이용 금지. 이미지 출처 Flickr | ✅ 약관 원문 확인 |
| UHRSD | MIT, 이미지 출처 Flickr/Pixabay(무료 저작권) | 🔸 2차 자료(검색 결과)만 확인 |
| schirrmacher/humans | Apache-2.0 | 🔸 2차 자료(검색 결과)만 확인 |
| P3M-10k / AM-2k / BG-20k | 저장소 README에 "Agreement (MIT License)" 표기. 단 동의서 PDF 본문은 확인하지 못함. P3M-500-NP는 얼굴 미가림 유명인 사진 | 🔸 표기만 확인 |
| DUTS | 명시적 라이선스 없음 (이미지는 ImageNet·SUN 유래로 알려짐 → 원본 이미지 저작권 불명확) | ❓ |
| HRSOD, HRS10K, COD10K, CAMO, AIM-500, Human-2k, HIM2K, PPM-100, Distinctions-646 | 약관 원문 미확인. Distinctions-646 등은 이메일 요청 기반 배포 | ❓ |

> 확인 환경 제약: 이 작업 환경에서는 huggingface.co와 일부 데이터셋 배포 사이트
> 접근이 차단되어, Hugging Face 모델 카드의 license 태그와 P3M/AM-2k/BG-20k
> 동의서 PDF 본문은 직접 열람하지 못했습니다. 위 🔸/❓ 항목은 상업 투입 전에 반드시
> 원문을 직접 확인하세요.

### 이 저장소의 선택

- 작업 지시에 따라 **기본 가중치는 `BiRefNet-general-epoch_244.pth`**(가장 범용)입니다.
- 상업 리스크를 줄여야 하고 피사체가 **사람**이라면, DIS5K·DUTS를 쓰지 않은
  **`BiRefNet-portrait-epoch_150.pth`** 가 대안 후보입니다(P3M-10k 동의서 원문
  확인이 남아 있음). 스크립트에서 가중치 파일 경로를 옵션으로 바꿀 수 있게 만들
  예정입니다.
- 공식 저장소 README에 따르면 BRIA의 RMBG-2.0(BiRefNet 구조)은 **비상업 전용**
  가중치이므로 이 저장소에서 사용하지 않습니다.

---

## 2. 설치 확인 기록 (1단계)

- 공식 저장소 `ZhengPeng7/BiRefNet` (커밋 `ebcc0bc`)를 clone.
- Python 3.11 가상환경에 `torch 2.14.1`, `torchvision 0.29.1`, `timm 1.0.30`,
  `kornia 0.8.3`, `einops 0.8.2`, `numpy 1.26.4`, `opencv-python-headless` 설치.
- `BiRefNet-general-epoch_244.pth`(약 844MB)를 GitHub Release v1에서 내려받아
  공식 `models.birefnet.BiRefNet`에 로드 → **모든 키 일치**, 1024×1024 입력 추론
  정상 동작 확인.
- 참고: GPU가 없는 CPU 환경에서는 1024×1024 한 장에 약 20초가 걸렸습니다.
  공식 README 기준 RTX 4090 FP16은 약 58ms/장이므로, **실시간 처리는 CUDA GPU가
  사실상 필수**입니다.

가중치 파일은 용량 때문에 Git에 올리지 않습니다(`weights/`는 `.gitignore` 처리).
