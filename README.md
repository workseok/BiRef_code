# BiRef_code — BiRefNet 기반 로컬 배경 분리 파이프라인

[BiRefNet](https://github.com/ZhengPeng7/BiRefNet)(MIT 라이선스 코드)을 사용해
카메라·캡처카드·영상 파일에서 사람/피사체 마스크를 뽑고, 새 배경과 합성하는
로컬 파이프라인입니다. 클라우드 API 없이 로컬에서만 동작합니다.

> ⚠️ 본 저장소는 `matanyone2`, `Part1`, `Part2` 저장소와 **완전히 독립된**
> 코드베이스입니다. 해당 저장소의 코드를 복사하거나 import하지 않습니다.

> 🚧 현재 진행 상황: 1단계(설치)·2단계(가중치 라이선스 확인) 완료.
> 지금 실행할 수 있는 것은 **"사진 1장 → 마스크 이미지 저장"** 동작 확인까지입니다
> ([3. 실행 방법](#3-실행-방법-지금-단계-사진-1장으로-마스크-만들기) 참고).
> `realtime_matte.py`(실시간/영상), `composite.py`(배경 합성)는 다음 단계에서 추가됩니다.

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

---

## 3. 실행 방법 (지금 단계: 사진 1장으로 마스크 만들기)

아래 절차는 **공식 BiRefNet 코드를 수정하지 않고 그대로** 사용합니다. 끝까지
따라 하면 내 사진(`test.jpg`)에서 피사체만 흰색으로 표시된 마스크 이미지
(`mask.png`)가 만들어집니다. 이 단계가 성공하면 이후 추가될 실시간 스크립트를
돌릴 준비가 된 것입니다.

완료 후 폴더 모양은 다음과 같습니다.

```
BiRef_code/
├── README.md
├── BiRefNet/        ← 공식 BiRefNet 코드 (3·4단계에서 내려받음, Git에 올리지 않음)
├── weights/
│   └── BiRefNet-general-epoch_244.pth   ← 가중치 (약 844MB)
├── test.jpg         ← 내가 넣은 테스트 사진
└── mask.png         ← 실행 결과
```

> 💡 용어 설명
> - **터미널**: 명령어를 글자로 입력하는 창입니다. Windows는 "PowerShell",
>   macOS는 "터미널(Terminal)" 앱을 씁니다.
> - **가상환경(venv)**: 이 프로젝트 전용 파이썬 설치 공간입니다. 다른 프로그램과
>   라이브러리 버전이 섞이지 않게 해줍니다.
> - **가중치(weights)**: 학습이 끝난 AI 모델의 "두뇌" 파일입니다.
> - 아래 회색 상자 안의 명령은 **한 줄씩 복사 → 터미널에 붙여넣기 → Enter** 하면 됩니다.

---

### 🪟 3-A. Windows

#### 준비물
- Windows 10 또는 11 (64비트)
- 실시간 처리까지 하려면 **NVIDIA 그래픽카드(GPU)** 권장. 없어도 사진 1장 테스트는
  가능하지만 수십 초가 걸립니다.

#### 1) Python 3.11 설치
1. https://www.python.org/downloads/windows/ 에서 **Python 3.11.x**
   "Windows installer (64-bit)"를 내려받습니다.
2. 설치 첫 화면 아래쪽의 **"Add python.exe to PATH"를 반드시 체크**한 뒤
   "Install Now"를 누릅니다.

> 3.12 이상이 아니라 3.11을 쓰는 이유: 공식 BiRefNet이 `numpy<2`를 요구하는데,
> 이 조합이 3.11에서 검증되었습니다.

#### 2) Git 설치
1. https://git-scm.com/download/win 에서 내려받아 기본 설정 그대로 설치합니다.

#### 3) PowerShell 열기 + 이 저장소 내려받기
1. 시작 메뉴에서 **PowerShell**을 검색해 엽니다.
2. 아래 명령을 한 줄씩 실행합니다.

```powershell
cd ~\Documents
git clone https://github.com/workseok/BiRef_code.git
cd BiRef_code
git checkout claude/nice-faraday-2q8lvr
```

#### 4) 공식 BiRefNet 코드 내려받기
라이선스 확인 및 동작 검증을 한 버전(커밋 `ebcc0bc`)으로 고정합니다.

```powershell
git clone https://github.com/ZhengPeng7/BiRefNet.git
cd BiRefNet
git checkout ebcc0bc
cd ..
```

#### 5) 가상환경 만들기 + 켜기

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- 성공하면 줄 맨 앞에 `(.venv)`가 붙습니다.
- **"스크립트를 실행할 수 없습니다"** 오류가 나면 아래를 한 번 실행하고 `Y`로 답한 뒤
  다시 `.\.venv\Scripts\Activate.ps1`을 실행하세요.
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  ```
- PowerShell을 새로 열 때마다 `cd ~\Documents\BiRef_code` →
  `.\.venv\Scripts\Activate.ps1` 두 줄을 다시 실행해야 합니다.

#### 6) PyTorch 설치 (GPU 유무에 따라 다름)

**NVIDIA GPU가 있는 경우**
1. 먼저 그래픽 드라이버가 설치됐는지 확인합니다. 표가 나오면 정상입니다.
   ```powershell
   nvidia-smi
   ```
2. https://pytorch.org/get-started/locally/ 에서
   `Stable / Windows / Pip / Python / CUDA(가장 높은 버전)`를 고르면 나오는 명령을
   복사해 실행합니다. 예시는 다음과 같은 형태입니다(버전 숫자는 공식 페이지 기준으로
   바꾸세요).
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
3. GPU 인식 확인 — `True`가 나와야 합니다.
   ```powershell
   python -c "import torch; print(torch.cuda.is_available())"
   ```

**GPU가 없는 경우 (CPU 전용)**
```powershell
pip install torch torchvision
```

#### 7) 나머지 라이브러리 설치

```powershell
pip install "numpy<2" opencv-python timm kornia einops
```

#### 8) 가중치 내려받기 (약 844MB)

```powershell
mkdir weights
curl.exe -L -o weights\BiRefNet-general-epoch_244.pth https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth
```

- PowerShell에서는 `curl`이 아니라 **`curl.exe`** 로 써야 합니다.
- 명령이 어렵다면 위 주소를 브라우저에 붙여넣어 내려받은 뒤, 파일을
  `Documents\BiRef_code\weights` 폴더로 옮겨도 됩니다.

#### 9) 테스트 사진 넣기
사람이나 물체가 찍힌 사진 1장을 `Documents\BiRef_code` 폴더에 **`test.jpg`**
라는 이름으로 넣습니다.

#### 10) 실행

```powershell
cd BiRefNet
python -c "import torch,cv2,numpy as np; torch.set_grad_enabled(False); from models.birefnet import BiRefNet; from utils import check_state_dict; d='cuda' if torch.cuda.is_available() else 'cpu'; m=BiRefNet(bb_pretrained=False); print(m.load_state_dict(check_state_dict(torch.load('../weights/BiRefNet-general-epoch_244.pth',map_location='cpu',weights_only=True)))); m=m.to(d).eval(); im=cv2.imread('../test.jpg'); h,w=im.shape[:2]; x=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(1024,1024)).astype('float32')/255; x=(x-np.array([0.485,0.456,0.406],'float32'))/np.array([0.229,0.224,0.225],'float32'); t=torch.from_numpy(x.transpose(2,0,1).copy())[None].to(d); y=m(t)[-1].sigmoid()[0,0].cpu().numpy(); cv2.imwrite('../mask.png',(cv2.resize(y,(w,h))*255).astype('uint8')); print('OK: mask.png saved, device =',d)"
cd ..
```

**성공 기준**: 경고(Warning) 문구가 몇 줄 섞여 나와도 괜찮습니다. 아래 두 줄이
보이면 성공이며, `BiRef_code` 폴더에 `mask.png`가 생깁니다.

```
<All keys matched successfully>
OK: mask.png saved, device = cuda      ← GPU가 없으면 cpu
```

#### Windows 문제 해결

| 증상 | 해결 |
| --- | --- |
| `py`/`python`을 찾을 수 없음 | 1)에서 "Add python.exe to PATH" 체크를 빠뜨림 → Python 재설치 |
| `No module named 'models'` | `cd BiRefNet`을 하지 않고 실행함 → 10)을 처음부터 |
| `'NoneType' object has no attribute 'shape'` | `test.jpg`가 없거나 이름이 다름 (`test.jpg.jpg`처럼 확장자가 두 번 붙었는지 확인 — 탐색기 "보기 → 파일 확장명" 켜기) |
| `FileNotFoundError ... .pth` | 8) 가중치 파일 위치·이름 확인 |
| GPU가 있는데 `device = cpu` | 6)에서 CPU용 PyTorch가 설치됨 → `pip uninstall torch torchvision` 후 GPU용 명령으로 재설치 |
| `CUDA out of memory` | 그래픽 메모리 부족(공식 기준 FP32 추론 약 4.8GB 필요) → 다른 프로그램 종료 |

---

### 🍎 3-B. macOS

#### 준비물
- **Apple Silicon(M1/M2/M3/M4…) Mac 권장.** 최신 PyTorch는 Intel Mac용 설치 파일을
  더 이상 제공하지 않으므로 Intel Mac에서는 아래 6)이 실패할 수 있습니다.
  (내 Mac 확인: 화면 왼쪽 위 애플(사과) 메뉴 → "이 Mac에 관하여" → "칩" 항목)
- 이번 단계의 테스트는 **CPU로 실행**됩니다. Mac GPU(MPS) 사용과 실패 시 CPU 자동
  전환은 다음 단계의 `realtime_matte.py`에서 처리할 예정입니다.

#### 1) Python 3.11 설치
1. https://www.python.org/downloads/macos/ 에서 **Python 3.11.x**
   "macOS 64-bit universal2 installer"를 내려받아 설치합니다.
2. 설치가 끝나면 열리는 폴더에서 **`Install Certificates.command`** 를 더블클릭해
   한 번 실행합니다(인터넷 다운로드 인증서 문제 예방).

#### 2) Git 설치
1. **터미널** 앱을 엽니다(Spotlight `⌘ + Space` → "터미널").
2. 아래 명령을 실행하고, 창이 뜨면 "설치"를 누릅니다. 이미 설치돼 있다는 메시지가
   나오면 그대로 넘어가면 됩니다.

```bash
xcode-select --install
```

#### 3) 이 저장소 내려받기

```bash
cd ~/Documents
git clone https://github.com/workseok/BiRef_code.git
cd BiRef_code
git checkout claude/nice-faraday-2q8lvr
```

#### 4) 공식 BiRefNet 코드 내려받기 (검증 버전 `ebcc0bc`로 고정)

```bash
git clone https://github.com/ZhengPeng7/BiRefNet.git
cd BiRefNet
git checkout ebcc0bc
cd ..
```

#### 5) 가상환경 만들기 + 켜기

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

- 성공하면 줄 맨 앞에 `(.venv)`가 붙습니다.
- 터미널을 새로 열 때마다 `cd ~/Documents/BiRef_code` →
  `source .venv/bin/activate` 두 줄을 다시 실행해야 합니다.

#### 6) PyTorch + 나머지 라이브러리 설치
Mac은 GPU 종류를 고를 필요 없이 아래 명령 하나로 됩니다.

```bash
pip install torch torchvision "numpy<2" opencv-python timm kornia einops
```

#### 7) 가중치 내려받기 (약 844MB)

```bash
mkdir -p weights
curl -L -o weights/BiRefNet-general-epoch_244.pth https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth
```

#### 8) 테스트 사진 넣기
사진 1장을 Finder에서 `문서(Documents) → BiRef_code` 폴더에 **`test.jpg`** 라는
이름으로 넣습니다.

#### 9) 실행

```bash
cd BiRefNet
python -c "import torch,cv2,numpy as np; torch.set_grad_enabled(False); from models.birefnet import BiRefNet; from utils import check_state_dict; d='cuda' if torch.cuda.is_available() else 'cpu'; m=BiRefNet(bb_pretrained=False); print(m.load_state_dict(check_state_dict(torch.load('../weights/BiRefNet-general-epoch_244.pth',map_location='cpu',weights_only=True)))); m=m.to(d).eval(); im=cv2.imread('../test.jpg'); h,w=im.shape[:2]; x=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(1024,1024)).astype('float32')/255; x=(x-np.array([0.485,0.456,0.406],'float32'))/np.array([0.229,0.224,0.225],'float32'); t=torch.from_numpy(x.transpose(2,0,1).copy())[None].to(d); y=m(t)[-1].sigmoid()[0,0].cpu().numpy(); cv2.imwrite('../mask.png',(cv2.resize(y,(w,h))*255).astype('uint8')); print('OK: mask.png saved, device =',d)"
cd ..
open mask.png
```

**성공 기준**: 경고(Warning) 문구가 섞여 나와도 괜찮습니다. 아래 두 줄이 보이고,
마지막 `open` 명령으로 흰색 피사체 / 검은 배경 이미지가 열리면 성공입니다.
CPU 실행이라 사진 1장에 수십 초 걸릴 수 있습니다.

```
<All keys matched successfully>
OK: mask.png saved, device = cpu
```

#### macOS 문제 해결

| 증상 | 해결 |
| --- | --- |
| `python3.11: command not found` | 1) Python 3.11 설치 확인 후 터미널을 새로 열기 |
| `SSL: CERTIFICATE_VERIFY_FAILED` | 1)-2의 `Install Certificates.command` 실행 |
| 6)에서 `No matching distribution found for torch` | Intel Mac이거나 Python 버전이 맞지 않음 → Apple Silicon + Python 3.11 확인 |
| `No module named 'models'` | `cd BiRefNet`을 하지 않고 실행함 |
| `'NoneType' object has no attribute 'shape'` | `test.jpg`가 없거나 이름이 다름 (Finder에서 "정보 가져오기"로 실제 확장자 확인) |
| `FileNotFoundError ... .pth` | 7) 가중치 파일 위치·이름 확인 |

---

### 검증 범위 (정직한 고지)

- 위 실행 명령(10번/9번의 `python -c ...` 한 줄)은 Linux + Python 3.11 + CPU
  환경에서 실제 사진으로 실행해 정상적인 마스크가 나오는 것을 확인했습니다.
- **Windows와 macOS 실기기에서는 아직 직접 실행해 보지 않았습니다.** 명령은 두 OS의
  셸 문법(PowerShell / zsh)에 맞춰 작성했지만, 처음 따라 해 보실 때 막히는 지점이
  있으면 알려주세요.
