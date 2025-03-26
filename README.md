# STEPS TO RUN THE AI PHOTOBOOTH


## some of output of the Ai Photobooth

![alt text](static/img1.png) 
<br>

![alt text](static/img2.png)

## Download and install the cpp build tool from microsoft

https://visualstudio.microsoft.com/visual-cpp-build-tools/
<br>
--> Select this option then install <br>
    ![alt text](image.png)

## install the python v3.9.8
https://www.python.org/downloads/release/python-398/

## install all this pip packages

```powershell
pip install opencv-python
pip install insightface
pip install onnxruntime
pip install firebase-admin
pip install qrcode
pip install Flask
pip install numpy
pip install Pillow
pip install Werkzeug
pip install mxnet-mkl
pip install requests
```






The installtion if ypu get err follow below 

(env) PS C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth> python main.py
Traceback (most recent call last):
  File "C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\main.py", line 22, in <module>
    from gfpgan import GFPGANer
  File "C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages\gfpgan\__init__.py", line 2, in <module>
    from .archs import *
  File "C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages\gfpgan\archs\__init__.py", line 2, in <module>
    from basicsr.utils import scandir
  File "C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages\basicsr\__init__.py", line 4, in <module>
    from .data import *
  File "C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages\basicsr\data\__init__.py", line 22, in <module>
    _dataset_modules = [importlib.import_module(f'basicsr.data.{file_name}') for file_name in dataset_filenames]
  File "C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages\basicsr\data\__init__.py", line 22, in <listcomp>
    _dataset_modules = [importlib.import_module(f'basicsr.data.{file_name}') for file_name in dataset_filenames]
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.9_3.9.3568.0_x64__qbz5n2kfra8p0\lib\importlib\__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages\basicsr\data\realesrgan_dataset.py", line 11, in <module>
    from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
  File "C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages\basicsr\data\degradations.py", line 8, in <module>
    from torchvision.transforms.functional_tensor import rgb_to_grayscale
ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'


step two


(env) PS C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth> python -m pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118
Looking in indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cu118
Collecting torch==2.0.1
  Downloading https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp39-cp39-win_amd64.whl (2619.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.6/2.6 GB 452.5 kB/s eta 0:00:00
Collecting torchvision==0.15.2
  Downloading https://download.pytorch.org/whl/cu118/torchvision-0.15.2%2Bcu118-cp39-cp39-win_amd64.whl (4.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 3.5 MB/s eta 0:00:00
Requirement already satisfied: jinja2 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torch==2.0.1) (3.1.4)
Requirement already satisfied: typing-extensions in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torch==2.0.1) (4.12.2)
Requirement already satisfied: networkx in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torch==2.0.1) (3.2.1)
Requirement already satisfied: sympy in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torch==2.0.1) (1.13.1)
Requirement already satisfied: filelock in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torch==2.0.1) (3.16.1)
Requirement already satisfied: numpy in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torchvision==0.15.2) (1.26.4)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torchvision==0.15.2) (11.0.0)
Requirement already satisfied: requests in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torchvision==0.15.2) (2.32.3)
Requirement already satisfied: MarkupSafe>=2.0 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from jinja2->torch==2.0.1) (3.0.2)
Requirement already satisfied: idna<4,>=2.5 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from requests->torchvision==0.15.2) (3.10)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.6/2.6 GB 452.5 kB/s eta 0:00:00
Collecting torchvision==0.15.2
  Downloading https://download.pytorch.org/whl/cu118/torchvision-0.15.2%2Bcu118-cp39-cp39-win_amd64.whl (4.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 3.5 MB/s eta 0:00:00
Requirement already satisfied: jinja2 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torch==2.0.1) (3.1.4)
Requirement already satisfied: typing-extensions in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torch==2.0.1) (4.12.2)
Requirement already satisfied: networkx in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torch==2.0.1) (3.2.1)
Requirement already satisfied: sympy in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torch==2.0.1) (1.13.1)
Requirement already satisfied: filelock in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torch==2.0.1) (3.16.1)
Requirement already satisfied: numpy in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torchvision==0.15.2) (1.26.4)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torchvision==0.15.2) (11.0.0)
Requirement already satisfied: requests in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torchvision==0.15.2) (2.32.3)
Requirement already satisfied: MarkupSafe>=2.0 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from jinja2->torch==2.0.1) (3.0.2)
Requirement already satisfied: idna<4,>=2.5 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from requests->torchvision==0.15.2) (3.10)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from requests->torchvision==0.15.2) (2024.8.30)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from requests->torchvision==0.15.2) (2.2.3)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from requests->torchvision==0.15.2) (3.4.0)
Requirement already satisfied: jinja2 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torch==2.0.1) (3.1.4)
Requirement already satisfied: typing-extensions in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torch==2.0.1) (4.12.2)
Requirement already satisfied: networkx in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torch==2.0.1) (3.2.1)
Requirement already satisfied: sympy in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torch==2.0.1) (1.13.1)
Requirement already satisfied: filelock in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torch==2.0.1) (3.16.1)
Requirement already satisfied: numpy in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torchvision==0.15.2) (1.26.4)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torchvision==0.15.2) (11.0.0)
Requirement already satisfied: requests in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torchvision==0.15.2) (2.32.3)
Requirement already satisfied: MarkupSafe>=2.0 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from jinja2->torch==2.0.1) (3.0.2)
Requirement already satisfied: idna<4,>=2.5 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from requests->torchvision==0.15.2) (3.10)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from requests->torchvision==0.15.2) (2024.8.30)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from requests->torchvision==0.15.2) (2.2.3)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from requests->torchvision==0.15.2) (3.4.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from sympy->torch==2.0.1) (1.3.0)
Installing collected packages: torch, torchvision
  Attempting uninstall: torch
Requirement already satisfied: numpy in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torchvision==0.15.2) (1.26.4)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torchvision==0.15.2) (11.0.0)
Requirement already satisfied: requests in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from torchvision==0.15.2) (2.32.3)
Requirement already satisfied: MarkupSafe>=2.0 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from jinja2->torch==2.0.1) (3.0.2)
Requirement already satisfied: idna<4,>=2.5 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from requests->torchvision==0.15.2) (3.10)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from requests->torchvision==0.15.2) (2024.8.30)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from requests->torchvision==0.15.2) (2.2.3)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from requests->torchvision==0.15.2) (3.4.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from sympy->torch==2.0.1) (1.3.0)
Installing collected packages: torch, torchvision
  Attempting uninstall: torch
    Found existing installation: torch 2.5.1
Requirement already satisfied: certifi>=2017.4.17 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from requests->torchvision==0.15.2) (2024.8.30)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from requests->torchvision==0.15.2) (2.2.3)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from requests->torchvision==0.15.2) (3.4.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from sympy->torch==2.0.1) (1.3.0)
Installing collected packages: torch, torchvision
  Attempting uninstall: torch
    Found existing installation: torch 2.5.1
Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\users\dhrit\desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages (from sympy->torch==2.0.1) (1.3.0)
Installing collected packages: torch, torchvision
  Attempting uninstall: torch
    Found existing installation: torch 2.5.1
    Found existing installation: torch 2.5.1
    Uninstalling torch-2.5.1:
      Successfully uninstalled torch-2.5.1
  Attempting uninstall: torchvision
    Uninstalling torchvision-0.20.1:
      Successfully uninstalled torchvision-0.20.1
Successfully installed torch-2.0.1+cu118 torchvision-0.15.2+cu118
WARNING: You are using pip version 22.0.4; however, version 24.3.1 is available.
You should consider upgrading via the 'C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\Scripts\python.exe -m pip install --upgrade pip' command.

step 5

(env) PS C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth> python main.py
C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py:69: UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
  warnings.warn(
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\dhrit/.insightface\models\buffalo_l\1k3d68.onnx landmark_3d_68 ['None', 3, 192, 192] 0.0 1.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\dhrit/.insightface\models\buffalo_l\2d106det.onnx landmark_2d_106 ['None', 3, 192, 192] 0.0 1.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\dhrit/.insightface\models\buffalo_l\det_10g.onnx detection [1, 3, '?', '?'] 127.5 128.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\dhrit/.insightface\models\buffalo_l\genderage.onnx genderage ['None', 3, 96, 96] 0.0 1.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\dhrit/.insightface\models\buffalo_l\w600k_r50.onnx recognition ['None', 3, 112, 112] 127.5 127.5
set det-size: (640, 640)
Traceback (most recent call last):
  File "C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\main.py", line 49, in <module>
    swapper = insightface.model_zoo.get_model(model_path)
  File "C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages\insightface\model_zoo\model_zoo.py", line 91, in get_model
    assert osp.exists(model_file), 'model_file %s should exist'%model_file
AssertionError: model_file ./inswapper_128.onnx should exist

step 6

(env) PS C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth> python main.py
C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py:69: UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
  warnings.warn(
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\dhrit/.insightface\models\buffalo_l\1k3d68.onnx landmark_3d_68 ['None', 3, 192, 192] 0.0 1.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\dhrit/.insightface\models\buffalo_l\2d106det.onnx landmark_2d_106 ['None', 3, 192, 192] 0.0 1.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\dhrit/.insightface\models\buffalo_l\det_10g.onnx detection [1, 3, '?', '?'] 127.5 128.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\dhrit/.insightface\models\buffalo_l\genderage.onnx genderage ['None', 3, 96, 96] 0.0 1.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\dhrit/.insightface\models\buffalo_l\w600k_r50.onnx recognition ['None', 3, 112, 112] 127.5 127.5
set det-size: (640, 640)
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
inswapper-shape: [1, 3, 128, 128]
C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
Downloading: "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" to C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\gfpgan\weights\detection_Resnet50_Final.pth

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 104M/104M [00:03<00:00, 29.0MB/s]
Downloading: "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth" to C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\gfpgan\weights\parsing_parsenet.pth

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 81.4M/81.4M [00:03<00:00, 26.1MB/s]
Traceback (most recent call last):
  File "C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\main.py", line 123, in <module>
    gfpganer = GFPGANer(model_path='GFPGANv1.4.pth', upscale=1, arch='clean', channel_multiplier=2)
  File "C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages\gfpgan\utils.py", line 92, in __init__
    loadnet = torch.load(model_path)
  File "C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages\torch\serialization.py", line 791, in load
    with _open_file_like(f, 'rb') as opened_file:
  File "C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages\torch\serialization.py", line 271, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages\torch\serialization.py", line 252, in __init__
    super().__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: 'GFPGANv1.4.pth'
(env) PS C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth> python main.py
C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py:69: UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
  warnings.warn(
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\dhrit/.insightface\models\buffalo_l\1k3d68.onnx landmark_3d_68 ['None', 3, 192, 192] 0.0 1.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\dhrit/.insightface\models\buffalo_l\2d106det.onnx landmark_2d_106 ['None', 3, 192, 192] 0.0 1.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\dhrit/.insightface\models\buffalo_l\det_10g.onnx detection [1, 3, '?', '?'] 127.5 128.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\dhrit/.insightface\models\buffalo_l\genderage.onnx genderage ['None', 3, 96, 96] 0.0 1.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: C:\Users\dhrit/.insightface\models\buffalo_l\w600k_r50.onnx recognition ['None', 3, 112, 112] 127.5 127.5
set det-size: (640, 640)
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
inswapper-shape: [1, 3, 128, 128]
C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
C:\Users\dhrit\Desktop\allprojects\photov4\prompt-ai-photobooth\env\lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
 * Serving Flask app 'main'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
127.0.0.1 - - [12/Nov/2024 18:16:39] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [12/Nov/2024 18:16:40] "GET /static/aict.png HTTP/1.1" 200 -
127.0.0.1 - - [12/Nov/2024 18:16:40] "GET /static/aibg.png HTTP/1.1" 200 -
127.0.0.1 - - [12/Nov/2024 18:17:18] "POST /index HTTP/1.1" 200 -
127.0.0.1 - - [12/Nov/2024 18:17:18] "GET /static/aibg.png HTTP/1.1" 304 -
127.0.0.1 - - [12/Nov/2024 18:17:18] "GET /static/aicp.png HTTP/1.1" 200 -
