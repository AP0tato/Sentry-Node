# Run Instructions

## Downloads

### Download python 3.12

``` bash
# Mac
brew install python@3.12

# If it doesn't work on apple silicon try:
arch -arm64 brew install python@3.12
```

### Create and activate venv

``` bash
cd your/project/dir
python3.12 -m venv .venv

# Unix
source .venv/bin/activate

# Windows
.venv/Scripts/activate
```

### Download libraries
``` bash
# Mac
pip install opencv-python
pip install pytorch torchvision
pip install ultralytics

# Windows
pip install pytorch torchvision # If using CPU
pip install pytorch torchvision --index-url https://download.pytorch.org/whl/cu128
# Replace 128 with your cuda version, the latest cuda version is 128
pip install opencv-python
pip install ultralytics
```