# 🍄 ML Mushroom Pipeline

The **ML Mushroom Pipeline** is a machine learning project that predicts whether a mushroom is **poisonous or edible** based on its physical features (such as color, odor, and cap shape).  

It combines **data preprocessing, model training, and visualization** into a single workflow.

## 📦 Environment Setup

This project uses **Miniconda** for environment and dependency management, as well as **PyCharm** from JetBrains as IDE.
- Miniconda install link: https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
- PyCharm install link: https://www.jetbrains.com/pycharm/download/download-thanks.html?platform=windows

### 1️⃣ Create the environment

Make sure you have **conda** installed, then go to **Anaconda Powershell Prompt** and go to this path in the project folder:
```bash
cd ml-mushroom-pipeline
```
Then run:
```bash
conda env create -f environment.yml
```

### 2️⃣ Activate the environment

```bash
conda activate ml-mushroom-pipeline
```

### 3️⃣ Verify installation

```bash
conda list
```

### 4️⃣ Set the environment as default in PyCharm (Windows)

To make sure PyCharm always uses your **`ml-mushroom-pipeline`** Conda environment by default:

1. Open your project in **PyCharm** (make sure `.idea/` is in your project root).
2. Go to: File → Settings → Python → Interpreter
3. Click: Add Interpreter → Add Local Interpreter
4. Choose: Select existing → Type:Conda
5. In the path field, browse to your Conda environment’s Python executable, usually found at:
```
C:\Users\<your_user>\miniconda3\envs\ml-mushroom-pipeline\python.exe
```
6. Click OK, then Apply.

PyCharm will now use this environment automatically for:
- Running scripts (`Shift + F10`)
- Using the built-in terminal
- Running tests and notebooks

### 5️⃣ Confirm it’s active in PyCharm

- Look in the bottom-right corner of PyCharm — it should say: ml-mushroom-pipeline

- You can also open PyCharm’s terminal (`View → Tool Windows → Terminal`) and run:
```bash
conda info --envs
```

### 6️⃣ Train Model
Open internal Terminal in PyCharm and run:
```
C:\Users\<user>\anaconda3\envs\ml-mushroom-pipeline\python.exe C:\<project_path>\bedste-gruppe-ml\ml\entrypoint\train.py 
```

### 7️⃣ Start API
Open internal Terminal in PyCharm and run:
```
C:\Users\<user>\anaconda3\envs\ml-mushroom-pipeline\python.exe C:\<project_path>\bedste-gruppe-ml\ml\entrypoint\inference_api.py 
```

### 8️⃣ Available endpoints (json):
```
http://localhost:5001/predictions
```
#### GET:
query options:
 - n=10 (10 = amount of latest predictions wanted)
#### POST:
```
{
  "cap-shape": "CONVEX",
  "cap-surface": "SMOOTH",
  "cap-color": "WHITE",
  "bruises": "BRUISES",
  "odor": "NONE",
  "gill-attachment": "FREE",
  "gill-spacing": "CROWDED",
  "gill-size": "NARROW",
  "gill-color": "WHITE",
  "stalk-shape": "TAPERING",
  "stalk-surface-above-ring": "SMOOTH",
  "stalk-surface-below-ring": "SMOOTH",
  "stalk-color-above-ring": "WHITE",
  "stalk-color-below-ring": "WHITE",
  "veil-color": "WHITE",
  "ring-number": "ONE",
  "ring-type": "PENDANT",
  "spore-print-color": "PURPLE",
  "population": "SEVERAL",
  "habitat": "WOODS"
}

```