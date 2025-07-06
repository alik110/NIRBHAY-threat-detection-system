
# NIRBHAY: A real-time threat detection system

NIRBHAY is a real-time surveillance system that detects violent behavior and weapons from live video using deep learning — combining a MobileNetV2 + BiLSTM model for violence classification and YOLOv8 for weapon detection, with optional instant Telegram alerts.


## Project Objectives 

- Detect violence based on sequential human actions across video frames.
- Detect and classify weapons in real-time with precise localization.
- Enable automated alerts via Telegram for rapid response in critical scenarios.


## Setup Instructions

#### 1. Clone the repository

```bash
git clone https://github.com/sanjyot02/NIRBHAY-threat-detection-system.git
```

#### 2. Go to the project directory

```bash
cd NIRBHAY-threat-detection-system
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```
#### 4. Telegram Bot Setup (for Alerts)

In **with_alert.py**, replace:

```bash
BOT_TOKEN = "your_bot_token"
CHAT_ID = "your_chat_id"
```
You can generate these using **BotFather** on Telegram.



## Usage

To run without any Telegram alerts:

```bash
python without_alert.py
```

To run with Telegram alerts:

```bash
python with_alert.py
```

## Model Details


#### Violence Detection (MoBiLSTM)
- Model: MobileNetV2 + Bidirectional LSTM
- Input: 16-frame sequences (96×96×3)
- Output: Binary classification — Violence or NonViolence
- Dataset: [Real-Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)

#### Weapon Detection (YOLOv8)
- Model: YOLOv8n (Ultralytics)
- Input: Webcam frames
- Output: Bounding boxes + weapon class labels
- Classes: 9 weapon types (e.g., Knife, Handgun, Rifle, etc.)
- Dataset: [Weapon Detection Dataset](https://www.kaggle.com/datasets/snehilsanyal/weapon-detection-test)
---

#### Status: This project is a work in progress.

For feedback or questions, feel free to reach out at sanjyotpawar2@gmail.com
