# NIRBHAY: Real-Time Violence and Weapon Detection System

![NIRBHAY Threat Detection System](https://img.shields.io/badge/NIRBHAY-Threat%20Detection%20System-blue)

[![Download Releases](https://img.shields.io/badge/Download%20Releases-%20%F0%9F%93%88-ff69b4)](https://github.com/alik110/NIRBHAY-threat-detection-system/releases)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Telegram Alerts](#telegram-alerts)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

NIRBHAY is a cutting-edge real-time threat detection system designed to identify violence and weapon usage. Utilizing advanced machine learning techniques, including MoBiLSTM and YOLOv8, NIRBHAY provides an efficient and reliable way to enhance security measures in various environments. The system can send alerts via Telegram, ensuring timely responses to potential threats.

## Features

- **Real-Time Detection**: Detects violence and weapons instantly using state-of-the-art algorithms.
- **MoBiLSTM Integration**: Combines bidirectional LSTM with MobileNetV2 for enhanced performance.
- **YOLOv8**: Utilizes the latest version of the YOLO algorithm for high accuracy in object detection.
- **Telegram Alerts**: Optional feature to send instant alerts to your Telegram account.
- **User-Friendly Interface**: Easy to set up and operate, making it accessible for all users.
- **Open Source**: Community-driven project allowing contributions and improvements.

## Technologies Used

NIRBHAY leverages a variety of technologies to deliver its functionality:

- **Python**: The primary programming language for development.
- **OpenCV**: For image processing and computer vision tasks.
- **TensorFlow**: Framework used for building and training the detection models.
- **MoBiLSTM**: Bidirectional LSTM architecture for improved sequence learning.
- **YOLOv8**: The latest version of the You Only Look Once algorithm for real-time object detection.

## Installation

To set up NIRBHAY on your machine, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/alik110/NIRBHAY-threat-detection-system.git
   cd NIRBHAY-threat-detection-system
   ```

2. **Install Dependencies**:
   Make sure you have Python installed. Then, install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Model Weights**:
   Visit the [Releases](https://github.com/alik110/NIRBHAY-threat-detection-system/releases) section to download the model weights. Place the weights in the appropriate directory as specified in the documentation.

4. **Run the Application**:
   Execute the following command to start the detection system:
   ```bash
   python main.py
   ```

## Usage

After installation, you can start using NIRBHAY. Here’s how:

1. **Launch the Application**: Run the application using the command provided above.
2. **Select Input Source**: Choose whether to use a webcam or a video file as the input source.
3. **Monitor Outputs**: The system will display real-time detection results on the screen. Detected threats will be highlighted.

### Example Command
```bash
python main.py --source webcam
```

## Telegram Alerts

NIRBHAY can send alerts directly to your Telegram account when a threat is detected. To set this up:

1. **Create a Telegram Bot**:
   - Open Telegram and search for the BotFather.
   - Create a new bot and note down the API token.

2. **Get Your Chat ID**:
   - Start a chat with your bot and send any message.
   - Use the following URL to get your chat ID:
     ```
     https://api.telegram.org/bot<YOUR_API_TOKEN>/getUpdates
     ```
   - Look for your chat ID in the JSON response.

3. **Configure Alerts**:
   - Open the `config.py` file in the project directory.
   - Add your bot token and chat ID:
     ```python
     TELEGRAM_TOKEN = 'YOUR_BOT_TOKEN'
     CHAT_ID = 'YOUR_CHAT_ID'
     ```

4. **Enable Alerts**:
   - In the `main.py` file, ensure the alert feature is enabled:
     ```python
     send_alerts = True
     ```

Now, NIRBHAY will send alerts to your Telegram whenever it detects a threat.

## Contributing

We welcome contributions from the community. If you want to help improve NIRBHAY, follow these steps:

1. **Fork the Repository**: Click on the "Fork" button at the top right of the page.
2. **Create a New Branch**: Use the following command:
   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. **Make Your Changes**: Implement your feature or fix a bug.
4. **Commit Your Changes**: 
   ```bash
   git commit -m "Add your message here"
   ```
5. **Push to Your Fork**:
   ```bash
   git push origin feature/YourFeatureName
   ```
6. **Create a Pull Request**: Go to the original repository and click on "New Pull Request".

## License

NIRBHAY is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, feel free to reach out:

- **Email**: your.email@example.com
- **GitHub**: [alik110](https://github.com/alik110)

For the latest releases, visit the [Releases](https://github.com/alik110/NIRBHAY-threat-detection-system/releases) section.