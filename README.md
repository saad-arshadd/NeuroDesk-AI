# 🧠 NeuroDesk AI — Voice-Controlled Desktop Assistant

> “Hello Saad, and good morning. It’s Tuesday and the time is 09:23 AM. How can I assist you today?”

**NeuroDesk AI** is a smart, real-time voice assistant built for your PC. From opening apps and solving math to checking your class schedule or searching Google — it listens, thinks, and acts like a personal desktop butler. Powered by deep learning and natural language processing, it understands your commands and executes them smoothly — all hands-free.

---

## 🧰 Features

- 🎙️ **Voice Recognition** (using `SpeechRecognition` and Google API)
- 🤖 **Intent Classification** (via a trained TensorFlow model)
- 📢 **Speech Response** (with `pyttsx3` for offline TTS)
- 🧠 **Custom Schedule Assistant** (day-wise academic schedule)
- 🧮 **Math Problem Solver** (natural language arithmetic)
- 🕹️ **App Launcher & Closer** (Calculator, Notepad, Zoom, VS Code, etc.)
- 🌐 **Web Access** (Google search, open Facebook, Instagram, etc.)
- 🔋 **System Status Monitor** (CPU usage, battery, charging status)
- 🧠 **Fallback AI Chatbot** (for unrecognized queries via intent matching)

---

## 🧪 Technologies Used

| Layer               | Technology           |
|--------------------|----------------------|
| Voice Recognition   | `speech_recognition` |
| Text-to-Speech      | `pyttsx3` (offline)  |
| Intent Detection    | `TensorFlow` + `Keras` |
| Data Serialization  | `pickle`, `json`     |
| Desktop Control     | `pyautogui`, `os`, `psutil` |
| Web Browsing        | `webbrowser`         |
| Assistant Memory    | Custom LSTM + tokenizer |

---

## 📁 Project Structure
├── main.py # Core voice assistant logic
├── model.py # Model training script
├── model_test.py # CLI testing interface for intents
├── intents.json # Training data (user intents + patterns)
├── Chat_model.h5 # Trained Keras model
├── tokenizer.pkl # Saved tokenizer for inputs
├── label_encoder.pkl # Label encoder for tags

## HOW TO RUN 
1) TRAIN THE MODEL : python model.py
2)python main.py

## LIBRARIES REQUIRED
pyttsx3
SpeechRecognition
pyaudio
pyautogui
psutil
tensorflow
numpy
scikit-learn



## DEVELOPED BY :
### MUHAMMAD SAAD ARSHAD

