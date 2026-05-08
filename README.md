# 🛡️ RL-Driven Autonomous Network Intrusion Defense System

An advanced AI-powered cybersecurity platform that uses Reinforcement Learning (RL) to detect, analyze, and respond to network intrusion threats in real time.

This project simulates a modern Security Operations Center (SOC) environment where intelligent agents continuously monitor network traffic, identify suspicious behavior, calculate risk levels, and take autonomous defensive actions against cyberattacks.

---

# 🚀 Project Objective

The main goal of this project is to build a self-learning intrusion defense system capable of:

* Detecting malicious network behavior
* Preventing cyberattacks in real time
* Reducing false alarms
* Adapting to evolving attack patterns
* Providing explainable AI-based security decisions

Unlike traditional rule-based IDS systems, this platform uses Reinforcement Learning to continuously improve its defensive strategies through experience.

---

# 🔥 Key Features

✅ Reinforcement Learning-based Intrusion Detection
✅ Real-Time Cyber Threat Monitoring
✅ Explainable AI Decisions
✅ Dynamic Risk Scoring System
✅ Multi-Attack Detection Engine
✅ Honeypot Redirection Simulation
✅ Predictive Threat Intelligence
✅ Autonomous Recovery System
✅ SOC-Style Streamlit Dashboard
✅ Flask API Integration
✅ CSV Logging & Analytics
✅ Professional Cybersecurity UI

---

# 🧠 Supported Attack Types

The system can simulate and detect:

* DDoS Attacks
* Brute Force Attacks
* Port Scanning
* Credential Stuffing
* Bot Traffic
* Insider Threats
* Unknown/Suspicious Behavior

---

# ⚙️ Technologies Used

* Python
* Reinforcement Learning (Q-Learning)
* Streamlit
* Flask API
* Pandas
* Matplotlib
* NumPy

---

# 🏗️ Project Architecture

```bash
Cybersecurity-Intrusion-Detection/
│
├── simulator.py
├── environment.py
├── agent.py
├── train.py
├── dashboard.py
├── api.py
├── utils.py
├── config.py
│
├── logs/
├── models/
├── data/
├── archive/
└── README.md
```

---

# 🧩 How the System Works

1. The traffic simulator generates realistic network traffic.
2. The RL agent analyzes traffic behavior.
3. The environment calculates rewards and penalties.
4. The AI selects security actions such as:

   * Allow
   * Monitor
   * Throttle
   * Block
   * Isolate Device
   * Redirect to Honeypot
5. The dashboard visualizes threats and AI decisions in real time.
6. All predictions and attack events are logged for analysis.

---

# 📊 Dashboard Features

* Live Traffic Monitoring
* Threat Severity Indicators
* Attack Detection Alerts
* Risk Distribution Graphs
* AI Decision Logs
* Accuracy Monitoring
* SOC-Style Cybersecurity Theme

---

# 🔌 API Endpoint

### POST `/predict`

Input:

```json
{
  "request_rate": 90,
  "failed_logins": 8,
  "unknown_ip": 1,
  "time_of_day": 2
}
```

Output:

```json
{
  "action": "BLOCK",
  "risk_score": 92,
  "severity": "CRITICAL",
  "attack_type": "ddos",
  "explanation": "High request rate detected matching DDoS behavior"
}
```

---

# ▶️ Running the Project

### Train RL Model

```bash
python train.py
```

### Run Dashboard

```bash
streamlit run dashboard.py
```

### Start API

```bash
python api.py
```

Dashboard URL:

```bash
http://localhost:8501
```

---

# 📈 Future Improvements

* Deep Reinforcement Learning
* Real Packet Capture Integration
* Docker Deployment
* Cloud-Based Threat Monitoring
* SIEM Integration
* Real Firewall Automation

---

# 👨‍💻 Author

AI & Cybersecurity Project
Developed for research, learning, and intelligent cyber defense simulation.

---

# ⭐ Project Highlights

This project combines:

* Artificial Intelligence
* Reinforcement Learning
* Cybersecurity
* Real-Time Monitoring
* Explainable AI
* Autonomous Defense Systems

to create a professional enterprise-style intrusion defense platform.
