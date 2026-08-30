# 🛡️ MuleNet AI — Banking Fraud & Mule Account Detection

> **An AI-powered banking fraud defense platform for detecting suspicious mule accounts, transaction anomalies, and interconnected fraud networks.**

## 🚀 Overview

**MuleNet AI** is a banking cybersecurity platform designed to help financial institutions identify potential **mule accounts** and suspicious transaction networks.

Instead of analyzing a transaction in isolation, MuleNet combines **transaction behaviour, account relationships, network patterns, and machine learning** to identify potentially suspicious accounts and prioritize them for investigation.

### 🎯 Core Idea

> **Don't just detect a suspicious transaction — detect the network and behaviour behind it.**

---

## 🔍 Problem

Fraudsters often use **mule accounts** to receive and move illegally obtained money.

A typical flow can look like:

```text
Victim
   ↓
Fraudster
   ↓
Mule Account
   ↓
Multiple Accounts
   ↓
Final Destination
```

Traditional transaction-level detection may miss the bigger picture because individual transactions can appear normal.

MuleNet addresses this by analyzing **relationships and behavioural patterns across the transaction network**.

---

## 💡 Solution

MuleNet provides a centralized intelligence layer that can:

* Analyze banking transactions
* Identify unusual transaction behaviour
* Detect suspicious account activity
* Build transaction/entity networks
* Identify suspicious clusters and potential fraud rings
* Calculate account risk
* Provide explainable reasons for alerts
* Help investigators prioritize high-risk cases

---

## 🧠 How MuleNet Works

```text
             Banking Data
                  │
                  ▼
        ┌──────────────────┐
        │ Data Processing  │
        └──────────────────┘
                  │
                  ▼
        ┌──────────────────┐
        │ Transaction Graph│
        └──────────────────┘
                  │
                  ▼
        ┌──────────────────┐
        │ Feature Engine   │
        └──────────────────┘
                  │
                  ▼
        ┌──────────────────┐
        │ AI / ML Analysis │
        └──────────────────┘
                  │
                  ▼
        ┌──────────────────┐
        │ Risk Scoring     │
        └──────────────────┘
                  │
                  ▼
        ┌──────────────────┐
        │ Explainable Alert│
        └──────────────────┘
                  │
                  ▼
        ┌──────────────────┐
        │ Investigation    │
        └──────────────────┘
```

---

# ⭐ Key Features

### 1. 🔎 Transaction Analysis

Analyzes transaction-level information such as:

* Sender
* Receiver
* Transaction amount
* Timestamp
* Transaction frequency
* Transaction direction
* Channel

This helps identify unusual financial behaviour.

---

### 2. 🧠 Behavioural Analysis

MuleNet can compare an account's current behaviour with its historical pattern.

For example:

```text
Normal behaviour
10 transactions/month
        ↓
Sudden change
150 transactions/month
        ↓
Behavioural anomaly
```

---

### 3. 🕸️ Transaction Network Analysis

Financial transactions can be represented as a graph.

```text
        Account B
           ↑
           │
Account C ← A → Account D
           │
           ↓
        Account E
```

Where:

* **Nodes** = accounts/entities
* **Edges** = financial relationships/transactions

This helps investigators understand **who is connected to whom**.

---

### 4. 🚨 Mule Account Detection

MuleNet looks for combinations of suspicious signals such as:

* Unusual transaction velocity
* High transaction frequency
* Large unexpected inflows
* Rapid outgoing transfers
* Many counterparties
* Sudden beneficiary activity
* Suspicious network relationships
* Behavioural deviations

A high-risk score indicates that an account deserves further investigation.

> **Risk score is not a criminal verdict.**

---

### 5. 🕸️ Fraud Ring Detection

Multiple individually normal-looking accounts can form a suspicious network.

Example:

```text
           A
         /   \
        B     C
         \   /
           D
           │
           E
```

Network analysis helps identify suspicious clusters and potential coordinated activity.

---

### 6. 📊 Risk Scoring

Accounts can be prioritized according to their calculated risk.

Example:

```text
0 – 40     → Low Risk
40 – 70    → Medium Risk
70 – 85    → High Risk
85 – 100   → Critical
```

> Thresholds can be configured according to the deployed model and banking requirements.

---

### 7. 💡 Explainable AI

Instead of simply saying:

```text
Risk Score = 94%
```

MuleNet can provide supporting signals such as:

```text
✓ Unusual transaction velocity
✓ High number of counterparties
✓ Rapid fund movement
✓ Sudden behavioural change
✓ Suspicious network connections
```

This makes the system more useful for human investigators.

---

### 8. 👨‍💻 Investigator Dashboard

The dashboard provides a centralized view of:

* Risk levels
* Suspicious accounts
* Transaction activity
* Network relationships
* Alerts
* Investigation priorities

The goal is to help investigators focus on the **highest-risk cases first**.

---

# 🧪 Example

Consider an account that normally performs only a few transactions every month.

Suddenly:

```text
12 different accounts
       ↓
   Mule Account
       ↓
Rapid transfers
       ↓
8 different accounts
```

MuleNet can combine:

```text
Transaction behaviour
        +
Account relationships
        +
Network structure
        +
Machine learning
        ↓
   Risk Analysis
        ↓
 Suspicious Account
```

The investigator can then inspect the account and its connected network.

---

# 🛠️ Technology Stack

### Frontend

* HTML
* CSS
* JavaScript
* React *(if applicable)*

### Data & Analytics

* Python
* Pandas
* NumPy
* Scikit-learn

### Machine Learning

* Random Forest
* Gradient Boosting
* Logistic Regression
* Ensemble/Voting models
* Graph-based analysis

### Graph Intelligence

* Graph-based network analysis
* Network centrality
* Shortest-path analysis
* Graph visualization

### Deployment

* Vercel

> Update this section to exactly match the technologies present in the repository.

---

# 📊 Model Performance

The project's reported evaluation metrics include:

| Metric              |     Result |
| ------------------- | ---------: |
| AUC-ROC             | **0.9879** |
| F1 Score            |   **0.95** |
| Precision           |  **93.4%** |
| False Positive Rate |   **0.5%** |

These figures should be interpreted in the context of the dataset, preprocessing pipeline, train/test methodology, and evaluation setup used by the project.

---

# 📁 Dataset

The repository can be tested with synthetic banking transaction data.

Example structure:

```csv
transaction_id,sender_account,receiver_account,amount_inr,timestamp,channel
TXN0001,ACC1003,ACC1000,8000,2026-08-06 14:14:00,UPI
TXN0002,ACC1003,ACC1017,2500,2026-08-13 23:27:00,UPI
```

⚠️ **No real customer banking data should be included in this repository.**

Use synthetic, anonymized, or publicly permitted datasets for development and demonstration.

---

# 🔐 Security & Privacy

MuleNet is designed as a **risk-detection and investigation-support system**.

Important principles:

* Do not expose personally identifiable information.
* Do not upload real customer banking data to public repositories.
* Use anonymized/synthetic data for development.
* Apply appropriate access controls in production.
* AI-generated risk scores should support, not replace, authorized human investigation.

---

# 🎯 Use Cases

MuleNet can potentially support:

* Banking fraud detection
* Mule account identification
* AML investigation
* Transaction monitoring
* Fraud-ring discovery
* Suspicious account prioritization
* Financial network analysis
* Investigator decision support

---

# 🌐 Live Demo

**MuleNet AI:**
https://mul-net-e1h2.vercel.app/

---

# 📌 Project Status

🚧 **Hackathon / Prototype**

The current version demonstrates the core concept of AI-assisted mule-account and financial-network analysis.

Future development can focus on:

* Real-time transaction streaming
* Larger-scale graph processing
* Improved model calibration
* Advanced GNN architectures
* Cross-bank intelligence
* Automated case management
* Model monitoring and drift detection
* Production-grade security
* Explainability improvements

---

# 🚀 Future Vision

The long-term vision of MuleNet is to evolve from a transaction monitoring tool into a **network-aware financial crime intelligence platform**.

```text
             Multiple Banks
                   │
                   ▼
          ┌─────────────────┐
          │   MuleNet AI    │
          └─────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    Behaviour   Network     Risk
    Analysis   Intelligence Scoring
        │          │          │
        └──────────┼──────────┘
                   ▼
             Investigation
```

---

# 👥 Team

**MuleNet AI Team**

Developed as a cybersecurity and AI solution for banking fraud and mule-account detection.

---

# 📜 Disclaimer

MuleNet AI is a research/prototype system intended to demonstrate AI-assisted financial fraud detection.

A risk score or alert **does not establish that an account or individual is fraudulent**. Any account restriction, investigation, reporting, or other enforcement action should be performed by authorized personnel according to applicable laws, regulations, policies, and institutional procedures.

---

## ⭐ If you find this project interesting

Give the repository a ⭐ and explore the project!

**MuleNet AI — Detect the behaviour. Understand the network. Stop the flow.**
