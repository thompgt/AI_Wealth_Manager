# 🏦 AI Wealth Manager Engine

## Overview
The **AI Wealth Manager** is a sophisticated, autonomous system designed to manage investment portfolios with the same care and rigor as a professional financial firm. It combines cutting-edge Artificial Intelligence (Google Gemini) with strict, automated "guardrails" to ensure every trade is smart, tax-efficient, and fully compliant with financial regulations.

## How It Works (The "Four-Eyes" Principle)
Unlike a simple computer program, this system uses a "Multi-Agent" approach—where four specialized virtual "experts" must work together to approve any portfolio changes:

1.  **🌍 Market Scout (Macro Sentinel):**
    Uses AI to read global market trends. It decides if the current environment is "Healthy" (Bull), "Risky" (Bear), or "Uncertain" (Volatile).
2.  **📈 Strategy Architect (Quant Builder):**
    A math-driven expert that looks at your current investments and proposes the best buy or sell moves based on the Market Scout's findings.
3.  **⚖️ Tax Guardian (Tax Architect):**
    Specifically looks for "Wash-Sale" violations—a complex tax rule that can cost investors money if they buy and sell the same stock too quickly. If it finds a violation, it forces the Strategy Architect to pick a different, safer stock.
4.  **🛡️ Compliance Officer (Compliance Critic):**
    The final firewall. It checks every proposed move against a list of strict "Must-Follow" rules (e.g., "Never put more than 30% of wealth into one stock" or "Never buy high-risk penny stocks"). If a move breaks even one rule, the whole process is stopped immediately.

## Key Benefits
*   **Always On:** Monitors and adjusts portfolios autonomously.
*   **Emotion-Free:** Decisions are based on data and AI analysis, not fear or greed.
*   **Automatic Protection:** Built-in tax and regulatory checks mean you don't have to worry about complex paperwork or rule-breaking.
*   **Auditable:** Every decision and hand-off between virtual experts is logged and transparent.

## Getting Started
### 1. Launch the Backend
The "engine" that powers the experts:
```bash
uvicorn server:app --reload
```

### 2. Launch the Dashboard
The easy-to-use visual interface for managers:
```bash
solara run app.py
```

### 3. Usage
Simply enter a **User ID** in the dashboard and click **Trigger AI Rebalance**. You will see the AI's logic, the market conditions it found, and the final portfolio breakdown instantly.
