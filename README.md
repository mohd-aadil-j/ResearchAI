# ResearchAI

AI-powered research report generator using Groq LLM, LangChain, and Streamlit.

## Features
- Generate structured research reports from any topic
- Choose depth: Beginner, Intermediate, Advanced
- Uses Groq LLM with DuckDuckGo and Wikipedia tools
- Download clean PDF reports
- Modern, attractive Streamlit UI

## Quick Start

1. **Clone the repo:**
   ```sh
   git clone https://github.com/mohd-aadil-j/ResearchAI.git
   cd ResearchAI
   ```
2. **Create a Python virtual environment (recommended):**
   ```sh
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Set your Groq API key:**
   - Create a `.env` file:
     ```env
     GROQ_API_KEY=your_groq_api_key_here
     ```
   - Or set as an environment variable.
5. **Run the app locally:**
   ```sh
   streamlit run app.py
   ```

## Deployment (Streamlit Cloud)
- Push your code to a GitHub repository
- In Streamlit Cloud, connect your repo and set the `GROQ_API_KEY` in Secrets
- Deploy and share your app

## Security
- **Never commit your `.env` file or API keys to GitHub.**
- Always rotate exposed keys immediately.

## Troubleshooting
- If you see `ModuleNotFoundError`, check `requirements.txt` for missing packages.
- For agent errors, ensure your LangChain version matches the code (see comments in `app.py`).

## License
MIT

---
Built by MOHAMAD AADIL J · B.Tech AI & DS · Powered by Groq, LangChain, Streamlit
