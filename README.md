# DL-training

Hands-on files from my Deep Learning / LLM training: Python basics, Streamlit apps, a chatbot demo, and a RAG (Retrieval-Augmented Generation) tutorial.

## Contents

| File | What it is |
|------|------------|
| `python_basic.pdf` | Python basics reference notes |
| `cpp.txt` | C++ notes |
| `llm.txt` | LLM concepts / notes |
| `streamlit_tutorial.py` | Streamlit practice: widgets, layout, basic app flow |
| `app.py` | Main Streamlit app built during the training |
| `chatBotDemo.py` | Simple chatbot demo |
| `RAGTutorial.ipynb` | Notebook walking through a RAG pipeline |
| `requirements.txt` | Python dependencies |

## Setup

```bash
# 1. Clone
git clone https://github.com/um034867-sys/DL-training.git
cd DL-training

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Streamlit tutorial
streamlit run streamlit_tutorial.py

# Main app
streamlit run app.py

# Chatbot demo (if it's a plain script)
python chatBotDemo.py
```

For the RAG tutorial, open `RAGTutorial.ipynb` in VS Code or Jupyter and run the cells top to bottom.

> If a script needs an API key (for example an LLM provider), put it in a `.env` file. Never commit it to the repo.

## Notes

- The `venv/` folder is not part of the repo. Recreate it using the setup steps above.
- Training reference repo: [Wazor2/DL-TRAINING](https://github.com/Wazor2/DL-TRAINING)

## Author

Utkarsh, B.Tech Mechatronics Engineering (Robotics & AI), Ramdeobaba University, Nagpur
