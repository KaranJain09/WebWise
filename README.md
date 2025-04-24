<h1 align="center">🌐 WebWise – Talk to Any Website, Instantly!</h1>
<p align="center">
  <img src="https://img.shields.io/badge/Built%20with-Streamlit-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/AI%20Powered-LangChain-blueviolet?style=for-the-badge&logo=openai" />
  <img src="https://img.shields.io/badge/Embeddings-ChromaDB-green?style=for-the-badge" />
  <img src="https://img.shields.io/github/license/Karanjain09/webwise-chat?style=for-the-badge" />
</p>

---

> 🧠 A Streamlit-based app that lets you chat with one or multiple websites. It extracts, analyzes, and responds using the power of **LLMs**, **web scraping**, and **vector embeddings**.

---

## 📸 Demo Snapshot

```
📎 Upload URLs -> 🔍 Scrape Pages -> 📚 Embed Content -> 🤖 Ask Questions -> 🎯 Get Answers (with images!)
```


---

## 🚀 Live Web App

👉 **Try it live**: [Click to Use WebWise](https://your-deployment-link.com)

> No signup. No fluff. Just paste a URL and start asking questions!

---

## 🧠 What Is WebWise?

**WebWise** turns any website into an interactive Q&A experience. It fetches website content – including text, images, and tables – and stores it intelligently using **ChromaDB** vector databases. Then, with help from **LLMs**, it allows users to:

- 📚 **Summarize websites**
- 🔍 **Compare two or more web pages**
- 💬 **Chat with content (text/images)**
- 🧾 **Extract insights without manual reading**

**Example use cases**:
- Students studying articles across sources
- Researchers comparing academic sites
- Buyers comparing product reviews across marketplaces
- News analysis from different publishers

---

## 🎯 Features at a Glance

| Feature                          | Description |
|----------------------------------|-------------|
| 🔗 Multi-URL Upload              | Paste one or more URLs and analyze them together |
| 🧽 Smart Scraper                 | Extracts clean text, images, and tables |
| 🧠 Contextual Chat               | Ask questions based on site content |
| 📸 Image Display                 | Displays relevant images related to the query |
| 🗃 Vector-Based Memory           | Uses `ChromaDB` and `LangChain` to remember & retrieve |
| 🪄 No Login Required             | Open access for everyone |
| 🚀 Fast & Lightweight            | Streamlit ensures real-time, interactive speed |

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/Karanjain09/WebWise.git
cd WebWise-chat

# 2. Install required dependencies
pip install -r requirements.txt

# 3. Set up your environment (API keys in .env file)
touch .env
# Add Groq_API_KEY and HUGGINGFACEHUB_API_TOKEN inside

# 4. Launch the app
streamlit run app.py
```

---

## 🔍 How Does It Work?

Let’s break it down step-by-step so even non-techies can follow!

```
🌐 Step 1: You upload URLs
🔍 Step 2: Pages are scraped with BeautifulSoup + newspaper3k
✂️ Step 3: Text is split into smart chunks (LangChain Splitter)
🧠 Step 4: Each chunk is converted into a vector using HuggingFace Embeddings
💾 Step 5: All vectors are stored in ChromaDB
🧵 Step 6: You ask a question
🔎 Step 7: Relevant chunks are retrieved using similarity search
🤖 Step 8: Chunks + Query go to the LLM
🎯 Step 9: LLM replies with a natural, informative response
🖼️ Step 10: Relevant images are shown from the original page
```

📌 **Visual Overview (ASCII-style)**

```text
  +------------+       +------------------+       +-----------------+
  |  Website   | --->  |   WebWise Crawler| --->  |   Text/Image    |
  +------------+       +------------------+       +-----------------+
                                                          ↓
                                                  +------------------+
                                                  | Vector Embeddings|
                                                  +------------------+
                                                          ↓
                                                      [ChromaDB]
                                                          ↓
                                                      User Question
                                                          ↓
                                                  +-------------------+
                                                  |   LLM (AI Model)  |
                                                  +-------------------+
                                                          ↓
                                              🎯 Final Answer + 📸 Images
```

---

## 🧪 Example Use Cases

| Scenario | What You Can Do |
|---------|------------------|
| 📰 Compare News Sites | Ask: _“What are the main differences in reporting between site A and B?”_ |
| 🛒 Product Analysis | _“Which product has better reviews?”_ |
| 🧾 Legal/Policy Review | _“Summarize the privacy policy of X”_ |
| 📷 Media Focus | _“Show me images related to the article topic”_ |
| 🏫 Research Simplification | _“Summarize this academic article”_ |

---

## ⚙️ Configuration

Create a `.env` file with your API keys:

```env
Groq_API_KEY=your_Groq_key
```

---

## 📦 Dependencies

Main packages used in `app.py`:

```python
streamlit
requests, re, json, os, shutil, time, hashlib
bs4 (BeautifulSoup)
newspaper3k
langchain
langchain_community
langchain_chroma
huggingface_hub
chromadb
dotenv
Pillow (PIL)
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## 🧰 Troubleshooting

| Problem | Solution |
|--------|----------|
| ❌ App not starting | Ensure Streamlit is installed and running `app.py` |
| 📷 No images displayed | Make sure the site has image content |
| 🔐 LLM errors | Double-check your API keys in `.env` |
| 🌐 URL not working | Some websites block scraping (try others) |

---

## 👨‍💻 Author

Made with 💡 and ☕ by **[Karan Jain](https://github.com/Karanjain09)**

If you love it, please ⭐ the repo and share it!

---

## 💡 Pro Tip

> 💬 Want to enhance this further? Integrate PDF parsing, PDF comparison, or even voice interaction!

---
