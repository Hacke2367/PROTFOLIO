import streamlit as st

# --- Page Config ---
st.set_page_config(page_title="AI/ML Portfolio - Abhishek", layout="centered")

# --- Custom Styling (Modern Tech / Cyberpunk Theme) ---
st.markdown("""
<style>
    /* Import Google Font - Inter for a modern, clean look */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    /* Optional: A more techy font for headings if desired, e.g., 'Orbitron' or 'Rajdhani' */
    /* @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap'); */

    html, body {
        background-color: #0A0A1A !important; /* Very Dark Blue/Black - Deep tech background, added !important */
    }

    .stApp {
        background-color: #0A0A1A !important; /* Ensure main Streamlit app container is dark, added !important */
        color: #E0E0E0; /* Light Grey for general text, ensures readability */
        font-family: 'Inter', sans-serif; /* Modern and clean font */
    }

    h1, h2, h3 {
        color: #00FFFF; /* Electric Cyan - Primary vibrant accent for headings */
        font-family: 'Inter', sans-serif; /* Keeping Inter for consistency, but can be Orbitron */
        font-weight: 800; /* Extra bold for impact */
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.5); /* Subtle glow effect */
    }
    a {
        color: #BB86FC; /* Electric Violet - Secondary accent for links */
        text-decoration: none;
        transition: color 0.3s ease, text-shadow 0.3s ease; /* Smooth transition for hover effect */
    }
    a:hover {
        color: #00FFFF; /* Electric Cyan on hover */
        text-decoration: underline;
        text-shadow: 0 0 5px rgba(0, 255, 255, 0.7); /* More prominent glow on hover */
    }
    .stMarkdown, .stText, p {
        font-size: 17px; /* Slightly increased font size for better readability */
        line-height: 1.7; /* Increased line height for better spacing */
        color: #F0F0F0; /* Slightly brighter text for paragraphs */
    }
    /* Style for the subheader */
    .stApp > header {
        color: #E0E0E0; /* Ensure subheader text is also light */
    }
    /* Markdown quote style */
    blockquote {
        border-left: 5px solid #00FFFF; /* Accent border for quotes */
        padding-left: 15px;
        margin-left: 0;
        color: #B0B0B0; /* Slightly muted color for quotes */
        font-style: italic;
        box-shadow: 0 0 5px rgba(0, 255, 255, 0.2); /* Subtle glow for quotes */
        border-radius: 5px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    /* Horizontal rule styling */
    hr {
        border-top: 1px solid #3A3A5E; /* Muted line for separators */
        margin-top: 30px;
        margin-bottom: 30px;
    }

    /* --- Custom Styles for Skills Boxes --- */
    .skill-category {
        font-weight: bold;
        color: #BB86FC; /* Electric Violet for category titles */
        margin-top: 20px;
        margin-bottom: 8px;
        font-size: 1.2em;
        text-shadow: 0 0 5px rgba(187, 134, 252, 0.3); /* Subtle glow */
    }
    .skill-container {
        display: flex;
        flex-wrap: wrap;
        gap: 12px; /* Space between skill tags */
        margin-bottom: 25px;
    }
    .skill-tag {
        background-color: #1A1A2E; /* Darker background for skill tags */
        color: #E0E0E0;
        padding: 10px 18px;
        border-radius: 25px; /* More rounded corners for pill-like tags */
        font-size: 0.98em;
        border: 1px solid #00FFFF; /* Electric Cyan border for accent */
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.3); /* Prominent blue glow */
        transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease;
    }
    .skill-tag:hover {
        transform: translateY(-3px); /* Slight lift on hover */
        background-color: #2A2A4A; /* Slightly lighter on hover */
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.7); /* More prominent glow on hover */
    }

    /* --- Custom Styles for Project Cards --- */
    .project-card {
        background-color: #151525; /* Slightly lighter dark background for cards */
        padding: 30px;
        border-radius: 12px; /* Rounded corners for cards */
        margin-bottom: 40px; /* Space between cards */
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.5), 0 0 15px rgba(187, 134, 252, 0.3); /* Deeper shadow + violet glow */
        border: 1px solid #2A2A4A; /* Subtle border */
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .project-card:hover {
        transform: translateY(-7px); /* Lift card on hover */
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.7), 0 0 20px rgba(0, 255, 255, 0.5); /* Stronger shadow + cyan glow on hover */
    }
    .project-card h3 {
        color: #00FFFF; /* Project title color */
        margin-top: 0;
        margin-bottom: 18px;
        text-shadow: 0 0 6px rgba(0, 255, 255, 0.4); /* Subtle glow for project titles */
    }
    .project-card strong {
        color: #BB86FC; /* Electric Violet for bold text within cards */
    }
    .project-card .stMarkdown p {
        margin-bottom: 12px; /* Spacing for paragraphs within cards */
    }
    .project-card .stMarkdown ul {
        list-style-type: '⚡ '; /* Custom bullet for lists */
        margin-left: 20px;
        margin-bottom: 12px;
        color: #E0E0E0; /* Ensure list text is visible */
    }
    .project-card .stMarkdown ul li {
        margin-bottom: 6px;
    }
    .project-card .stMarkdown a {
        font-weight: bold;
        text-decoration: none; /* Remove underline by default for links in cards */
    }
    .project-card .stMarkdown a:hover {
        text-decoration: underline; /* Add underline on hover */
    }
</style>
""", unsafe_allow_html=True)

# --- Header / About Me ---
st.title("👨‍💻 Abhishek Maurya")
st.subheader("AI/ML Developer · Specializing in NLP, FastAPI & Intelligent Applications")

st.markdown("""
Hi, I'm **Abhishek Maurya** — an AI/ML Developer who loves building real-world applications with production-grade code and a strong focus on impact.
My interest in tech began back in school, but I didn’t always get the chance to showcase my skills. I’ve missed opportunities in the past — but now I’m building, learning, and thriving by solving real problems with AI.

I specialize in **Machine Learning**, **Natural Language Processing**, and intelligent backend systems using tools like **FastAPI**, **Gemini API**, **Hugging Face**, and **FAISS**.
From chatbots that understand human emotion to multilingual systems and knowledge-powered assistants — I’ve built a wide range of AI apps, end to end.

Most of my growth comes from hands-on work, not just theory. I’m currently focused on freelancing and open-source projects where I can apply AI meaningfully — not just for demos, but for tools that help people.

> If it involves **AI, challenges, and real-world impact** — I’m all in. 🚀
""")

st.header("🛠️ Skills & Tech Stack")

skills_data = {
    "Machine Learning": ["Regression", "Classification", "Clustering", "Feature Engineering", "Model Evaluation",
                         "Hyperparameter Tuning"],
    "NLP & Text Analytics": ["Text Cleaning", "Tokenization", "Sentiment Analysis", "Embedding Generation",
                             "Custom Pipelines"],
    "Generative AI & LLMs": ["Prompt Engineering", "RAG (Retrieval-Augmented Generation)", "LangChain", "Langflow",
                             "Hugging Face Transformers", "PEFT", "LoRA", "QLoRA"],
    "Model Deployment & MLOps": ["FastAPI", "Flask", "Streamlit", "CI/CD", "MLflow", "DVC", "GitHub Actions"],
    "Programming & Databases": ["Python", "SQL", "OOPs Concepts"],
    "Data & Visualization": ["Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Seaborn"],
    "Vector Databases": ["FAISS", "Astra DB", "ChromaDB"],
    "Cloud & Tools": ["AWS (EC2, S3)", "Git", "GitHub", "Postman"]
}

for category, skills in skills_data.items():
    st.markdown(f"<div class='skill-category'>{category}:</div>", unsafe_allow_html=True)
    skill_tags_html = "".join([f"<span class='skill-tag'>{skill}</span>" for skill in skills])
    st.markdown(f"<div class='skill-container'>{skill_tags_html}</div>", unsafe_allow_html=True)

st.header("🚀 Projects")

st.markdown("""
<div class="project-card">
    <h3>🧠 Sentiment-Aware Chatbot</h3>
    <p>A smart chatbot that detects the emotional tone of user messages (positive, negative, neutral) in real-time and generates responses accordingly using the Gemini API. Designed with modular FastAPI backend and Streamlit frontend.</p>
    <p><strong>Key Features:</strong></p>
    <ul>
        <li>Real-time sentiment detection using prompt-engineered LLM (Gemini)</li>
        <li>Adaptive chatbot responses based on detected emotion</li>
        <li>Modular FastAPI backend + Streamlit frontend</li>
        <li>Emotion-based analytics ready (e.g., mood trends, most common tone)</li>
    </ul>
    <p><strong>Tech Stack:</strong> FastAPI, Gemini API, Prompt Engineering, Streamlit</p>
    <p>[🔗 <a href="https://github.com/Hacke2367/ADVANCED_SENTIMENT_ANAYLZER">GitHub Repository</a>]</p>
</div>
""", unsafe_allow_html=True)

# Project 2: Multilingual Medical Q&A Chatbot (MedQuAD)
st.markdown("""
<div class="project-card">
    <h3>🩺 Multilingual Medical Q&A Chatbot (MedQuAD)</h3>
    <p>A specialized question-answering chatbot built using the MedQuAD dataset, capable of understanding and responding to medical queries in <strong>English, Hindi, Spanish, and French</strong>. It uses a combination of translation pipelines and traditional retrieval to ensure accurate and localized responses.</p>
    <p><strong>Key Features:</strong></p>
    <ul>
        <li>Trained on 15,000+ Q&A pairs from the NIH-backed MedQuAD dataset</li>
        <li>Supports multilingual questions and answers (hi, en, es, fr)</li>
        <li>Translates queries to English → retrieves → returns answer in original language</li>
        <li>Smart medical question preprocessing and fallback handling</li>
        <li>Optimized for long-answer chunked translation and streaming via Streamlit</li>
    </ul>
    <p><strong>Tech Stack:</strong> Python, Pandas, scikit-learn, TF-IDF, cosine similarity, langdetect, deep_translator, Streamlit</p>
    <p>[🔗 <a href="https://github.com/Hacke2367/Medical-Q-A-Chatbot">GitHub Repository</a>]</p>
</div>
""", unsafe_allow_html=True)

# Project 3: CS Expert Chatbot (arXiv)
st.markdown("""
<div class="project-card">
    <h3>🧠 CS Expert Chatbot (arXiv)</h3>
    <p>An intelligent chatbot that answers complex Computer Science queries by retrieving and summarizing relevant research papers from the arXiv dataset. It combines <strong>semantic search</strong>, <strong>LLM-powered explanations</strong>, and <strong>research-level summarization</strong> in a clean conversational flow.</p>
    <p><strong>Key Features:</strong></p>
    <ul>
        <li>Trained on filtered arXiv CS papers (metadata + abstracts)</li>
        <li>Retrieves papers using <code>sentence-transformers</code> + FAISS vector search</li>
        <li>Summarizes content and explains complex CS topics using Mistral-7B</li>
        <li>Handles follow-up questions and conversational flow</li>
        <li>Streamlit-based frontend with title/author/abstract preview + paper links</li>
    </ul>
    <p><strong>Tech Stack:</strong> SentenceTransformers, FAISS, Pandas, Hugging Face InferenceClient, Ollama, Streamlit</p>
    <p>📁 <strong>Download Project (Google Drive):</strong><br>
    [🔗 <a href="https://drive.google.com/file/d/1lXQhZ5ERlz1Dv9DS7AuBHtlk32cZgiRV/view?usp=sharing">CS Expert Chatbot Project Folder</a>]</p>
    <blockquote><strong>Note:</strong> This project was not uploaded to GitHub due to its large file size (dataset + embedding index). The complete code and resources are available via Google Drive.</blockquote>
</div>
""", unsafe_allow_html=True)

# Project 4: Knowledge Base Chatbot (Self-Updating)
st.markdown("""
<div class="project-card">
    <h3>📚 Knowledge Base Chatbot (Self-Updating)</h3>
    <p>An advanced AI chatbot with a modular FastAPI backend and Streamlit frontend. It delivers accurate, up-to-date answers from a curated knowledge base that updates itself using web scraping and change monitoring. Powered by Mistral LLM via Ollama and ChromaDB vector store.</p>
    <p><strong>Key Features:</strong></p>
    <ul>
        <li>Automatically monitors & scrapes trusted sources (MIT, Wikipedia, Stanford CS231n)</li>
        <li>Updates vector DB (Chroma) when content changes are detected</li>
        <li>Answers queries using Mistral via Ollama (local LLM inference)</li>
        <li>Debug tools, SQLite chat logs, and document inspection endpoint</li>
        <li>Modular architecture for scraping, embedding, retrieval, and chat handling</li>
    </ul>
    <p><strong>Tech Stack:</strong> FastAPI, Streamlit, Ollama (Mistral), ChromaDB, SentenceTransformers, BeautifulSoup, SQLite</p>
    <p>[🔗 <a href="https://github.com/Hacke2367/knowledge-base-chatbot">GitHub Repository</a>]</p>
    <blockquote><strong>Note:</strong> Project includes a robust architecture with logging, update triggers, and test scripts. Ideal for building production-ready, evolving knowledge bots.</blockquote>
</div>
""", unsafe_allow_html=True)

# Project 5: Multimodal AI Chatbot (Image + Text with Gemini)
st.markdown("""
<div class="project-card">
    <h3>🖼️ Multimodal AI Chatbot (Image + Text with Gemini)</h3>
    <p>A next-gen AI chatbot that supports both text and image interaction using the <strong>Gemini Pro API</strong> and local <strong>Stable Diffusion</strong>. Users can chat naturally, generate images from text, or upload their own images to get intelligent visual analysis — all within a single interface.</p>
    <p><strong>Key Features:</strong></p>
    <ul>
        <li>💬 <strong>AI Chat:</strong> Conversational interface powered by Gemini Pro with memory/context support</li>
        <li>🎨 <strong>Text-to-Image Generation:</strong> Uses local Stable Diffusion (Diffusers + Torch) to create images from prompts</li>
        <li>🖼️ <strong>Image Understanding:</strong> Upload an image and ask the chatbot to describe, explain, or analyze it via Gemini Pro Vision</li>
        <li>⚡ <strong>Real-Time UI:</strong> Interactive Streamlit interface with loading animations and clear responses</li>
        <li>🧱 <strong>Modular Codebase:</strong> Clean folder structure with separate logic for text, vision, and generation modules</li>
    </ul>
    <p><strong>Tech Stack:</strong> Google Gemini API (Pro + Vision), Stable Diffusion (Diffusers), Hugging Face Transformers, Streamlit, Python, Torch</p>
    <p>[🔗 <a href="https://github.com/Hacke2367/IMAGE_CHAT_BOT">GitHub Repository</a>]</p>
    <blockquote><strong>Note:</strong> This project demonstrates the integration of LLMs + Vision + Generation — ideal for futuristic chatbot applications.</blockquote>
</div>
""", unsafe_allow_html=True)

# Project 6: WhatsApp Chat Analyzer
st.markdown("""
<div class="project-card">
    <h3>📊 WhatsApp Chat Analyzer</h3>
    <p>An interactive web app that analyzes exported WhatsApp chats to extract meaningful insights. Built with NLP and visual dashboards, it provides sentiment analysis, response times, and activity patterns for both individuals and group chats.</p>
    <p><strong>Key Features:</strong></p>
    <ul>
        <li>Parses messy WhatsApp chat exports into structured data</li>
        <li>Performs sentiment analysis, keyword extraction, and user-wise activity tracking</li>
        <li>Dynamic visualizations using Matplotlib and Streamlit</li>
        <li>Regex-based cleaning pipeline handles various formats and noise</li>
    </ul>
    <p><strong>Tech Stack:</strong> Python, Streamlit, Regex, Pandas, Matplotlib, Seaborn</p>
    <p>[🔗 <a href="https://github.com/Hacke2367/Whatsapp-Chat-Analyizer">GitHub Repository</a>]</p>
</div>
""", unsafe_allow_html=True)

# Project 7: Customer Churn Prediction
st.markdown("""
<div class="project-card">
    <h3>📈 Customer Churn Prediction</h3>
    <p>A production-ready machine learning app that predicts customer churn using telecom data. Trained with SMOTE for class imbalance and deployed via Flask for real-time prediction.</p>
    <p><strong>Key Features:</strong></p>
    <ul>
        <li>Uses Random Forest + SMOTE-ENN for high-accuracy classification (94%)</li>
        <li>Real-time prediction API built with Flask</li>
        <li>EDA to uncover churn patterns (tenure, service type, etc.)</li>
        <li>Clean frontend for business usage</li>
    </ul>
    <p><strong>Tech Stack:</strong> Python, Flask, scikit-learn, Pandas, SMOTE, Matplotlib</p>
    <p>[🔗 <a href="https://github.com/Hacke2367/CHURN-PREDICTION">GitHub Repository</a>]</p>
</div>
""", unsafe_allow_html=True)

# Project 8: RAG-Based Multi-Document Chatbot
st.markdown("""
<div class="project-card">
    <h3>🧾 RAG-Based Multi-Document Chatbot</h3>
    <p>A retrieval-augmented generation chatbot capable of understanding and answering questions from multiple uploaded documents. It combines vector search and LLM-based generation in a seamless pipeline.</p>
    <p><strong>Key Features:</strong></p>
    <ul>
        <li>Upload multiple PDFs or text files and ask cross-document questions</li>
        <li>Embeds chunks using Hugging Face Transformers + LangChain</li>
        <li>Uses ChromaDB for vector search and LLM (via Hugging Face or Ollama) for response generation</li>
        <li>Asynchronous FastAPI backend with robust error handling</li>
    </ul>
    <p><strong>Tech Stack:</strong> LangChain, FastAPI, Hugging Face Transformers, ChromaDB, Python, Streamlit</p>
    <p>[🔗 <a href="https://github.com/Hacke2367/Document-Reader-Chatbot">GitHub Repository</a>]</p>
</div>
""", unsafe_allow_html=True)

# --- Optional Video Section ---
st.header("📽️ Demos")
st.markdown("You can embed short Loom videos or YouTube links of your projects here.")
st.markdown("Uploading Soon")

# --- Contact Info ---
st.header("📬 Contact Me")
st.markdown("""
**Email**: abhishekmlen25@gmail.com 
[LinkedIn](https://www.linkedin.com/in/abhishek-maurya-148542292/) | [GitHub](https://github.com/Hacke2367)
""")

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | © 2025 Abhishek Maurya")
