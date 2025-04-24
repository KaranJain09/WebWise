import streamlit as st
import os
import shutil
import hashlib
import requests
import json
import time
import re
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from newspaper import Article, ArticleException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY environment variable is not set. Please set it in your .env file.")
    st.stop()

# Constants
DB_DIR = "chroma_db"
CACHE_DIR = "image_cache"
MODEL_NAME = "llama-3.3-70b-versatile"  # Groq model
EMBED_MODEL = "all-MiniLM-L6-v2"  # Local embedding model

# Create cache directories if they don't exist
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Helper functions for URL and domain handling
def get_domain(url):
    """Extract the domain name from a URL."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    return domain

def format_source_name(url):
    """Format the URL for display."""
    domain = get_domain(url)
    return domain

def is_valid_url(url):
    """Check if the URL is valid."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

# Improved content extraction functions
def clean_text(text):
    """Clean extracted text by removing excessive whitespace and normalizing line breaks."""
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def extract_website_content(url):
    """Enhanced website content extraction with comprehensive error handling and retries."""
    if not is_valid_url(url):
        raise ValueError(f"Invalid URL: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        title = article.title
        main_text = clean_text(article.text)
        metadata = {
            "title": title,
            "authors": article.authors,
            "publish_date": article.publish_date,
            "url": url,
            "domain": get_domain(url)
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script in soup(["script", "style", "noscript", "iframe", "footer", "nav", "aside"]):
                script.decompose()
            
            headings = []
            for h in soup.find_all(['h1', 'h2', 'h3']):
                text = h.get_text(strip=True)
                if text and len(text) > 3 and text not in headings:
                    tag_name = h.name
                    headings.append(f"{tag_name.upper()}: {text}")
            
            meta_desc = soup.find("meta", {"name": "description"})
            description = meta_desc.get("content", "") if meta_desc else ""
            
            meta_keywords = soup.find("meta", {"name": "keywords"})
            keywords = meta_keywords.get("content", "") if meta_keywords else ""
            
            tables = []
            for i, table in enumerate(soup.find_all('table')):
                table_data = []
                for tr in table.find_all('tr'):
                    row = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if row:
                        table_data.append(row)
                
                if table_data:
                    table_title = f"Table {i+1}"
                    caption = table.find('caption')
                    if caption:
                        table_title = f"Table {i+1}: {caption.get_text(strip=True)}"
                    
                    table_text = f"{table_title}\n"
                    for row in table_data:
                        table_text += " | ".join(row) + "\n"
                    tables.append(table_text)
            
            lists = []
            for i, list_el in enumerate(soup.find_all(['ul', 'ol'])):
                if len(list_el.find_all('li')) < 3:
                    continue
                
                if any(c in list_el.get('class', []) for c in ['nav', 'menu', 'navigation']):
                    continue
                
                list_items = [li.get_text(strip=True) for li in list_el.find_all('li')]
                if list_items:
                    list_type = "Bulleted" if list_el.name == 'ul' else "Numbered"
                    list_text = f"{list_type} List:\n"
                    for idx, item in enumerate(list_items):
                        if list_el.name == 'ol':
                            list_text += f"{idx+1}. {item}\n"
                        else:
                            list_text += f"• {item}\n"
                    lists.append(list_text)
                    
            metadata.update({
                "description": description,
                "keywords": keywords
            })
            
            # Extract images
            images_data = extract_images_from_website(url, soup, response.text)
            metadata['images'] = images_data
            
            enhanced_content = ""
            if title:
                enhanced_content += f"TITLE: {title}\n\n"
            if description:
                enhanced_content += f"DESCRIPTION: {description}\n\n"
            if headings:
                enhanced_content += "HEADINGS:\n" + "\n".join(headings) + "\n\n"
            if main_text:
                enhanced_content += f"MAIN CONTENT:\n{main_text}\n\n"
            if tables:
                enhanced_content += "TABLES:\n" + "\n\n".join(tables) + "\n\n"
            if lists:
                enhanced_content += "LISTS:\n" + "\n\n".join(lists) + "\n\n"
            
            return enhanced_content, metadata
            
        except Exception as e:
            return f"TITLE: {title}\n\nMAIN CONTENT:\n{main_text}", metadata
            
    except ArticleException as e:
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script in soup(["script", "style", "noscript", "iframe", "footer"]):
                script.decompose()
            
            title = soup.title.string if soup.title else "Unknown Title"
            body_text = clean_text(soup.body.get_text()) if soup.body else ""
            
            metadata = {
                "title": title,
                "url": url,
                "domain": get_domain(url)
            }
            
            images_data = extract_images_from_website(url, soup, response.text)
            metadata['images'] = images_data
            
            return f"TITLE: {title}\n\nMAIN CONTENT:\n{body_text}", metadata
            
        except Exception as e:
            raise Exception(f"Failed to extract content from {url}: {str(e)}")
    
    except Exception as e:
        raise Exception(f"Failed to extract content from {url}: {str(e)}")

# Image handling functions
def generate_image_filename(url, image_url):
    """Generate a unique filename for an image."""
    base = hashlib.md5(url.encode()).hexdigest()[:10]
    img_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
    
    parsed_url = urlparse(image_url)
    path = parsed_url.path
    ext = os.path.splitext(path)[1].lower()
    
    if not ext or ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        ext = '.jpg'
        
    return f"{base}_{img_hash}{ext}"

def download_and_save_image(image_url, filename):
    """Download and save an image to the cache directory."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36',
        }
        response = requests.get(image_url, headers=headers, timeout=10, stream=True)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image/'):
            return None
        
        if int(response.headers.get('Content-Length', 0)) > 5 * 1024 * 1024:
            return None
        
        img_path = os.path.join(CACHE_DIR, filename)
        with open(img_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                if width < 50 or height < 50 or width > 3000 or height > 3000:
                    os.remove(img_path)
                    return None
                return img_path
        except Exception:
            if os.path.exists(img_path):
                os.remove(img_path)
            return None
            
    except Exception as e:
        return None

def extract_image_info(image_element, page_text, url):
    """Extract image info and context from the image element."""
    image_url = image_element.get('src', '')
    
    if image_url and not image_url.startswith(('http://', 'https://')):
        image_url = urljoin(url, image_url)
    
    if not image_url or not is_valid_url(image_url):
        return None
    
    if any(skip in image_url.lower() for skip in ['icon', 'logo', 'pixel.', 'tracker', 'advertisement', 'ad.', '/ad/']):
        return None
    
    alt_text = image_element.get('alt', '')
    
    caption = ""
    parent = image_element.parent
    if parent:
        figcaption = parent.find('figcaption')
        if figcaption:
            caption = figcaption.get_text(strip=True)
    
    heading = ""
    current_element = image_element
    for _ in range(5):
        if not current_element.parent:
            break
        current_element = current_element.parent
        heading_tags = current_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if heading_tags:
            heading = heading_tags[0].get_text(strip=True)
            break
    
    img_str = str(image_element)
    try:
        start_idx = max(0, page_text.find(img_str) - 100)
        end_idx = min(len(page_text), page_text.find(img_str) + len(img_str) + 100)
        surrounding_text = page_text[start_idx:end_idx].strip()
    except:
        surrounding_text = ""
    
    return {
        "url": image_url,
        "alt_text": alt_text,
        "caption": caption,
        "heading": heading,
        "surrounding_text": surrounding_text
    }

def extract_images_from_website(url, soup, page_text):
    """Extract images from a website and save them locally."""
    images_data = []
    
    img_elements = soup.find_all('img')
    
    image_count = 0
    for img in img_elements:
        if image_count >= 10:
            break
        
        image_info = extract_image_info(img, page_text, url)
        if not image_info:
            continue
            
        filename = generate_image_filename(url, image_info['url'])
        local_path = download_and_save_image(image_info['url'], filename)
        
        if local_path:
            image_info['local_path'] = local_path
            image_info['description'] = (
                image_info['caption'] or 
                image_info['alt_text'] or 
                image_info['heading'] or 
                f"Image from {get_domain(url)}"
            )
            images_data.append(image_info)
            image_count += 1
    
    return images_data

def find_relevant_images(query, metadata, max_images=3):
    """Find images relevant to the user query."""
    if not metadata or 'images' not in metadata or not metadata['images']:
        return []
    
    relevant_images = []
    query_terms = set(query.lower().split())
    
    for img in metadata['images']:
        score = 0
        
        if img['alt_text']:
            alt_terms = set(img['alt_text'].lower().split())
            score += len(query_terms.intersection(alt_terms)) * 2
        
        if img['caption']:
            caption_terms = set(img['caption'].lower().split())
            score += len(query_terms.intersection(caption_terms)) * 2
        
        if img['heading']:
            heading_terms = set(img['heading'].lower().split())
            score += len(query_terms.intersection(heading_terms)) * 3
        
        if img['surrounding_text']:
            surr_terms = set(img['surrounding_text'].lower().split())
            score += len(query_terms.intersection(surr_terms))
        
        if score > 0:
            relevant_images.append((img, score))
    
    relevant_images.sort(key=lambda x: x[1], reverse=True)
    return [img for img, score in relevant_images[:max_images]]

# Improved Groq API integration
def query_groq(prompt, history=None, temperature=0.5, max_tokens=1024):
    """Query the Groq API with a prompt and return the response."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = [{"role": "system", "content": prompt}]
    
    if history and isinstance(history, list):
        messages.extend(history)
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error querying Groq API: {str(e)}")
        return "I'm sorry, I encountered an error processing your request. Please try again."

# Streamlit UI setup with enhanced animations and design
st.set_page_config(
    page_title="Website Knowledge Assistant",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with animations and modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Theme Colors */
    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --primary-dark: #4f46e5;
        --secondary: #06b6d4;
        --accent: #f472b6;
        --bg-light: #f9fafb;
        --bg-white: #ffffff;
        --text-dark: #1e293b;
        --text-gray: #64748b;
        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --radius-sm: 0.375rem;
        --radius-md: 0.5rem;
        --radius-lg: 1rem;
    }
    
    /* General Layout */
    .stApp {
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        color: var(--text-dark);
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        padding: 2rem;
        border-radius: var(--radius-lg);
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg);
        transform: translateY(0);
        animation: fadeInDown 1s ease-out;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .main-header p {
        font-size: 1.25rem;
        opacity: 0.9;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-hxt7ib {
        background-color: var(--bg-white) !important;
        border-right: 1px solid #e5e7eb;
    }
    
    .sidebar-header {
        background: linear-gradient(90deg, var(--primary-dark) 0%, var(--primary) 100%);
        color: white;
        padding: 1.5rem 1rem;
        border-radius: var(--radius-md);
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-sm);
        animation: pulse 2s infinite ease-in-out;
    }
    
    /* Form Elements */
    .url-input {
        background-color: var(--bg-white);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        box-shadow: var(--shadow-md);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 1s ease;
        border: 1px solid #e5e7eb;
    }
    
    .url-input:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-lg);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        background: linear-gradient(90deg, var(--primary-light) 0%, var(--primary) 100%);
    }
    
    /* Chat Container */
    .chat-container {
        background-color: var(--bg-white);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow-md);
        animation: fadeIn 1s ease;
        border: 1px solid #e5e7eb;
        margin-bottom: 2rem;
    }
    
    /* Message Styling */
    .user-message {
        background: linear-gradient(90deg, var(--primary-light) 0%, var(--primary) 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 1.5rem 1.5rem 0 1.5rem;
        margin: 1rem 0;
        max-width: 80%;
        align-self: flex-end;
        box-shadow: var(--shadow-sm);
        animation: fadeInRight 0.5s ease;
    }
    
    .assistant-message {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 1rem 1.5rem;
        border-radius: 1.5rem 1.5rem 1.5rem 0;
        margin: 1rem 0;
        max-width: 80%;
        align-self: flex-start;
        box-shadow: var(--shadow-sm);
        animation: fadeInLeft 0.5s ease;
    }
    
    /* Source Tags */
    .source-tag {
        background-color: var(--primary-light);
        color: white;
        font-size: 0.75rem;
        padding: 0.2rem 0.6rem;
        border-radius: 2rem;
        margin-right: 0.5rem;
        display: inline-block;
        animation: fadeIn 0.5s ease;
    }
    
    /* Card Styling */
    .feature-card {
        background-color: var(--bg-white);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        height: 100%;
        box-shadow: var(--shadow-md);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 1s ease;
        border: 1px solid #e5e7eb;
        margin-bottom:10px;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-lg);
    }
    
    .feature-card h3 {
        color: var(--primary);
        margin-top: 0.5rem;
    }
    
    .feature-card img {
        filter: drop-shadow(0 4px 3px rgba(0, 0, 0, 0.07));
        margin-bottom: 0.5rem;
    }
    
    /* Use Case Cards */
    .use-case-card {
        background-color: var(--bg-white);
        border-radius: var(--radius-md);
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-sm);
        border-left: 4px solid var(--primary);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .use-case-card:hover {
        transform: translateX(5px);
        box-shadow: var(--shadow-md);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes fadeInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(99, 102, 241, 0); }
        100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
    }
    
    /* Loading Animation */
    .loader {
        border: 3px solid #f3f3f3;
        border-top: 3px solid var(--primary);
        border-radius: 50%;
        width: 25px;
        height: 25px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Status Messages */
    .error-message {
        color: #ef4444;
        background-color: #fee2e2;
        padding: 1rem;
        border-radius: var(--radius-md);
        margin-bottom: 1rem;
        border-left: 4px solid #ef4444;
        animation: fadeIn 0.5s ease;
    }
    
    .success-message {
        color: #10b981;
        background-color: #d1fae5;
        padding: 1rem;
        border-radius: var(--radius-md);
        margin-bottom: 1rem;
        border-left: 4px solid #10b981;
        animation: fadeIn 0.5s ease;
    }
    
    .info-message {
        color: #3b82f6;
        background-color: #dbeafe;
        padding: 1rem;
        border-radius: var(--radius-md);
        margin-bottom: 1rem;
        border-left: 4px solid #3b82f6;
        animation: fadeIn 0.5s ease;
    }
    
    /* Chat Input */
    .stTextInput input {
        border-radius: 2rem !important;
        padding: 0.75rem 1.5rem !important;
        border: 2px solid #e5e7eb !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    .stTextInput input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Images */
    img {
        border-radius: var(--radius-md);
        transition: transform 0.3s ease;
    }
    
    img:hover {
        transform: scale(1.05);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-light);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary);
    }
    
    /* Misc */
    .stExpander {
        border: 1px solid #e5e7eb;
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
    }
    
    .st-emotion-cache-16txtl3 h1 {
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="main-header">
    <h1>🔮 Website Knowledge Assistant</h1>
    <p>Discover the power of instant website analysis - just add URLs and ask away!</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "message_history" not in st.session_state:
    st.session_state.message_history = []
if "websites" not in st.session_state:
    st.session_state.websites = {}
if "website_metadata" not in st.session_state:
    st.session_state.website_metadata = {}
if "websites_processed" not in st.session_state:
    st.session_state.websites_processed = False
if "processing_errors" not in st.session_state:
    st.session_state.processing_errors = {}
# Initialize session state if not already present
if 'cleaned' not in st.session_state:
    st.session_state.cleaned = False

# Enhanced Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header"><h3>🌟 Control Center</h3></div>', unsafe_allow_html=True)
    
    with st.form("url_form", clear_on_submit=False):
        st.markdown("### 🔗 Add Websites")
        urls = st.text_area(
            "Enter website URLs (one per line):", 
            placeholder="https://example.com\nhttps://another-site.com", 
            height=150
        )
        with st.expander("⚙️ Advanced Options"):
            chunk_size = st.slider(
                "Chunk Size", 
                min_value=200, 
                max_value=1000, 
                value=500, 
                help="Size of text chunks for processing"
            )
            chunk_overlap = st.slider(
                "Chunk Overlap", 
                min_value=50, 
                max_value=400, 
                value=200,
                help="Overlap between chunks"
            )
            temperature = st.slider(
                "Response Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.2, 
                step=0.1,
                help="Lower for focused responses, higher for creativity"
            )
        submitted = st.form_submit_button("✨ Process Websites", type="primary", use_container_width=True)
        clear_btn = st.form_submit_button("🗑️ Clear", type="secondary", use_container_width=True)
        if clear_btn:
            urls = ""
    
    if submitted:
        url_list = [url.strip() for url in urls.split("\n") if url.strip()]
        
        if not url_list:
            st.sidebar.error("⚠️ Please enter at least one URL.")
        else:
            if os.path.exists(DB_DIR):
                shutil.rmtree(DB_DIR)
            os.makedirs(DB_DIR, exist_ok=True)
            if os.path.exists(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR, exist_ok=True)
            
            st.session_state.websites = {}
            st.session_state.website_metadata = {}
            st.session_state.processing_errors = {}
            
            # progress_placeholder = st.sidebar.empty()
            # status_text = st.sidebar.empty()
            
            for i, url in enumerate(url_list):
                progress_bar = st.sidebar.progress(0)
                status_text = st.sidebar.empty()
                
                # For simulation purposes:
                try:
                    if not is_valid_url(url):
                        raise ValueError("Invalid URL format")
                    
                    status_text.text(f"Processing: {url}")
                    
                    content, metadata = extract_website_content(url)
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        separators=["\n\n", "\n", ". ", " ", ""]
                    )
                    chunks = text_splitter.split_text(content)
                    
                    if not chunks:
                        raise ValueError("No content could be extracted")
                    
                    documents = []
                    for idx, chunk in enumerate(chunks):
                        section = "unknown"
                        if "TITLE:" in chunk:
                            section = "title"
                        elif "DESCRIPTION:" in chunk:
                            section = "description"
                        elif "HEADINGS:" in chunk:
                            section = "headings"
                        elif "MAIN CONTENT:" in chunk:
                            section = "main_content"
                        elif "TABLES:" in chunk:
                            section = "tables"
                        elif "LISTS:" in chunk:
                            section = "lists"
                        
                        documents.append(
                            Document(
                                page_content=chunk,
                                metadata={
                                    "source": url, 
                                    "domain": get_domain(url),
                                    "title": metadata.get("title", ""),
                                    "section": section,
                                    "url": url,
                                    "chunk_id": idx
                                }
                            )
                        )
                    
                    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
                    url_hash = hashlib.md5(url.encode()).hexdigest()
                    db_path = os.path.join(DB_DIR, url_hash)
                    
                    vectordb = Chroma.from_documents(
                        documents,
                        embeddings,
                        persist_directory=db_path
                    )
                    
                    st.session_state.websites[url] = vectordb
                    st.session_state.website_metadata[url] = metadata
                    
                    progress_bar.progress((i + 1) / len(url_list))
                    
                except Exception as e:
                    st.session_state.processing_errors[url] = str(e)
                    st.sidebar.error(f"❌ Failed to process {url}: {str(e)}")
                
                # The user will add the actual processing logic here
                
            # progress_placeholder.empty()
            # status_text.empty()
            
            if st.session_state.websites:
                st.session_state.websites_processed = True
                st.sidebar.success(f"✅ Successfully processed {len(st.session_state.websites)} websites!")
                
                if st.session_state.processing_errors:
                    with st.sidebar.expander(f"⚠️ {len(st.session_state.processing_errors)} Processing Errors"):
                        for url, error in st.session_state.processing_errors.items():
                            st.error(f"{url}: {error}")
            else:
                st.sidebar.error("⚠️ No websites were successfully processed.")
    
    if st.session_state.websites:
        st.markdown("### 📊 Processed Websites")
        for i, url in enumerate(st.session_state.websites):
            domain = format_source_name(url) if 'format_source_name' in globals() else url
            title = st.session_state.website_metadata[url].get('title', 'Unknown')
            
            expander_label = f"✅ {domain}"
            with st.expander(expander_label):
                st.markdown(f"**Title:** {title}")
                st.markdown(f"**URL:** [Visit Site]({url})")
        
        if st.button("🧹 Clear All Data", use_container_width=True):
                # Perform your cleaning operations here
            st.session_state.cleaned = True
            st.session_state.websites = {}
            st.session_state.website_metadata = {}
            st.session_state.websites_processed = False
            st.session_state.chat_history = []
            st.session_state.message_history = []
            if os.path.exists(DB_DIR):
                shutil.rmtree(DB_DIR)
            os.makedirs(DB_DIR, exist_ok=True)
            if os.path.exists(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR, exist_ok=True)
            st.rerun()
# Main chat interface
if st.session_state.websites_processed:
    st.markdown('<div class="sub-header"><h2>💬 Chat with your websites</h2></div>', unsafe_allow_html=True)
    
    # Example questions in animated cards
    with st.expander("💡 Example questions you can ask"):
        cols = st.columns(3)
        
        examples = [
            {"icon": "🔍", "title": "General", "text": "What is this website about?"},
            {"icon": "📝", "title": "Summary", "text": "Provide a summary of this website."},
            {"icon": "🎯", "title": "Specific", "text": "What does the website say about [topic]?"},
            {"icon": "⚖️", "title": "Comparison", "text": "Compare information across these websites."},
            {"icon": "🧩", "title": "Details", "text": "What are the main features mentioned?"},
            {"icon": "🏗️", "title": "Structure", "text": "How is the content organized?"},
            {"icon": "🖼️", "title": "Images", "text": "Show me images related to [topic]."},
            {"icon": "📊", "title": "Data", "text": "What statistics or numbers are mentioned?"},
            {"icon": "🔗", "title": "Links", "text": "What external resources are linked?"}
        ]
        
        for i, example in enumerate(examples):
            col = cols[i % 3]
            with col:
                st.markdown(f"""
                <div class="use-case-card">
                    <div style="font-size: 1.25rem; margin-bottom: 0.5rem;">{example['icon']} <strong>{example['title']}</strong></div>
                    <div>{example['text']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat container with animation
    chat_container = st.container()
    
    with chat_container:
        for i, chat in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(chat["question"])
            
            with st.chat_message("assistant", avatar="🔮"):
                st.write(chat["answer"])
                
                if "sources" in chat and chat["sources"]:
                    st.markdown("**Sources:**")
                    source_html = ""
                    for source in chat["sources"]:
                        source_name = format_source_name(source) if 'format_source_name' in globals() else source
                        source_html += f"<span class='source-tag'>{source_name}</span>"
                    st.markdown(f"{source_html}", unsafe_allow_html=True)
    
    # Animated chat input
    user_query = st.chat_input("Ask a question about these websites...", key="chat_input")
    
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        
        with st.chat_message("assistant"):
            typing_placeholder = st.empty()
            typing_placeholder.markdown("*Thinking...*")
            
            start_time = time.time()
            
            all_contexts = []
            sources = set()
            
            target_website = None
            for url in st.session_state.websites:
                domain = get_domain(url)
                if domain.lower() in user_query.lower():
                    target_website = url
                    break
            
            websites_to_search = [target_website] if target_website else st.session_state.websites.keys()
            
            st.session_state.message_history.append({"role": "user", "content": user_query})
            
            for url in websites_to_search:
                vectordb = st.session_state.websites[url]
                metadata = st.session_state.website_metadata[url]
                
                search_query = user_query
                
                if any(term in user_query.lower() for term in ["summary", "summarize", "what is this website about", "overview"]):
                    search_query = f"{user_query} main content overview summary website purpose"
                
                elif any(term in user_query.lower() for term in ["what does", "how does", "tell me about", "specific"]):
                    match = re.search(r"about\s+(.+?)(?:\s+in|\s+on|\s+at|\s+\?|$)", user_query.lower())
                    if match:
                        specific_topic = match.group(1)
                        search_query = f"{user_query} {specific_topic} details information"
                
                is_image_related = any(term in user_query.lower() for term in 
                                      ["image", "picture", "photo", "figure", "diagram", "graph", "show me", "visual"])
                if is_image_related:
                    search_query = f"{user_query} visual visual_content image picture photo figure"
                
                try:
                    relevant_docs = vectordb.max_marginal_relevance_search(
                        search_query, 
                        k=8, 
                        fetch_k=15, 
                        lambda_mult=0.7
                    )
                except:
                    relevant_docs = vectordb.similarity_search(search_query, k=8)
                
                if relevant_docs:
                    sources.add(url)
                    
                    domain = get_domain(url)
                    title = metadata.get("title", domain)
                    
                    sections = {}
                    for doc in relevant_docs:
                        section = doc.metadata.get('section', 'unknown')
                        if section not in sections:
                            sections[section] = []
                        sections[section].append(doc.page_content)
                    
                    context_text = f"SOURCE: {domain}\nTITLE: {title}\n\n"
                    section_order = ['title', 'description', 'headings', 'main_content', 'tables', 'lists', 'unknown']
                    
                    for section_name in section_order:
                        if section_name in sections:
                            if section_name in ['title', 'description']:
                                context_text += f"{sections[section_name][0]}\n\n"
                            else:
                                context_text += "\n".join(sections[section_name]) + "\n\n"
                    
                    all_contexts.append(context_text)
            
            if not all_contexts:
                response_text = "I don't have enough information to answer this question based on the websites you've provided. Could you rephrase your question or provide more websites with relevant information?"
                sources = []
            else:
                prompt_template = """You are an expert website analyst and information retriever. You help users understand website content without them having to read the entire site.

Your goal is to provide PRECISE, ACCURATE and DIRECTLY RELEVANT answers based ONLY on the information from the websites provided.

Follow these instructions carefully:

1. Answer ONLY based on the website content provided below - NEVER make up information
2. Be specific and detailed when answering questions about website content
3. If asked for a summary, provide a concise but comprehensive overview
4. If asked about specific information, focus precisely on that topic
5. Always cite which website provided the information you're sharing
6. If the answer isn't in the provided content, say so clearly
7. Format your answers with clear headings and structure when appropriate
8. Match your answer's length to the complexity of the question
9. Use bullet points for lists and clear formatting for better readability
10. For complex topics, organize the information in a logical flow

Website information:
{context}

Remember: Your purpose is to help users quickly extract and understand website content without reading it themselves.
"""
                context_text = "\n\n=====WEBSITE INFORMATION=====\n\n".join(all_contexts)
                
                response_text = query_groq(
                    prompt=prompt_template.format(context=context_text),
                    history=st.session_state.message_history[-10:],
                    temperature=temperature,
                    max_tokens=1200
                )
            
            st.session_state.message_history.append({"role": "assistant", "content": response_text})
            
            if len(st.session_state.message_history) > 20:
                st.session_state.message_history = st.session_state.message_history[-20:]
            
            response_time = time.time() - start_time
            
            st.session_state.chat_history.append({
                "question": user_query,
                "answer": response_text,
                "sources": list(sources)
            })
            
            typing_placeholder.empty()
            st.write(response_text)
            
            all_relevant_images = []
            for url in sources:
                metadata = st.session_state.website_metadata.get(url, {})
                relevant_images = find_relevant_images(user_query, metadata)
                all_relevant_images.extend(relevant_images)

            if all_relevant_images:
                st.markdown("**Related Images:**")
                img_cols = st.columns(min(len(all_relevant_images), 3))
                
                for i, img in enumerate(all_relevant_images):
                    col = img_cols[i % len(img_cols)]
                    with col:
                        try:
                            local_path = img['local_path']
                            if os.path.exists(local_path):
                                image = Image.open(local_path)
                                st.image(image, caption=img['description'], use_container_width=True)
                        except Exception as e:
                            pass
            
            if sources:
                st.markdown("**Sources:**")
                for source in sources:
                    st.markdown(f"<span class='source-tag'>{format_source_name(source)}</span>", unsafe_allow_html=True)
            
            st.caption(f"Response time: {response_time:.2f} seconds")

else:
    # Welcome screen with animations and modern design
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div style="padding: 2rem; animation: fadeIn 1.5s ease;">
            <h1>🔮 Discover Website Insights Instantly</h1>
            <p style="font-size: 1.2rem; margin-bottom: 2rem;">
                Skip reading entire websites - just add URLs and I'll analyze them for you!
                Ask questions naturally and get accurate, concise answers.
            </p>
            </>
            <div class="info-message">
                <h3>💫 Getting Started</h3>
                <ol>
                    <li>Add website URLs in the sidebar</li>
                    <li>Click "Process Websites" to extract content</li>
                    <li>Ask questions about the websites in the chat</li>
                </ol>
                <p><strong>Pro Tip:</strong> For best results, add related websites to compare information across sources.</p>
            </div>
            </>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="margin-top:70px ;padding: 1rem; animation: fadeInRight 1.5s ease;">
            <img src="https://img.icons8.com/fluency/240/000000/web.png" style="width: 180px; margin: 0 auto; display: block;">
            <div style="background: linear-gradient(135deg, var(--primary-light) 0%, var(--primary) 100%); color: white; padding: 1.5rem; border-radius: 1rem; margin-top: 2rem; box-shadow: var(--shadow-lg);">
                <h3>✨ AI-Powered Website Analysis</h3>
                <p>Let me handle the reading so you can focus on understanding the key information!</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header"><h2>✨ Amazing Features</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    feature_cards = [
        {
            "icon": "https://img.icons8.com/fluency/48/000000/search--v1.png",
            "title": "Smart Extraction",
            "description": "Automatically extracts and organizes content from websites including tables, lists, and images."
        },
        {
            "icon": "https://img.icons8.com/fluency/48/000000/chat--v1.png",
            "title": "Natural Conversations",
            "description": "Ask questions in plain language and get concise, accurate answers from the website content."
        },
        {
            "icon": "https://img.icons8.com/fluency/48/000000/time-machine.png",
            "title": "Time Saving",
            "description": "Get information instantly without reading entire websites - saving you valuable time and energy."
        },
        {
            "icon": "https://img.icons8.com/fluency/48/000000/picture.png",
            "title": "Visual Content",
            "description": "View relevant images from the websites based on your queries and interests."
        },
        {
            "icon": "https://img.icons8.com/fluency/48/000000/compare.png",
            "title": "Cross-Site Analysis",
            "description": "Compare information across multiple websites to get comprehensive insights."
        },
        {
            "icon": "https://img.icons8.com/fluency/48/000000/reading.png",
            "title": "Intelligent Summaries",
            "description": "Get concise summaries of website content tailored to your specific interests."
        }
    ]
    
    cols = [col1, col2, col3]
    for i, feature in enumerate(feature_cards):
        col = cols[i % 3]
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <img src="{feature['icon']}" width="48">
                <h3>{feature['title']}</h3>
                <p>{feature['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header"><h2>🚀 Popular Use Cases</h2></div>', unsafe_allow_html=True)
    
    use_cases = {
        "Research": "Compare information across multiple research papers or academic websites.",
        "Shopping": "Extract product features and reviews from e-commerce sites.",
        "News": "Summarize and compare news coverage across different publications.",
        "Learning": "Extract key concepts from educational websites for faster learning.",
        "Business": "Analyze competitor websites to understand their offerings.",
        "Travel": "Compare destinations and attractions across travel sites."
    }
    
    cols = st.columns(3)
    for i, (title, desc) in enumerate(use_cases.items()):
        col = cols[i % 3]
        with col:
            st.markdown(f"""
            <div class="use-case-card">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 2rem; background: linear-gradient(135deg, var(--primary-light) 0%, var(--primary) 100%); color: white; border-radius: 1rem; animation: pulse 2s infinite ease-in-out;">
        <h2>Ready to explore websites effortlessly?</h2>
        <p style="font-size: 1.2rem; margin-bottom: 1rem;">Add your first website in the sidebar and get started!</p>
        <div style="font-size: 2rem; margin-top: 1rem;">👈</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; padding: 20px; font-size: 0.8rem; color: #6B7280;">
        <p>Powered by Groq LLM API and Streamlit</p>
        <p>Using embeddings from Hugging Face</p>
    </div>
    """, unsafe_allow_html=True)

    # # Add footer with app info
    # st.sidebar.markdown("""
    # <div style="position: fixed; bottom: 0; width: 17%; background-color: #f5f7f9; padding: 10px; font-size: 0.8rem; color: #6B7280; text-align: center;">
    #     <p>Website Knowledge Assistant v1.0</p>
    #     <p>Built with Streamlit + Groq + LangChain</p>
    # </div>
    # """, unsafe_allow_html=True)