import json
import logging
from urllib.parse import urljoin
import requests
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from bs4 import BeautifulSoup
import PyPDF2  # for PDF parsing
import pandas as pd  # for handling tables
import numpy as np
import functools
import io
import base64

class RufusClient:
    def __init__(self, api_key=None, max_depth=2):
        """
        Initialize Rufus client with an API key and max crawl depth.
        """
        if api_key and not self.validate_api_key(api_key):
            logging.error("Invalid API key!")
            raise ValueError("Invalid API key!")
        else:
            logging.info("API key validated successfully.")
        
        self.api_key = api_key
        self.session = requests.Session()
        self.max_depth = max_depth

        # Load tokenizer and dense retrieval models
        self.query_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.passage_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        self.query_model = AutoModel.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.passage_model = AutoModel.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        
        # Initialize FAISS index for fast similarity search
        self.index = faiss.IndexFlatIP(768)  # Use dot-product similarity (Inner Product)
        self.visited_urls = set()
        self.embedded_passages = []
        self.passages_text = []

    def encode_api_key(self, api_key):
        """
        Encodes the API key to base64.
        """
        # Convert API key to bytes
        api_key_bytes = api_key.encode('utf-8')
        # Encode the API key to base64
        encoded_key = base64.b64encode(api_key_bytes).decode('utf-8')
        return encoded_key

    def validate_api_key(self, api_key):
        """
        Validates the provided API key by comparing it to the expected encoded value.
        """
        # The expected encoded API key (you can store this securely, e.g., in environment variables)
        valid_encoded_key = "cGFzc3dvcmQxMjM="  # Base64 encoding of "password123" as an example
        
        # Encode the provided API key
        encoded_api_key = self.encode_api_key(api_key)

        # Compare the provided encoded key to the expected one
        if encoded_api_key == valid_encoded_key:
            return True
        else:
            return False

    @functools.lru_cache(maxsize=100)  # Cache up to 100 query embeddings
    def embed_query(self, query, max_length=512):
        """
        Generate and cache embeddings for the query text using the query model.
        """
        tokens = self.query_tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        outputs = self.query_model(**tokens)
        query_embedding = outputs.pooler_output.detach().cpu().numpy().squeeze()
        return query_embedding

    def embed_text(self, text, is_query=False, max_length=512):
        """
        Generate embeddings for the given text using pre-trained models, splitting long texts into chunks.
        """
        tokenizer = self.query_tokenizer if is_query else self.passage_tokenizer
        model = self.query_model if is_query else self.passage_model

        # Split the text into smaller chunks if it's too long
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = tokens["input_ids"].squeeze().tolist()

        if len(input_ids) > max_length:
            chunked_input_ids = [input_ids[i:i + max_length] for i in range(0, len(input_ids), max_length)]
            embeddings = []

            # Process each chunk
            for chunk in chunked_input_ids:
                chunk_tokens = tokenizer.convert_tokens_to_string(chunk)
                inputs = tokenizer(chunk_tokens, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                outputs = model(**inputs)
                embeddings.append(outputs.pooler_output.detach().cpu().numpy())

            return np.mean(np.array(embeddings), axis=0)

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        outputs = model(**inputs)
        embeddings = outputs.pooler_output.detach().cpu().numpy()
        return embeddings

    def parse_pdf(self, pdf_url):
        """
        Parse PDF content from an online PDF URL and extract text using PyPDF2.
        """
        try:
            # Download the PDF file content
            response = requests.get(pdf_url)
            response.raise_for_status()  # Check if the request was successful

            # Open the PDF from the response content as a binary stream
            pdf_file = io.BytesIO(response.content)

            # Initialize PyPDF2 PDF reader
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # Extract text from each page
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()

            return text
        except Exception as e:
            logging.error(f"Error parsing PDF: {e}")
            return ""

    def parse_table(self, soup):
        """
        Parse and extract data from tables within the HTML content.
        """
        tables = soup.find_all("table")
        parsed_tables = []
        for table in tables:
            headers = [th.get_text().strip() for th in table.find_all("th")]
            rows = []
            for row in table.find_all("tr"):
                cols = [td.get_text().strip() for td in row.find_all("td")]
                if cols:
                    rows.append(cols)
            if rows:
                parsed_tables.append({
                    "headers": headers,
                    "data": rows
                })
        return parsed_tables

    def get_relevant_sections(self, soup):
        """
        Extract and encode sections from HTML soup and find the most relevant content.
        """
        passages = []
        tables = self.parse_table(soup)
        for table in tables:
            # Concatenate table headers and rows into a single string for embedding
            table_text = " | ".join(table["headers"]) + " | "
            for row in table["data"]:
                table_text += " | ".join(row) + " | "
            self.passages_text.append(table_text)
            table_embedding = self.embed_text(table_text)
            passages.append(table_embedding)

        sections = soup.find_all(["p", "section", "div", "article"])
        for section in sections:
            section_text = section.get_text(strip=True)
            if section_text:
                self.passages_text.append(section_text)
                passages.append(self.embed_text(section_text))

        if passages:
            self.index.add(np.array(passages).reshape(len(passages), -1))

    def crawl_website(self, url, instructions, depth=0):
        """
        Crawl the website and recursively handle nested links, including PDF links.
        """
        if depth > self.max_depth or url in self.visited_urls:
            return []

        try:
            self.visited_urls.add(url)
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            self.get_relevant_sections(soup)

            # Find PDF links and add to the index
            pdf_links = soup.find_all("a", href=True)
            for link in pdf_links:
                pdf_url = link["href"]
                if pdf_url.lower().endswith(".pdf"):
                    full_pdf_url = urljoin(url, pdf_url)
                    pdf_text = self.parse_pdf(full_pdf_url)
                    if pdf_text:
                        pdf_embedding = self.embed_text(pdf_text)
                        # Reshape pdf_embedding to 2D if necessary
                        pdf_embedding = pdf_embedding.reshape(1, -1)  
                        self.passages_text.append(pdf_text)
                        self.index.add(pdf_embedding) # Remove extra brackets to handle 2D array

            nested_links = soup.find_all("a", href=True)
            for link in nested_links:
                next_url = urljoin(url, link["href"])
                if next_url not in self.visited_urls:
                    self.crawl_website(next_url, instructions, depth + 1)
        except requests.RequestException as e:
            logging.error(f"Error in crawling {url}: {e}")
            return []

    def search_crawled_data(self, prompt):
        query_embedding = self.embed_query(prompt)
        # query_embedding shape should be (1, 768) for faiss
        query_embedding = query_embedding.reshape(1, -1)  

        k = 5
        # Check if the index is empty before searching
        if self.index.ntotal == 0:
            logging.warning("FAISS index is empty. No data to search.")
            return []
        
        _, I = self.index.search(query_embedding, k)

        relevant_data = []
        # Iterate through results using I[0] (as I is a 2D array)
        for i in I[0]:  
            if 0 <= i < len(self.passages_text): # Check if index is valid
                relevant_data.append(
                    {
                        "text": self.passages_text[i],
                        "similarity_score": _[0][I[0].tolist().index(i)]  # Get corresponding similarity score
                    }
                )
            else:
                logging.warning(f"Invalid index {i} encountered during search.")

        return relevant_data


    def scrape(self, url, prompt, depth=0):
        self.crawl_website(url, prompt, depth)
        relevant_data = self.search_crawled_data(prompt)
        return relevant_data

    def synthesize_to_json(self, data):
        """
        Convert extracted data into structured JSON format.
        """
        structured_data = {
            "query": data.get("query", ""),
            "results": []
        }

        for item in data.get("results", []):
            result = {
                "text": item.get("text", ""),
                "similarity_score": float(item.get("similarity_score", 0))
            }

            structured_data["results"].append(result)

        return json.dumps(structured_data, indent=4)



# Define the query instructions and website URL
instructions = "<your query here>"
base_url = "<website URL here>"
max_depth = 0  # Set the maximum crawl depth

# Initialize the Rufus client
try:
    client = RufusClient(api_key="password123", max_depth=max_depth)  # Replace with actual API key
except ValueError as e:
    print(e)

# Scrape the data
documents = client.scrape(base_url, instructions)

# Save output to JSON
with open("output.json", "w") as f:
    json_data = client.synthesize_to_json({"query": instructions, "results": documents})
    f.write(json_data)