# RufusClient: Web Scraping and Document Retrieval System

## Approach Summary

In this project, I built **RufusClient**, a web scraping and document retrieval system that uses advanced NLP models for text embedding and vector search. The system is designed to crawl websites, extract and process content (including PDFs and HTML tables), and retrieve relevant information based on a query. Here's an overview of the approach, challenges faced, optimization strategies, and system design principles.

---

## Key Components

1. **Web Scraping**: The system uses the `requests` library to crawl websites and the BeautifulSoup library to extract HTML content. The scraper handles nested links and visits pages up to a specified depth (`max_depth`).

2. **Text Embedding**: Leveraging pre-trained models from Hugging Face's `transformers` library, the system generates dense embeddings for queries and content (including PDFs and HTML tables). These embeddings are used for efficient similarity search.

3. **Data Parsing**:
   - **PDF Parsing**: Using `PyMuPDF` (fitz), I extract and process text from PDF files.
   - **HTML Table Parsing**: The system extracts data from HTML tables, converting headers and rows into a format that can be embedded and indexed.

4. **Similarity Search**: FAISS is used to store and search the embeddings efficiently. The system performs a similarity search on the indexed data to return the most relevant content based on the user's query.

5. **Structured Output**: The results are returned in a structured JSON format, which includes the relevant sections, tables, PDFs, and metadata such as titles, URLs, and similarity scores. This ensures that the output is clean and ready to integrate into downstream systems like LLMs.

---

## Additional Features

### PDF Parsing

Parsing PDFs was essential to the project, as many websites contain important information in this format. I used the `PyMuPDF` library to extract raw text from PDF files. Despite some challenges with text formatting, the parser handles different layouts and extracts meaningful text for further processing.

### Table Parsing

Websites often present data in HTML tables. I implemented a custom parser using BeautifulSoup to extract both headers and rows from HTML tables. This data is then combined into a single string for embedding and indexed using FAISS to allow for relevant search results based on table data.

---

## Requirements

- Python 3.11.6 or higher.
- Virtual environment for managing dependencies.

## Setup Instructions

### Step 1: Install Python 3.11.6 or Higher

Ensure you have Python 3.11.6 or higher installed on your system. You can verify this by running:

```bash
python3 --version
```

If you don’t have Python 3.11.6 or higher, please download and install it from python.org.

### Step 2: Clone the Repository

Clone the repository to your local machine using:

```bash
git clone https://github.com/XerXes100/RufusClient
```

### Step 3: Create a Virtual Environment

Create a virtual environment using `venv` or `virtualenv`. For `venv`, run:

```bash
python3 -m venv venv
```

Activate the virtual environment:

- On Windows:

```bash
venv\Scripts\activate
```

- On macOS and Linux:

```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

Install the required dependencies using `pip`:

```bash
pip3 install -r requirements.txt
```

### Step 5: Specify the website to crawl and the query

In the `rufus_client.py` file, specify the `base_url` of the website you want to crawl and the `query` you want to search for. You can also set the `max_depth` while initializing `client` to control the depth of the crawling process.

Example:

```python
query = "Extract information about research"
base_url = "https://www.cmu.edu/about/index.html"
max_depth = 0
```

**Caution**:

If you set max_depth > 0, it will significantly increase the retrieval time, as the system will crawl more pages at greater depths, potentially adding substantial overhead to the process.

### Step 6: Run the Script

Run the script using:

```bash
python3 rufus_client.py
```

The script will crawl the specified website, extract content, generate embeddings, and perform a similarity search to return relevant results based on the query. The output will be stored in a JSON file for further analysis, which can be used as an input to LLMs or other downstream applications.

---

## Evaluation

The evaluation of the system involved testing the retrieval quality and accuracy. Key steps included:
- **Manual Inspection**: I manually checked the top retrieved results to ensure they were relevant to the query.
- **Query Relevance**: I assessed the relevance and consistency of the retrieved data by inspecting how well the system’s answers matched the queries, ensuring that the responses were meaningful and aligned with user expectations.

---

## Challenges I Faced

### 1. Choosing the Retriever

One of the first decisions was choosing the appropriate retriever for the task. After testing different retrieval methods, I selected **FAISS** due to its efficiency in performing similarity searches on large datasets. FAISS is fast, supports high-dimensional vectors, and integrates well with the embedding models used in this project.

### 2. Choosing the Metric

I had to determine the best metric for evaluating relevance. Since the embeddings are dense vectors, the **dot-product (inner product)** similarity measure was chosen. This metric aligns well with the transformer-based models used in this project and provides efficient retrieval.

### 3. Choosing the Index

FAISS offers different types of indexes, such as **flat**, **inverted file**, and **product quantization** indexes. For the development phase, I chose to use the **IndexFlatIP(768)**, which provides exact search results based on inner product (dot product) similarity. This index was selected for its accuracy in retrieving precise matches and its simplicity in implementation for this use case.

The **768** in **IndexFlatIP(768)** refers to the dimensionality of the embeddings used for the search. In this project, I employed pre-trained models from Hugging Face, specifically the **DPR-question encoder** and **DPR-ctx encoder**, which generate embeddings of **768 dimensions** for both the query and the passages. The number **768** is the output size of these models, and it represents the length of the vector for each piece of text (query or passage). When using the **IndexFlatIP(768)**, the inner product similarity is computed between the query vector and the indexed passage vectors, allowing the retrieval of the most relevant passages.

The **IndexFlatIP(768)** was chosen because it directly uses these 768-dimensional embeddings for exact match retrieval. This index works well with the embeddings' dimensionality, ensuring high-quality and accurate search results. In cases where scalability or faster retrieval with a slight trade-off in accuracy is required, other indexes like **IVF** (Inverted File) or **PQ** (Product Quantization) would be more appropriate.

### 4. Dimensionality Challenges with PDF Embedding

While working with PDF files, I encountered challenges related to the dimensionality of the extracted text. PDF text often comes in a multi-dimensional structure (3D array), which is not compatible with the 2D vector-based embeddings typically used in similarity search.

To resolve this, I needed to reshape the PDF text data into a 2D array format. I implemented a strategy to transform the 3D structure of text embeddings into a 2D structure, ensuring that the extracted content could be indexed and retrieved efficiently. This reshaping process allowed the embeddings to align with the required format for FAISS indexing, ensuring that the PDF data was properly processed and could be included in the search results.

### 5. How to Evaluate?

Evaluating a retrieval system can be challenging, especially when there is no clear ground truth or benchmark. In this case, I encountered difficulties in applying traditional performance metrics due to the lack of labeled data for comparison. As a result, I relied on:
- **Manual Inspection**: I manually checked the top retrieved results to ensure they were relevant to the queries and aligned with expectations.
- **Relevance Assessment**: I evaluated the quality of the retrieved data by inspecting how well the responses matched the user queries, although this process was subjective without a clear set of performance metrics.

The challenge in evaluation stemmed from the absence of structured ground truth data, making it difficult to apply standard metrics such as precision or recall effectively.

---

## Optimization

Several optimization techniques were employed to improve both the performance and accuracy of the system:
- **Embedding Efficiency**: I implemented chunking for large texts, ensuring that documents were broken into manageable pieces before embedding them. This reduces memory usage and improves retrieval speed.

- **FAISS Indexing**: To optimize retrieval speed, I used **IndexFlatIP(768)** for its simplicity and accuracy in this project. This index leverages the 768-dimensional embeddings generated by the pre-trained models. It provides exact search results based on inner product (dot product) similarity, which is efficient for the relatively smaller dataset in this use case. While other FAISS indexing options, like **Product Quantization** (PQ), offer a trade-off between speed and recall, the **IndexFlatIP** provided optimal results for this project’s requirements.

- **Threading**: I added threading to parallelize the crawling process, which speeds up the data collection from multiple pages concurrently. This significantly reduced the overall time for crawling and indexing.

---

## API Design

The API design follows the **SOLID principles**, particularly **Single Responsibility Principle (SRP)**. Each function in the system serves a single purpose, making the code modular, maintainable, and scalable:
- **Separation of Concerns**: Different parts of the system (e.g., PDF parsing, HTML parsing, embedding, FAISS indexing) are handled by separate functions, ensuring that each component only focuses on one task.
- **Modular Code**: By modularizing the code, I ensured that each part of the system could be easily extended, tested, and maintained without affecting other components.

---

## Added Threading

To improve the performance of web scraping, I implemented threading for parallel crawling. This allows the system to visit multiple links simultaneously, reducing the time spent on collecting data from large websites. By utilizing Python’s `threading` library, I was able to add concurrent processing without introducing significant complexity into the system.

---

## Steep Learning Curve

This project was my first experience working with **Retrieval-Augmented Generation (RAG)** systems, and the learning curve was steep. I had to quickly get up to speed with several concepts and technologies, including **dense retrieval**, **embedding-based similarity search**, **FAISS indexing**, and **web scraping**. Despite the steep learning curve, I successfully completed the entire project in just **24 hours**. This fast-paced learning experience demonstrated my ability to quickly grasp new concepts, adapt to new tools, and apply them effectively to build a functioning system. It showcased my dedication to learning and problem-solving, even under time constraints.

---

## Conclusion

By combining modern NLP techniques for text embedding, vector similarity search using FAISS, and effective parsing of diverse document formats (text, PDF, tables), I built a flexible and scalable retrieval system. I faced challenges related to data diversity, choice of retrieval methods, and performance optimizations, but these were overcome through careful evaluation, performance tuning, and system design following SOLID principles.

The project is capable of scaling to large datasets and provides accurate, relevant results for user queries, making it a valuable tool for information retrieval in web-based applications.