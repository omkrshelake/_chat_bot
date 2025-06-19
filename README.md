
#  Hybrid Document QA App

A Streamlit-based Question Answering (QA) system that supports:

- All files are using**TF-IDF-based search**
-  Uses **`microsoft/xdoc-base-squad2.0`** transformer model for extracting answers
-  Fast and intuitive UI for asking questions based on uploaded documents
---

##  Supported File Types

- `.csv` (TF-IDF)
- `.pdf`, `.docx`, `.txt` (Embeddings)

---

##  Features

- Intelligent QA from **your own documents**
- Automatic detection of file type
- Uses **transformers** and **sentence-transformers** under the hood
- Embedding model: `all-MiniLM-L6-v2`
- Built with **Streamlit** for instant deployment

---

##  Installation

Create a virtual environment (optional but recommended):

```bash
python -m venv myenv
myenv\Scripts\activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

<details>
<summary>📋 <strong>requirements.txt</strong></summary>

```
streamlit
torch
transformers
pandas
scikit-learn
sentence-transformers
PyPDF2
python-docx
langchain
```

</details>

---

## Run the App

```bash
streamlit run app.py
```

---

##  Example Usage

1. Upload your file (CSV / PDF / DOCX / TXT)
2. Type a question (e.g., “What is the name for ABPER004?”)
3. Get precise answers from the document!

---

##  Project Structure

```
project-root
├── .env 
├── app.py               
├── requirements.txt     
└── README.md    
 
```

---

##  Models Used

| Purpose               | Model                                    |
|----------------------|-------------------------------------------|
| Question Answering   | `microsoft/xdoc-base-squad2.0`            |


---

##  Notes

- TF-IDF is used for **structured CSV data**.
- Embedding-based retrieval is used for **unstructured files (PDF, TXT, DOCX)**.
- The app automatically selects the method based on file type.


---

##  Future Improvements

- [ ] Add multi-file support
- [ ] Add chat history memory
- [ ] Support for image-based PDFs (OCR)
- [ ] Fine-tune with custom datasets

---

##  Author

**Omkar Shelake**
*Generative AI & Azure Enthusiast**
