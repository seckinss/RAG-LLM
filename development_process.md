**Step 1: Implementing the Data Ingestion Pipeline**

My first task was building a robust data ingestion pipeline. To create a diverse knowledge base, I began by fetching e-books from Project Gutenberg, using classic literature like Jane Austen's "Pride and Prejudice," Mary Shelley's "Frankenstein," and Sun Tzu's "The Art Of War." I encapsulated all the logic within my `TextPreprocessor` class. To handle different document formats, I wrote functions for both plain text and PDFs. For PDFs, I implemented a basic fallback strategy: the code first tries to extract text with one library, and if that fails, it automatically falls back to a second one to maximize success.

After extracting the raw text, the next step was cleaning and chunking it. To break the text into meaningful pieces, I chose to implement a recursive splitting strategy. My code splits text along natural semantic boundaries like paragraphs and sentences before resorting to a fixed character limit. This was a critical design choice because it produces more coherent chunks for the embedding model to process. The final output I designed for this stage was a directory of JSON files, where each file contains the text chunks and source metadata for one of the original documents.

**Step 2: Creating the Embedding and Storage Workflow**

With the text processed, I implemented the workflow to convert these chunks into vector embeddings. The core of this stage is my `DocumentEmbedder` class. Inside this class, I have used `all-MiniLM-L6-v2` as embedding model as suggested. For the vector database, `ChromaDB` making the system self-contained and portable.

I had to solve handling a large number of documents without running out of memory. I addressed this by implementing a batch-processing system. My script loads the processed JSON chunks, feeds them to the embedding model in batches of 32, and then adds them to the vector database. I specifically used an "upsert" operation for this, an implementation detail that makes the process idempotent—I can re-run the script on updated documents without creating duplicate entries. While I didn't use that many big files. It could help if we need to use larger files.

**Step 3 & 4: Designing an Abstracted Retriever**

Before tackling the retrieval logic, I made a key architectural decision: to abstract the vector database interaction. I created a `VectorStore` class to serve as a dedicated interface. This solution decouples the rest of my application from the specific database, meaning I could swap out the backend database technology in the future by only changing this one class.

This `VectorStore` served as the foundation for my `Retriever` class. This is where I implemented the core RAG retrieval logic. When a user query is received, my `retrieve` method first generates a vector embedding for the query using the same model from the ingestion step. It then passes this vector to the `VectorStore` to fetch the most similar document chunks using cosine similarity. I added a final filtering step to discard any chunks below a configurable relevance threshold, guaranteeing that only high-quality context is passed to the language model.

**Step 5: Implementing the Language Model Manager**

My implementation for the "Generation" stage is encapsulated within the `LLMManager` class. Here, I wrote the logic to load a large language model and manage its lifecycle. A core feature I built was hardware auto-detection; my code checks for CUDA, MPS, or CPU availability and configures the model to run on the best available device. To make the system work on consumer hardware, I personally integrated support for model quantization, adding parameters to my loading function that can dynamically load the model in 4-bit or 8-bit precision.

The most critical part of working with LLMs is prompt engineering. I created a `_format_prompt` method that programmatically constructs a detailed prompt. It combines a system message (currently a constant value I plan to make dynamic in later versions), the retrieved documents from the `Retriever`, and the user's query into the precise format that the instruction-tuned model expects.

**Step 6 & 7: Building the API and UI**

To make the system functional, I implemented a two-part serving solution. First, I built a backend API to expose the RAG pipeline. I defined several endpoints: `/health` for diagnostics, `/retrieve` to test the retrieval component independently, and `/query` for the full end-to-end process. For performance, a crucial detail was how I implemented model loading. Instead of loading the heavy models at server startup, I wrapped their initialization functions in a cache decorator. This solution ensures the models are loaded into memory only once, on the first request that needs them.

For the user interface, my solution was to implement a completely separate frontend application that communicates with my backend API. This client-server architecture was a deliberate choice to decouple the UI from the heavy model inference. In the UI code, I implemented a chat interface, a settings sidebar for tweaking parameters like `top_k` and `temperature`, and an expandable section to display the source documents for each answer. I felt this feature was important for providing transparency into how the system arrives at an answer.

After finishing the project I have tried to run it on my MacBook Air M2 device, but I have got the response for my question in ~30 minutes. After that I have decided to transfer my backend part into the colab with T4 GPU. I have read instructions on how to 