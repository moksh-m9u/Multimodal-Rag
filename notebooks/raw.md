code
"""
%pip install -Uq "unstructured[all-docs]" 
%pip install -Uq langchain_chroma 
%pip install -Uq langchain langchain-community langchain-openai langchain-groq langchain-huggingface huggingface_hub
%pip install -Uq python_dotenv
%pip install -Uq typing

import json
from typing import List 
# unstructured document parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
# langchain components
from langchain_core.documents import Document 
from langchain_huggingface import HuggingFaceEndpointEmbeddings,ChatHuggingFace
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

"""
---
code
"""
    def partition_document(file_path: str):
        """Extract elements from PDF using unstructured"""
        print(f"Partitioning document: {file_path}")
        
        elements = partition_pdf(
            filename=file_path,  # Path to your PDF file
            strategy="hi_res", # Use the most accurate (but slower) processing method of extraction
            infer_table_structure=True, # Keep tables as structured HTML, not jumbled text
            extract_image_block_types=["Image"], # Grab images found in the PDF
            extract_image_block_to_payload=True # Store images as base64 data you can actually use
        )
        
        print(f"Extracted {len(elements)} elements")
        return elements

    # Test with your PDF file
    file_path = "../data/datasheet.pdf"  # Change this to your PDF path
    elements = partition_document(file_path)
"""

output
"""
    No languages specified, defaulting to English.
    Partitioning document: ../data/datasheet.pdf
    Loading weights: 100%|██████████| 367/367 [00:00<00:00, 4348.28it/s]
    Extracted 522 elements
"""
---
code
'''
    elements
'''
output
'''
    [<unstructured.documents.elements.Text at 0x14a6b4ec0>,
    <unstructured.documents.elements.Header at 0x14a6b4440>,
    <unstructured.documents.elements.Text at 0x14a744b90>,
    <unstructured.documents.elements.Image at 0x14a6b46e0>,
    <unstructured.documents.elements.Title at 0x14a6b4830>,
    <unstructured.documents.elements.NarrativeText at 0x14a6b4980>,
    <unstructured.documents.elements.Title at 0x14a744190>,
    <unstructured.documents.elements.Title at 0x14a7442d0>,
    <unstructured.documents.elements.Title at 0x126066190>,
    <unstructured.documents.elements.ListItem at 0x14a6b4ad0>,
    <unstructured.documents.elements.ListItem at 0x14a744550>,
    <unstructured.documents.elements.ListItem at 0x14a744690>,
    <unstructured.documents.elements.ListItem at 0x126065cd0>,
    <unstructured.documents.elements.ListItem at 0x125eef950>,
    <unstructured.documents.elements.ListItem at 0x14a803d10>,
    <unstructured.documents.elements.ListItem at 0x14a8d89e0>,
    <unstructured.documents.elements.Text at 0x14a744f50>,
    <unstructured.documents.elements.ListItem at 0x14a8d8af0>,
    <unstructured.documents.elements.NarrativeText at 0x14a744410>,
    <unstructured.documents.elements.ListItem at 0x14a667b50>,
    <unstructured.documents.elements.Title at 0x125eec8a0>,
    <unstructured.documents.elements.ListItem at 0x14a667a50>,
    <unstructured.documents.elements.ListItem at 0x14a71b5c0>,
    <unstructured.documents.elements.ListItem at 0x14a71b4d0>,
    <unstructured.documents.elements.FigureCaption at 0x14a6b4c20>,
    ...
    <unstructured.documents.elements.NarrativeText at 0x14b203f50>,
    <unstructured.documents.elements.NarrativeText at 0x14b22c0c0>,
    <unstructured.documents.elements.NarrativeText at 0x14b22c440>,
    <unstructured.documents.elements.NarrativeText at 0x14b22c7c0>,
    <unstructured.documents.elements.NarrativeText at 0x14b22cc90>]
    Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
'''

---
code
'''
    set([str(type(el)) for el in elements])
'''
output
'''
    {"<class 'unstructured.documents.elements.FigureCaption'>",
    "<class 'unstructured.documents.elements.Footer'>",
    "<class 'unstructured.documents.elements.Header'>",
    "<class 'unstructured.documents.elements.Image'>",
    "<class 'unstructured.documents.elements.ListItem'>",
    "<class 'unstructured.documents.elements.NarrativeText'>",
    "<class 'unstructured.documents.elements.Table'>",
    "<class 'unstructured.documents.elements.Text'>",
    "<class 'unstructured.documents.elements.Title'>"}
'''

---
code
'''
    def visualize_all_pages(pdf_path, elements, zoom=2, out_path="annotated.pdf"):
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        page_images = []

        for page_num in range(1, num_pages + 1):
            page = doc[page_num - 1]
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            draw = ImageDraw.Draw(img)

            img_w, img_h = img.size

            for el in elements:
                meta = el["metadata"]
                if meta.get("page_number") != page_num:
                    continue

                coords = meta.get("coordinates")
                if coords is None:
                    continue

                layout_w = coords["layout_width"]
                layout_h = coords["layout_height"]
                pts = coords["points"]

                scale_x = img_w / layout_w
                scale_y = img_h / layout_h

                xs = [p[0] * scale_x for p in pts]
                ys = [p[1] * scale_y for p in pts]
                box = [min(xs), min(ys), max(xs), max(ys)]

                color = "red" if el["type"] == "ListItem" else "blue"
                draw.rectangle(box, outline=color, width=2)
                draw.text((box[0], max(0, box[1] - 12)), el["type"], fill=color)

            page_images.append(img)
            print(f"Processed page {page_num}/{num_pages}")

        # Save all pages as a single multi-page PDF
        page_images[0].save(
            out_path,
            save_all=True,
            append_images=page_images[1:]
        )
        print(f"Saved {num_pages}-page annotated PDF to {out_path}")
    
    visualize_all_pages("../data/datasheet.pdf", elements_dict, out_path="annotated.pdf")
'''
output 
'''
    Processed page 1/29
    Processed page 2/29
    Processed page 3/29
    Processed page 4/29
    Processed page 5/29
    Processed page 6/29
    Processed page 7/29
    Processed page 8/29
    Processed page 9/29
    Processed page 10/29
    Processed page 11/29
    Processed page 12/29
    Processed page 13/29
    Processed page 14/29
    Processed page 15/29
    Processed page 16/29
    Processed page 17/29
    Processed page 18/29
    Processed page 19/29
    Processed page 20/29
    Processed page 21/29
    Processed page 22/29
    Processed page 23/29
    Processed page 24/29
    Processed page 25/29
    ...
    Processed page 27/29
    Processed page 28/29
    Processed page 29/29
    Saved 29-page annotated PDF to annotated.pdf
    Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
'''

code
'''
elements[14].to_dict()
'''
output
'''
{'type': 'ListItem',
 'element_id': '481c4b1f66e30a64be98ee1365c92177',
 'text': '; * Output and Supply TTL Compatible',
 'metadata': {'detection_class_prob': 0.775875449180603,
  'coordinates': {'points': ((np.float64(259.8442687988281),
     np.float64(941.0170288085938)),
    (np.float64(259.8442687988281), np.float64(991.5430908203125)),
    (np.float64(1112.7889404296875), np.float64(991.5430908203125)),
    (np.float64(1112.7889404296875), np.float64(941.0170288085938))),
   'system': 'PixelSpace',
   'layout_width': 2975,
   'layout_height': 3850},
  'last_modified': '2026-07-03T20:06:51',
  'filetype': 'application/pdf',
  'languages': ['eng'],
  'page_number': 1,
  'file_directory': '../data',
  'filename': 'datasheet.pdf',
  'parent_id': '73ad10aa88e34ed1448f5ac0cf5b3446'}}
'''

code
'''
# Gather all images
images = [element for element in elements if element.category == 'Image']
print(f"Found {len(images)} images")

images[4].to_dict()

# Use https://codebeautify.org/base64-to-image-converter to view the base64 text
'''
output
'''
{'type': 'Image',
 'element_id': '1365bd05f4a98d4fbb17a4ffc91d09ef',
 'text': 'GND TRIGGER OUTPUT RESET +Voc COMPAR- ATOR DISCHARGE OUTPUT THRESHOLD CONTROL VOLTAGE',
 'metadata': {'detection_class_prob': 0.8783202767372131,
  'coordinates': {'points': ((np.float64(626.8497924804688),
     np.float64(660.6483764648438)),
    (np.float64(626.8497924804688), np.float64(2125.626953125)),
    (np.float64(2339.59716796875), np.float64(2125.626953125)),
    (np.float64(2339.59716796875), np.float64(660.6483764648438))),
   'system': 'PixelSpace',
   'layout_width': 2975,
   'layout_height': 3850},
  'last_modified': '2026-07-03T20:06:51',
  'filetype': 'application/pdf',
  'languages': ['eng'],
  'page_number': 3,
  'image_base64': '/9j/4AAQS(large corpus of base64 encoding)
  'image_mime_type': 'image/jpeg',
  'file_directory': './data',
  'filename': 'datasheet.pdf'}}
'''

def create_chunks_by_title(elements):
    """Create intelligent chunks using title-based strategy"""
    print("Creating smart chunks...")
    
    chunks = chunk_by_title(
        elements, # The parsed PDF elements from previous step
        max_characters=3000, # Hard limit - never exceed 3000 characters per chunk
        new_after_n_chars=2100, # Try to start a new chunk after 2400 characters
        combine_text_under_n_chars=500, # Merge tiny chunks under 500 chars with neighbors
        isolate_table=False
    )
    
    print(f"Created {len(chunks)} chunks")
    return chunks

# Create chunks
chunks = create_chunks_by_title(elements)


Creating smart chunks...
Created 36 chunks

code
'''
def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    
    # Check for tables and images in original elements
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            
            # Handle tables
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            
            # Handle images
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    
    content_data['types'] = list(set(content_data['types']))
    return content_data
'''
Code
"""
    from langchain_openai import ChatOpenAI
    import os
    llm = ChatOpenAI(
        model="zai-org/GLM-4.5V",
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
        temperature=0,
        max_tokens=1024,
    )
"""
code
'''
def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content"""
    
    try:
        # Initialize LLM (needs vision model for images)
        
        # Build the text prompt
        prompt_text = f"""You are creating a searchable description for document content retrieval.

        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}

        """
        
        # Add tables if present
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"
        
                prompt_text += """
                YOUR TASK:
                Generate a comprehensive, searchable description that covers:

                1. Key facts, numbers, and data points from text and tables
                2. Main topics and concepts discussed  
                3. Questions this content could answer
                4. Visual content analysis (charts, diagrams, patterns in images)
                5. Alternative search terms users might use

                Make it detailed and searchable - prioritize findability over brevity.

                SEARCHABLE DESCRIPTION:"""

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add images to the message
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        print(f"     ❌ AI summary failed: {e}")
        # Fallback to simple summary
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary


'''

code
'''
def summarise_chunks(chunks):
    """Process all chunks with AI Summaries"""
    print("🧠 Processing chunks with AI Summaries...")
    
    langchain_documents = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f"   Processing chunk {current_chunk}/{total_chunks}")
        
        # Analyze chunk content
        content_data = separate_content_types(chunk)
        
        # Debug prints
        print(f"     Types found: {content_data['types']}")
        print(f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")
        
        # Create AI-enhanced summary if chunk has tables/images
        if content_data['tables'] or content_data['images']:
            print(f"     → Creating AI summary for mixed content...")
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data['text'],
                    content_data['tables'], 
                    content_data['images']
                )
                print(f"     → AI summary created successfully")
                print(f"     → Enhanced content preview: {enhanced_content[:200]}...")
            except Exception as e:
                print(f"     ❌ AI summary failed: {e}")
                enhanced_content = content_data['text']
        else:
            print(f"     → Using raw text (no tables/images)")
            enhanced_content = content_data['text']
        
        # Create LangChain Document with rich metadata
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data['text'],
                    "tables_html": content_data['tables'],
                    "images_base64": content_data['images']
                })
            }
        )
        
        langchain_documents.append(doc)
    
    print(f"Processed {len(langchain_documents)} chunks")
    return langchain_documents

output
'''
# Process chunks with AI
processed_chunks = summarise_chunks(chunks)
'''
🧠 Processing chunks with AI Summaries...
   Processing chunk 1/36
     Types found: ['image', 'text']
     Tables: 0, Images: 1
     → Creating AI summary for mixed content...
     → AI summary created successfully
     → Enhanced content preview: 
Texas Instruments LM555 Timer datasheet (SNAS548D, February 2000, revised January 2015) – Direct replacement for SE555/NE555, timing from microseconds to hours, operates in astable/monostable modes, ...
   Processing chunk 2/36
     Types found: ['image', 'text', 'table']
     Tables: 1, Images: 2
     → Creating AI summary for mixed content...
     → AI summary created successfully
     → Enhanced content preview: 
# Searchable Description: LM555 Timer IC (Texas Instruments)  

## Key Facts & Data Points  
- **Part Number**: LM555  
- **Manufacturer**: Texas Instruments (TI)  
- **Package Options & Body Sizes**...
   Processing chunk 3/36
     Types found: ['text', 'table']
     Tables: 2, Images: 0
     → Creating AI summary for mixed content...
     → AI summary created successfully
...
NOTES: (continued)

11. Laser c...
Processed 36 chunks
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
'''

code
'''
def export_chunks_to_json(chunks, filename="chunks_export.json"):
    """Export processed chunks to clean JSON format"""
    export_data = []
    
    for i, doc in enumerate(chunks):
        chunk_data = {
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content,
            "metadata": {
                "original_content": json.loads(doc.metadata.get("original_content", "{}"))
            }
        }
        export_data.append(chunk_data)
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported {len(export_data)} chunks to {filename}")
    return export_data

# Export your chunks
json_data = export_chunks_to_json(processed_chunks,filename="chunks_huggingface.json")
'''

code
'''
from langchain_huggingface import HuggingFaceEndpointEmbeddings
def create_vector_store(documents, persist_directory="dbv1/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("🔮 Creating embeddings and storing in ChromaDB...")
        
    embedding_model= HuggingFaceEndpointEmbeddings(model="ibm-granite/granite-embedding-97m-multilingual-r2")
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"✅ Vector store created and saved to {persist_directory}")
    return vectorstore

# Create the vector store
db = create_vector_store(processed_chunks)
'''

# After your retrieval
query = "explain the internal architecture"
retriever = db.as_retriever(search_kwargs={"k": 5})
chunks = retriever.invoke(query)

# Export to JSON
export_chunks_to_json(chunks, "rag_results.json")


def run_complete_ingestion_pipeline(pdf_path: str):
    """Run the complete RAG ingestion pipeline"""
    print("🚀 Starting RAG Ingestion Pipeline")
    print("=" * 50)
    
    # Step 1: Partition
    elements = partition_document(pdf_path)
    
    # Step 2: Chunk
    chunks = create_chunks_by_title(elements)
    
    # Step 3: AI Summarisation
    summarised_chunks = summarise_chunks(chunks)
    
    # Step 4: Vector Store
    db = create_vector_store(summarised_chunks, persist_directory="dbv2/chroma_db")
    
    print("Pipeline completed successfully!")
    return db