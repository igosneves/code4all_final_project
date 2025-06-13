import gradio as gr
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
import re
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
from langchain_chroma import Chroma
from dotenv import load_dotenv
import tempfile

# Load environment variables from .env file
load_dotenv()

# Now we can access the API key from .env
embeddings_model = OpenAIEmbeddings(
    model = "text-embedding-3-large"
)

vector_db = Chroma(
    collection_name="multiriscos",
    embedding_function=embeddings_model,
    persist_directory="db_seguros",  # Where to save data locally, remove if not necessary
)

messages = []

FAQ_URL = "https://www.generalitranquilidade.pt/perguntas-frequentes"

system_prompt = """
    És um assistente de seguros, que se chama Zé Cautela, que responde a perguntas sobre cobertura dos seguros que tens no teu contexto.

    Instruções:
    Response à pergunta do utilizador (user query) baseado no contexto fornecido.
    Se a resposta para a pergunta não estiver no contexto, verifica se está na história da conversa. Se estiver, responde baseado nessa informação.
    Se após verificar o contexto e a história da conversa ainda não conseguires responder, não uses o teu conhecimento interno, responde com 'Não tenho o contexto necessário para essa pergunta'.
    Se as tuas respostas forem do contexto, adiciona o número do capítulo, da cláusula e das páginas (caso estes existam) que usaste como referência para a tua resposta, num parágrafo separado.
    Se a tua resposta remeter o utilizador para informação presenta nas condições particulares, sugere-lhe que carregue o ficheiro de condições particulares no chat.
    Se o utilizador carregar com sucesso um ficheiro de condições particulares, pede-lhe o Número de Apólice no chat.
    Se a pergunta for sobre condições particulares ou coberturas, só podes responder se tiveres o "Policy number" (número da apólice) no contexto com uma mensagem do humano "human" que corresponde ao número da apólice "Policy number" que está no documento carregado por ele.
    É importante que apenas respondas sobre condições particulares e valores de capitais se no teu histórico tiveres uma mensagem do humano "human" que corresponde ao número da apólice "Policy number" que está no documento carregado por ele.
    Se não tiveres contexto com esse número de apólice, response que não tens esse número de apólice no contexto.
    Nas tuas respostas nunca incluas dados pessoais, como nome, email, número de telefone, morada, data de nascimento, cartão de cidadão. Dados específicos de seguro como valores de capitais e números de apólice podes escrever.
    Responde sempre em português de Portugal.
"""

messages.append(
    ("system", system_prompt)
)

def documentsFromPdfFile(file_path):
  loader = PyPDFLoader(file_path, mode="single")
  documents = loader.load()
  return documents

def documentsFromWebSite(url):
  loader = WebBaseLoader(url)
  documents = loader.load()
  return documents

def cleanDocuments(documents):
  for document in documents:
    document.page_content = re.sub('\n', ' ', document.page_content)

def validDocument(documents):
    if not documents:  # Check if documents list is empty
        return False

    # Get the first document's content and convert to lowercase for case-insensitive comparison
    first_doc_content = documents[0].page_content.lower()

    # Check for both required strings in the first document.
    # TODO improve this
    has_condicoes = "condições particulares" in first_doc_content
    has_apolice = "n.ºapólice" in first_doc_content

    return has_condicoes and has_apolice

def removeUselessWebChunks(chunks):
  # remove all chunks until the first ocurrence of "Ver todas as perguntas frequentes"
  newChunks = []
  for chunk in chunks:
    if "Ver todas as perguntas frequentes" in chunk.page_content:
      newChunks = chunks[chunks.index(chunk):]
      break
  return newChunks

def getChunks(documents, chunk_size, chunk_overlap):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap
  )
  chunks = text_splitter.split_documents(documents)

  return chunks

def addMetadataToChunks(chunks):
    currentChapter = ""
    currentPage = "1"
    for chunk in chunks:
        # Extract page number using regex
        page_match = re.search(r'(\d+)\s*Mod\.\s*465\.031\s*–\s*janeiro\s*2025', chunk.page_content)
        if page_match:
            currentPage = page_match.group(1)  # group(1) gets the first captured group (the number)

        if "CAPÍTULO" in chunk.page_content:
            currentChapter = chunk.page_content.split("CAPÍTULO")[1].split(" ")[1].strip()

        # Add both chapter and page to metadata
        chunk.metadata = {
            "chapter": currentChapter,
            "page": currentPage,
            "type": "pdf"
        }

def extractPolicyNumberFromParticularConditions(documents):
  for document in documents:
    policy_number = re.search(r'N\.ºApólice:\s*(\d+)Data', document.page_content)
    if policy_number:
      return policy_number.group(1)
  return None

def addWebMetadataToParticularConditionsChunks(chunks, policy_number):
  for chunk in chunks:
      # Extract page number if it matches the format "Pág. X de Y"
      page_match = re.search(r'Pág\.\s*(\d+)\s*de\s*\d+', chunk.page_content)
      if page_match:
          currentPage = page_match.group(1)  # Extract the page number
      else:
          currentPage = "1"  # Default to page 1 if not found

      chunk.metadata = {
          "type": "web",
          "page": currentPage,
          "policy_number": policy_number
      }

def removeUselessParticularConditionsChunks(chunks):
  # remove all chunks that are from page 1
  newChunks = []
  for chunk in chunks:
    if chunk.metadata["page"] != "1":
      newChunks.append(chunk)
  return newChunks

def addWebMetadataToChunks(chunks):
    for chunk in chunks:
        chunk.metadata = {
            "type": "pdf",
            "page": FAQ_URL
        }

# Store the chunks in the vector db
def storeChunks(chunks):
  global vector_db
  ids = vector_db.add_documents(chunks)
  return ids

def ingestionStage():
  # Skip ingestion stage if we already have the data in the vector db
  global vector_db
  if len(vector_db.get()['documents']) > 0:
    print("Data already in vector db, skipping ingestion stage")
    return

  # Phase 1 load (Pre-RAG)

  ### Step 1.1 Document Load
  pdfDocuments = documentsFromPdfFile("seguro.pdf")
  webDocuments = documentsFromWebSite(FAQ_URL)
  ### Step 1.2 Document Transformation

  # Intermediate clean up PDF
  cleanDocuments(pdfDocuments)
  chunks = getChunks(pdfDocuments, 1100, 100)
  addMetadataToChunks(chunks)

  # Intermediate clean up Web
  cleanDocuments(webDocuments)
  webChunks = getChunks(webDocuments, 1000, 100)
  webChunks = removeUselessWebChunks(webChunks)
  addWebMetadataToChunks(webChunks)

  ## Step 1.4 Store on Vector DB - happens step 1.3 and 1.4 at the same time
  storeChunks(chunks)
  storeChunks(webChunks)
  # End Phase 1


############# PREDICT Phase
def getSimilarChunks(user_query):
  global vector_db
  # I've increased to 10 given some times the covers are mentioned in different pages and is relevant for final output.
  return vector_db.similarity_search(user_query, k=10) # devolve os X valores mais relevantes (valor default)

def getPrompt(user_query, relevant_chunks_text):
  return f"""
    User Query:
    {user_query}

    Context:
    {relevant_chunks_text}
  """

def invokeLLM(prompt):
  global messages

  # Step 2.5 - Call LLM
  llm = ChatOpenAI(
      model="gpt-4.1",
      temperature=0.1
  )

  # 3 roles - human, system, assistant
  messages.append(
      ("human", prompt)
  )

  llm_response = llm.invoke(messages)
  messages.append(
      ("assistant", llm_response.content)
  )

  return llm_response

def process_uploaded_file(file):
    if file is None:
        return "Por favor, carregue um ficheiro PDF."

    try:
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # In Gradio, the file object contains the path in the 'name' attribute
            with open(file.name, 'rb') as uploaded_file:
                tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Process the uploaded PDF
        # TODO try to load data with unstructured.io, given tables in PDF is hard to read
        documents = documentsFromPdfFile(tmp_path)
        cleanDocuments(documents)
        if not validDocument(documents):
            os.unlink(tmp_path)
            return "Por favor, carregue apenas ficheiros de condições particulares válidas."

        policy_number = extractPolicyNumberFromParticularConditions(documents)

        chunks = getChunks(documents, 1500, 100)
        addWebMetadataToParticularConditionsChunks(chunks, policy_number)
        chunks = removeUselessParticularConditionsChunks(chunks)
        storeChunks(chunks)

        # Clean up the temporary file
        os.unlink(tmp_path)

        return "Ficheiro processado com sucesso! Pode agora fazer perguntas sobre o conteúdo do documento."
    except Exception as e:
        # Clean up the temporary file if it exists
        if 'tmp_path' in locals():
            os.unlink(tmp_path)
        return f"Erro ao processar o ficheiro: {str(e)}"

def predict(message, history, file=None):
    # If a file was uploaded, process it first
    if file is not None:
        return process_uploaded_file(file)

    # Continue with the normal chat prediction if no file was uploaded
    user_query = message
    relevant_chunks = getSimilarChunks(user_query)

    relevant_chunks_text = ""
    for relevant_chunk in relevant_chunks:
        relevant_chunks_text += (
            "Content: " + relevant_chunk.page_content + "\n" +
            "Page: " + (relevant_chunk.metadata["page"] if "page" in relevant_chunk.metadata else "N/A") + "\n" +
            "Chapter: " + (relevant_chunk.metadata["chapter"] if "chapter" in relevant_chunk.metadata else "N/A") + "\n" +
            "Type: " + relevant_chunk.metadata["type"] + "\n" +
            "Policy number: " + (relevant_chunk.metadata["policy_number"] if "policy_number" in relevant_chunk.metadata else "N/A") + "\n"
        )

    prompt = getPrompt(user_query, relevant_chunks_text)
    llm_response = invokeLLM(prompt)
    return llm_response.content

ingestionStage()

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        value=[
            [None, "Olá! Sou o Zé Cautela, o seu assistente virtual especializado em seguros. Como posso ajudar?"]
        ],
        label="Assistente de Seguros Zé Cautela"
    )
    msg = gr.Textbox(label="Escreva a sua mensagem")
    upload_button = gr.UploadButton(
        "Carregar PDF",
        file_types=[".pdf", "application/pdf"],
        file_count="single"
    )

    # Separate handler for chat messages
    def chat_respond(message, chat_history):
        bot_message = predict(message, chat_history, None)
        chat_history.append([message, bot_message])
        return "", chat_history

    # Separate handler for file uploads
    def file_respond(file, chat_history):
        if file is None:
            return chat_history
        bot_message = process_uploaded_file(file)
        chat_history.append([f"Uploaded: {file.name}", bot_message])
        return chat_history

    # Connect the handlers to their respective events
    msg.submit(chat_respond, [msg, chatbot], [msg, chatbot])
    upload_button.upload(file_respond, [upload_button, chatbot], [chatbot])

demo.launch(debug=True)