import gradio as gr
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
import re
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Now we can access the API key from .env
embeddings_model = OpenAIEmbeddings(
    #model = "text-embedding-3-small"
    model = "text-embedding-3-large"
)

vector_db = Chroma(
    collection_name="multiriscos",
    embedding_function=embeddings_model,
    persist_directory="db_seguros",  # Where to save data locally, remove if not necessary
)

messages = []

FAQ_URL = "https://www.generalitranquilidade.pt/perguntas-frequentes"


# system_prompt = f"""
#     You are a professional assistant in the insurance industry and know a lot about insurances.

#     Instructions:
#     Answer the provided user query based on the provided Context.
#     If the answer for the question is not on the provided Context, check if it's on the Chat History. If it is, answer based on that.
#     If after checking the context and chat history you still don't know the answer,
#     don't use internal knowledge, answer with 'I don't have the provided context for that question'.
#     When the answer is related to clauses, mention the chapter and clause number as reference, in a separate paragraph.
#     Reply in portuguese from Portugal.
# """
system_prompt = """
    És um assistente de seguros, que se chama Zé Cautela, que responde a perguntas sobre cobertura dos seguros que tens no teu contexto.

    Instruções:
    Response à pergunta do utilizador (user query) baseado no contexto fornecido.
    Se a resposta para a pergunta não estiver no contexto, verifica se está na história da conversa. Se estiver, responde baseado nessa informação.
    Se após verificar o contexto e a história da conversa ainda não conseguires responder, não uses o teu conhecimento interno, responde com 'Não tenho o contexto necessário para essa pergunta'.
    Se as tuas respostas forem do contexto, adiciona o número do capítulo, da cláusula e das páginas (caso estes existam) que usaste como referência para a tua resposta, num parágrafo separado.
    Responde sempre em português de Portugal.
"""

messages.append(
    ("system", system_prompt)
)

def documentsFromPdfFile(file_path):
  # e.g. can use a CSVLoader
  loader = PyPDFLoader(file_path, mode="single") # we can use different loaders, eg by folder, by url, etc
  documents = loader.load() #langchain já faz por vezes algum chunking e mete metadata
  return documents

def documentsFromWebSite(url):
  loader = WebBaseLoader(url)
  documents = loader.load()
  return documents

def cleanDocuments(documents):
  for document in documents:
    document.page_content = re.sub('\n', ' ', document.page_content)

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
      chunk_size=chunk_size, #começar com 500 ou 1000 e ir mexendo de 100 em 100
      chunk_overlap=chunk_overlap #0.15 - 0.3 do valor do size só para dados não estruturados
  )
  chunks = text_splitter.split_documents(documents)

  # TIP - Por norma, dados estruturados não precisam de chunk size (e.g. CSV é lido por linha)

  #text_splitter = CharacterTextSplitter(
  #    separator=".",
  #    chunk_size=1000,
  #    chunk_overlap=100
  #)

  return chunks

def addMetadataToChunks(chunks):
    currentChapter = ""
    currentClause = ""
    currentPage = "1"
    for chunk in chunks:
        # Extract page number using regex
        page_match = re.search(r'(\d+)\s*Mod\.\s*465\.031\s*–\s*janeiro\s*2025', chunk.page_content)
        if page_match:
            currentPage = page_match.group(1)  # group(1) gets the first captured group (the number)

        if "CAPÍTULO" in chunk.page_content:
            currentChapter = chunk.page_content.split("CAPÍTULO")[1].split(" ")[1].strip()

        # print("Current Page: ", currentPage)
        # Add both chapter and page to metadata
        chunk.metadata = {
            "chapter": currentChapter,
            "page": currentPage,
            "type": "pdf"
        }
def addWebMetadataToChunks(chunks):
    for chunk in chunks:
        chunk.metadata = {
            "type": "web",
            "page": FAQ_URL
        }

#Store the chunks in the vector db
def storeChunks(chunks):
  global vector_db
  ids = vector_db.add_documents(chunks)
  return ids

def ingestionStage(): # 1 - x (sempre que quisermos colocar dados novos na nossa vectordb)
  # Skip ingestion stage if we already have the data in the vector db
  global vector_db
  if len(vector_db.get()['documents']) > 0:
    print("Data already in vector db, skipping ingestion stage")
    return

  # Phase 1 load (Pre-RAG)

  # Importante -> escolher o melhor loader baseado no formato dos dados
  ### Step 1.1 Document Load
  pdfDocuments = documentsFromPdfFile("seguro.pdf")
  webDocuments = documentsFromWebSite(FAQ_URL)
  ### Step 1.2 Document Transformation

  # Step intermédio de limpeza PDF
  cleanDocuments(pdfDocuments)
  chunks = getChunks(pdfDocuments, 1100, 100)
  addMetadataToChunks(chunks)

  # Step intermédio de limpeza Web
  cleanDocuments(webDocuments)
  webChunks = getChunks(webDocuments, 1000, 100)
  webChunks = removeUselessWebChunks(webChunks)
  addWebMetadataToChunks(webChunks)
  # print("Chunk 0: ", webChunks[0].page_content)
  # print("Chunk 1: ", webChunks[1].page_content)
  # print("Chunk 2: ", webChunks[2].page_content)
  # print("Chunk 3: ", webChunks[3].page_content)
  # print("Chunk 4: ", webChunks[4].page_content)
  # print("Chunk 5: ", webChunks[5].page_content)
  # print("Chunk 6: ", webChunks[6].page_content)
  # print("Chunk 7: ", webChunks[7].page_content)
  # print("Chunk 8: ", webChunks[8].page_content)
  # print("Chunk 9: ", webChunks[9].page_content)
  # print("Chunk 10: ", webChunks[10].page_content)

#   print("Chunk 0: ", chunks[0].page_content)
#   print("Chunk 1: ", chunks[1].page_content)
#   print("Chunk 2: ", chunks[2].page_content)
#   print("Chunk 3: ", chunks[3].page_content)
#   print("Chunk 4: ", chunks[4].page_content)
#   print("Chunk 5: ", chunks[5].page_content)
#   print("Chunk 6: ", chunks[6].page_content)
#   print("Chunk 7: ", chunks[7].page_content)
#   print("Chunk 8: ", chunks[8].page_content)
#   print("Chunk 9: ", chunks[9].page_content)
#   print("Chunk 10: ", chunks[10].page_content)


  ## Step 1.4 Store on Vector DB - acontece o step 1.3 e 1.4 ao mesmo tempo
  storeChunks(chunks)
  storeChunks(webChunks)
  #print(ids)
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
      # model="gpt-4.1-nano"
      #model="gpt-4.1-mini"
      model="gpt-4.1",
      temperature=0.1
  )

  # 3 roles - human, system, assistant
  # system - usado para instruções, cenas globais para a conversa
  # posso usar o tokenizer para saber número de tokens que estão a ser enviados
  messages.append(
      ("human", prompt)
  )

  llm_response = llm.invoke(messages)
  messages.append(
      ("assistant", llm_response.content)
  )

  #print(messages)

  return llm_response

def predict(message, history): # y  (sempre que existe uma pergunta de um user no frontend)

  # Phase 2 - RAG - Generation
  # Step 2.1 - Embed user query or message
  user_query = message

  # Step 2.2 - Embedding of user query (demonstração) # mesma coleção usa-se SEMPRE o mesmo embedding_model
  # embeddings_user_query = embeddings_model.embed_query(user_query)
  # When we do this. we can do similarity_Search_by_vector
  ## iriamos ter 1 vector com o significado semantico da user query


  #Step 2.3. - Similarity search
  # This step has costs, it converts stuff using embedding model
  relevant_chunks = getSimilarChunks(user_query) # devolve os 4 valores mais relevantes (valor default)
  # poderia adicionar metadata aos chunks e depois na search filtrar por metadata, se quiser que alguns contextos sejam específicos de users.

  #print(len(relevant_chunks))
  #print(relevant_chunks)

  relevant_chunks_text = ""
  for relevant_chunk in relevant_chunks:
    # Há quem adicionei ids das chunks no prompt
    relevant_chunks_text += (
        "Content: " + relevant_chunk.page_content + "\n" +
        "Page: " + (relevant_chunk.metadata["page"] if "page" in relevant_chunk.metadata else "N/A") + "\n" +
        "Chapter: " + (relevant_chunk.metadata["chapter"] if "chapter" in relevant_chunk.metadata else "N/A") + "\n" +
        "Type: " + relevant_chunk.metadata["type"] + "\n"
    )

  #print("Relevant Chunks Text: ", relevant_chunks_text)

  # Step 2.4 - Create prompt
  prompt = getPrompt(user_query, relevant_chunks_text)

  llm_response = invokeLLM(prompt)

  return llm_response.content

ingestionStage()
#print(predict("O seguro cobre danos de explosão? Se sim, qual a cobertura?", ""))
#print(predict("Como posso reportar um sinistro?", ""))

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        value=[
            [None, "Olá! Sou o Zé Cautela, o seu assistente virtual especializado em seguros. Como posso ajudar?"]
        ],
        label="Assistente de Seguros Zé Cautela"
    )
    msg = gr.Textbox(label="Escreva a sua mensagem")

    def respond(message, chat_history):
        bot_message = predict(message, chat_history)
        chat_history.append([message, bot_message])
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch(debug=True)