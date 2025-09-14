import boto3
import streamlit as st
import uuid
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


## s3_client
s3_client = boto3.client("s3")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
print(f"BUCKET_NAME is: {BUCKET_NAME}")
if BUCKET_NAME is None:
    raise ValueError("BUCKET_NAME environment variable not set")

## Bedrock
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat

## prompt and chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Pdf Loader
from langchain_community.document_loaders import PyPDFLoader

## import FAISS
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings

session = boto3.Session(
    aws_access_key_id="AKIAWMN3TUK55UMGDIOU",
    aws_secret_access_key="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",
    region_name="us-east-1"
)

bedrock_client = session.client("bedrock-runtime")

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",  # <-- change this
    client=bedrock_client
)
folder_path="/tmp/"

def get_unique_id():
    return str(uuid.uuid4())

## load index
def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")

def get_llm():
    llm=BedrockChat(model_id="arn:aws:bedrock:us-east-1:439016989371:inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0", client=bedrock_client,
                    provider="anthropic", model_kwargs={"max_tokens": 512})
    return llm

# get_response()
def get_response(llm,vectorstore, question ):
    ## create prompt / template
    prompt_template = """

    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer = qa.invoke({"query": question})
    return answer['result']


## main method
def main():
    st.header(""" Blood Report 🩸Analysis & Lifestyle Guidance Agent AI ⛑️
              Client Site for Chat with sample PDF uploaded by admin in s3 bucket. 
              Technology used Amazon Bedrock,S3 Bucket,Anthropic Claude 3.5 Haiku LLM, RAG,
              streamlit, pypdf, langchain-aws, faiss-cpu, boto3 etc.""")

    load_index()

    dir_list = os.listdir(folder_path)
   # st.write(f"Files and Directories in {folder_path}")
    #st.write(dir_list)

    ## create index
    faiss_index = FAISS.load_local(
        index_name="my_faiss",
        folder_path = folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    st.write("INDEX IS READY")
    question = st.text_input("Please ask your question")
    if st.button("Ask Question"):
        with st.spinner("Querying..."):

            llm = get_llm()

            # get_response
            st.write(get_response(llm, faiss_index, question))
            st.success("Done")

if __name__ == "__main__":
    main()