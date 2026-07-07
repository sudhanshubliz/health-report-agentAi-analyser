import os
import tempfile
import uuid

import boto3
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def get_config(name, default=None):
    value = os.getenv(name)
    if value:
        return value

    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def normalize_r2_endpoint_url(endpoint_url, bucket_name=None):
    if not endpoint_url:
        return endpoint_url

    endpoint_url = endpoint_url.rstrip("/")
    if bucket_name:
        bucket_suffix = "/" + bucket_name.strip("/")
        if endpoint_url.endswith(bucket_suffix):
            endpoint_url = endpoint_url[: -len(bucket_suffix)]

    return endpoint_url


AWS_REGION = get_config("AWS_REGION", "us-east-1")
S3_REGION = get_config("AWS_S3_REGION", "eu-north-1")
BUCKET_NAME = get_config("R2_BUCKET_NAME") or get_config("BUCKET_NAME")
R2_ENDPOINT_URL = normalize_r2_endpoint_url(get_config("R2_ENDPOINT_URL"), BUCKET_NAME)
STORAGE_PROVIDER = "Cloudflare R2" if R2_ENDPOINT_URL else "Amazon S3"
HF_EMBEDDING_MODEL_NAME = get_config("HF_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


def get_boto3_client(service_name, region_name):
    client_kwargs = {"service_name": service_name, "region_name": region_name}
    aws_access_key_id = get_config("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = get_config("AWS_SECRET_ACCESS_KEY")

    if service_name == "s3" and R2_ENDPOINT_URL:
        client_kwargs["endpoint_url"] = R2_ENDPOINT_URL
        client_kwargs["region_name"] = get_config("R2_REGION", "auto")
        aws_access_key_id = get_config("R2_ACCESS_KEY_ID") or aws_access_key_id
        aws_secret_access_key = get_config("R2_SECRET_ACCESS_KEY") or aws_secret_access_key

    if aws_access_key_id and aws_secret_access_key:
        client_kwargs["aws_access_key_id"] = aws_access_key_id
        client_kwargs["aws_secret_access_key"] = aws_secret_access_key

    return boto3.client(**client_kwargs)


s3_client = get_boto3_client("s3", S3_REGION)


@st.cache_resource(show_spinner="Loading Hugging Face embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_unique_id():
    return str(uuid.uuid4())


## Split the pages / text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

## create vector store
def create_vector_store(request_id, documents):
    if not BUCKET_NAME:
        raise ValueError("BUCKET_NAME is not configured")

    vectorstore_faiss = FAISS.from_documents(documents, get_embeddings())
    file_name = f"{request_id}.bin"
    folder_path = tempfile.gettempdir()
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    ## upload to S3-compatible storage
    s3_client.upload_file(Filename=os.path.join(folder_path, file_name + ".faiss"), Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=os.path.join(folder_path, file_name + ".pkl"), Bucket=BUCKET_NAME, Key="my_faiss.pkl")

    return True

## main method
def main():
    with st.container():
        col1, col2 = st.columns([1, 5])
        with col1:
            st.image("https://img.icons8.com/color/48/medical-doctor.png", width=80)
        with col2:
            st.markdown("### Please upload the latest **blood report** for AI analysis")
            st.caption(f"⏳ Wait a moment while uploading to {STORAGE_PROVIDER}...")
    if not BUCKET_NAME:
        st.error("Storage bucket is not configured. Add R2_BUCKET_NAME or BUCKET_NAME to Streamlit secrets.")
        return

    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
    if uploaded_file is not None:
        request_id = get_unique_id()
        st.write(f"Request Id: {request_id}")
        saved_file_name = os.path.join(tempfile.gettempdir(), f"{request_id}.pdf")
        with open(saved_file_name, mode="wb") as w:
            w.write(uploaded_file.getvalue())

        try:
            loader = PyPDFLoader(saved_file_name)
            pages = loader.load_and_split()
        except Exception as exc:
            st.error(f"Could not read this PDF: {exc}")
            return

        st.write(f"Total Pages: {len(pages)}")

        ## Split Text
        splitted_docs = split_text(pages, 1000, 200)
        st.write(f"Splitted Docs length: {len(splitted_docs)}")

        if not splitted_docs:
            st.error("No readable text was found in this PDF.")
            return

        st.write("Creating the Vector Store")
        try:
            create_vector_store(request_id, splitted_docs)
            st.success("PDF processed successfully. You can ask questions below.")
        except Exception as exc:
            st.error(f"Error while creating the vector store: {exc}")



if __name__ == "__main__":
    main()
