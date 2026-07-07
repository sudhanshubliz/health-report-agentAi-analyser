import os
import tempfile

from botocore.exceptions import ClientError
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS

import admin


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

S3_REGION = admin.get_config("AWS_S3_REGION", admin.S3_REGION)
BUCKET_NAME = admin.BUCKET_NAME
STORAGE_PROVIDER = admin.STORAGE_PROVIDER
HF_API_TOKEN = admin.get_config("HUGGINGFACEHUB_API_TOKEN") or admin.get_config("HF_TOKEN")
HF_LLM_REPO_ID = admin.get_config("HF_LLM_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.3")
INDEX_NAME = "my_faiss"
FOLDER_PATH = tempfile.gettempdir()

s3_client = admin.get_boto3_client("s3", S3_REGION)


def load_index():
    if not BUCKET_NAME:
        raise ValueError("BUCKET_NAME is not configured")

    faiss_path = os.path.join(FOLDER_PATH, f"{INDEX_NAME}.faiss")
    pkl_path = os.path.join(FOLDER_PATH, f"{INDEX_NAME}.pkl")
    s3_client.download_file(Bucket=BUCKET_NAME, Key=f"{INDEX_NAME}.faiss", Filename=faiss_path)
    s3_client.download_file(Bucket=BUCKET_NAME, Key=f"{INDEX_NAME}.pkl", Filename=pkl_path)


def get_llm():
    if not HF_API_TOKEN:
        raise ValueError("Hugging Face token is not configured")

    return HuggingFaceHub(
        repo_id=HF_LLM_REPO_ID,
        huggingfacehub_api_token=HF_API_TOKEN,
        model_kwargs={
            "temperature": 0.1,
            "max_new_tokens": 512,
            "return_full_text": False,
        },
    )


def get_response(llm, vectorstore, question):
    prompt_template = """

    Human: Please use the given context to provide a concise answer to the question.
    If you don't know the answer, just say that you don't know. Do not make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    answer = qa.invoke({"query": question})
    return answer["result"]


def render_question_box(faiss_index):
    st.markdown(
        """
        <div style="padding:15px; border-radius:12px; background-color:#f8f9fa; border:1px solid #ddd;">
            <h4 style="margin:0;">💬 Ask Your Question</h4>
            <p style="color:#555; margin-bottom:10px;">Type your query below and let AI assist you.</p>
            <h4 style="margin:0;">💬 Examples</h4>
            <p style="color:#555; margin-bottom:10px;">What health parameters are analyzed in this report?</p>
            <p style="color:#555; margin-bottom:10px;">What are the key findings from the latest health tests?</p>
            <p style="color:#555; margin-bottom:10px;">How do the current test results compare to previous test results?</p>
            <p style="color:#555; margin-bottom:10px;">Are there any health concerns or areas that need attention?</p>
            <p style="color:#555; margin-bottom:10px;">What recommendations are provided for physical activity and diet?</p>
            <p style="color:#555; margin-bottom:10px;">Can you explain the stress management suggestions?</p>
            <p style="color:#555; margin-bottom:10px;">What is the significance of results like TSH, Iron, and Cholesterol?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    question = st.text_input("Question", placeholder="Ask about your uploaded report")
    if st.button("🚀 Ask Question"):
        if not question.strip():
            st.warning("Please enter a question first.")
            return

        with st.spinner("🤖 Querying AI... please wait"):
            try:
                llm = get_llm()
                st.write(get_response(llm, faiss_index, question.strip()))
                st.success("Answer received ✅")
            except ValueError as exc:
                st.error(f"{exc}. Add HUGGINGFACEHUB_API_TOKEN or HF_TOKEN to Streamlit secrets.")
            except Exception as exc:
                st.error(f"Error while querying Hugging Face: {exc}")


def main():
    st.markdown(
        """
        <div style="padding:20px; border-radius:15px; background-color:#f0f9ff;
                    border:2px solid #cce5ff; text-align:center;">
            <h2 style="color:#004085; margin:0;">
                🩸 Blood Report Analysis & Lifestyle Guidance Agent AI ⛑️
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    admin.main()

    if not BUCKET_NAME:
        return

    try:
        load_index()
        faiss_index = FAISS.load_local(
            index_name=INDEX_NAME,
            folder_path=FOLDER_PATH,
            embeddings=admin.get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "Unknown")
        if error_code in {"403", "AccessDenied", "Forbidden"}:
            st.error(
                f"{STORAGE_PROVIDER} denied access to the FAISS index. "
                "Check bucket name, endpoint URL, access keys, and permissions."
            )
        elif error_code in {"404", "NoSuchKey", "NotFound"}:
            st.info("Upload a PDF report first, then the question box will appear.")
        else:
            st.error(f"Could not load the FAISS index from {STORAGE_PROVIDER}: {error_code}")
        return
    except Exception as exc:
        st.info("Upload a PDF report first, then the question box will appear.")
        st.caption(f"Index status: {exc}")
        return

    st.success("Index is ready")
    render_question_box(faiss_index)


if __name__ == "__main__":
    main()
