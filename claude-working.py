import boto3
import json
import streamlit as st
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Initialize Bedrock Client
bedrock = boto3.client(service_name="bedrock-runtime")

# Using Titan Embeddings Model for generating embeddings
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Data ingestion function
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # Split the documents using Character Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Function to create the vector store using FAISS
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

# Function to get response from Claude model
def get_claude_response(prompt):
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.999,
        "messages": [{"role": "user", "content": prompt}]
    }

    body = json.dumps(payload)
    model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    
    try:
        response = bedrock.invoke_model_with_response_stream(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )
        print(f"test:{response}")
        response_text = ""
        for event in response['body']:
            event_payload = event['chunk']['bytes']
            print(f"event_payload: {event_payload}")
            chunk = json.loads(event_payload.decode('utf-8'))
            if 'delta' in chunk and 'text' in chunk['delta']:
                response_text += chunk['delta']['text']

        return response_text
    except Exception as e:
        st.error(f"Error invoking model: {e}")
        return None

# Main function for the Streamlit application
def main():
    st.set_page_config(page_title="Claude Chat")
    
    st.header("Chat with Claude")

    user_prompt = st.text_input("Enter your prompt")

    with st.sidebar:
        st.title("Update or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Get Response"):
        with st.spinner("Processing..."):
            response_text = get_claude_response(user_prompt)
            if response_text:
                st.write(response_text)
            else:
                st.write("No response received.")
            st.success("Done")

if __name__ == "__main__":
    main()
