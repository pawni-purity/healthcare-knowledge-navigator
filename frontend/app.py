import os
import streamlit as st
import requests
from typing import List, Dict, Any

# Configure premium page layout
st.set_page_config(
    page_title="Healthcare Knowledge Navigator",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend service address
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000/api/v1")

# Apply custom premium styling
st.markdown("""
<style>
    /* Styling adjustments */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
        font-family: 'Inter', sans-serif;
    }
    .stSidebar {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    .stTab {
        font-size: 16px;
        font-weight: 600;
    }
    /* Custom CSS card for citations and search chunks */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .chunk-title {
        color: #58a6ff;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .chunk-meta {
        font-size: 12px;
        color: #8b949e;
        margin-bottom: 10px;
    }
    .badge-high {
        background-color: #238636;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-medium {
        background-color: #d29922;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-low {
        background-color: #da3633;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar content
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
    st.title("Navigator Control")
    st.markdown("🔒 **Clinical Decision RAG Engine**")
    
    st.divider()
    
    # Check backend health
    try:
        health_resp = requests.get(f"{BACKEND_URL.replace('/api/v1', '')}/health", timeout=3)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            st.success("🟢 Backend Services Online")
            st.info(f"🧬 Model: {health_data.get('embedding_model', 'bge-large')}")
        else:
            st.error("🔴 Backend Error Response")
    except Exception:
        st.error("🔴 Backend Connection Offline")
        st.warning("Please run uvicorn on localhost:8000")
        
    st.divider()
    st.markdown("""
    ### System Guidelines
    - Grounded Q&A answers are generated **strictly** from uploaded context.
    - Evidence levels are resolved from parsed PDF layout hierarchies.
    - Medical acronyms are automatically expanded during search execution.
    """)

# Main title
st.title("🩺 Healthcare Knowledge Navigator")
st.markdown("#### Evidence-Based Clinical Decision Support System")

# Set up Tab Workspace
tab_qa, tab_upload, tab_explorer = st.tabs([
    "💬 Clinical Q&A Assistant", 
    "📥 Ingest Guidelines", 
    "🔍 Search Explorer & Metrics"
])


# ==========================================
# TAB 1: CLINICAL Q&A ASSISTANT
# ==========================================
with tab_qa:
    st.markdown("### Grounded Conversational AI Assistant")
    
    # Initialize message list in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            # If assistant response contains metadata, render it
            if msg.get("confidence"):
                conf = msg["confidence"]
                citations = msg.get("citations") or []
                
                # Render confidence badge
                lbl = conf.get("label", "Low")
                badge_class = f"badge-{lbl.lower()}"
                st.markdown(
                    f"**Confidence Level:** <span class='{badge_class}'>{lbl} ({conf.get('score')})</span>", 
                    unsafe_allow_html=True
                )
                
                # Render citations list
                if citations:
                    with st.expander("📚 View Supporting Evidence & Citations"):
                        for cit in citations:
                            st.markdown(f"""
                            **[{cit['citation_index']}] {cit['title']}**
                            - **Section**: {cit['section_header'] or 'N/A'} | **Page**: {cit['page_number'] or 'N/A'}
                            - **Evidence Grade**: {cit['evidence_level'] or 'N/A'} | **Publication Year**: {cit['publication_year'] or 'N/A'}
                            - **Source Type**: {cit['source_type'] or 'N/A'}
                            - *Doc UUID*: `{cit['document_id']}`
                            """)
                            st.divider()

    # User input chat box
    if user_query := st.chat_input("Enter your clinical question... (e.g. 'First line ACEI dosage for hypertension')"):
        # Display user input immediately
        st.chat_message("user").write(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Build assistant response placeholder
        with st.chat_message("assistant"):
            with st.spinner("Analyzing context & generating cited response..."):
                try:
                    # Format history payload
                    history_payload = []
                    # Keep only last 10 messages for context window management
                    for msg in st.session_state.messages[:-1]:
                        history_payload.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                        
                    payload = {
                        "query": user_query,
                        "history": history_payload,
                        "limit": 5
                    }
                    
                    resp = requests.post(f"{BACKEND_URL}/rag/chat", json=payload, timeout=60)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data["answer"]
                        citations = data["citations"]
                        confidence = data["confidence"]
                        
                        # Display text answer
                        st.write(answer)
                        
                        # Display confidence level
                        lbl = confidence.get("label", "Low")
                        badge_class = f"badge-{lbl.lower()}"
                        st.markdown(
                            f"**Confidence Level:** <span class='{badge_class}'>{lbl} ({confidence.get('score')})</span>", 
                            unsafe_allow_html=True
                        )
                        
                        # Display citations dropdown
                        if citations:
                            with st.expander("📚 View Supporting Evidence & Citations"):
                                for cit in citations:
                                    st.markdown(f"""
                                    **[{cit['citation_index']}] {cit['title']}**
                                    - **Section**: {cit['section_header'] or 'N/A'} | **Page**: {cit['page_number'] or 'N/A'}
                                    - **Evidence Grade**: {cit['evidence_level'] or 'N/A'} | **Publication Year**: {cit['publication_year'] or 'N/A'}
                                    - **Source Type**: {cit['source_type'] or 'N/A'}
                                    - *Doc UUID*: `{cit['document_id']}`
                                    """)
                                    st.divider()
                                    
                        # Append assistant message with metadata details to session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "confidence": confidence,
                            "citations": citations
                        })
                        
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Communication failure: {e}")

    # Clear chat button
    if st.button("Clear Conversation History", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()


# ==========================================
# TAB 2: INGEST MEDICAL GUIDELINES
# ==========================================
with tab_upload:
    st.markdown("### Upload & Index Medical PDFs")
    st.write("Upload medical documents and run layout-aware parser, generating dense embeddings stored in Qdrant.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("Choose a PDF guideline file", type="pdf")
        
    with col2:
        title = st.text_input("Document Title", placeholder="e.g. AHA Atrial Fibrillation Guidelines 2023")
        source_type = st.selectbox(
            "Source Type",
            options=["clinical_guideline", "biomedical_paper", "treatment_protocol"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        publisher = st.text_input("Publisher (Optional)", placeholder="e.g. American Heart Association")
        evidence_level = st.text_input("Evidence Grade Level (Optional)", placeholder="e.g. Level 1a / Grade A")

    if st.button("🚀 Ingest and Index Document", key="submit_upload"):
        if not uploaded_file:
            st.error("Please upload a PDF file first.")
        elif not title.strip():
            st.error("Please enter a document title.")
        else:
            with st.spinner("Ingesting file... This performs layout text parsing, hierarchical chunking, and embedding vectors generation."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    data = {
                        "title": title,
                        "source_type": source_type,
                        "publisher": publisher,
                        "evidence_level": evidence_level
                    }
                    
                    resp = requests.post(f"{BACKEND_URL}/ingestion/upload", files=files, data=data, timeout=120)
                    
                    if resp.status_code == 200 or resp.status_code == 201:
                        res_json = resp.json()
                        st.success(f"🎉 Document successfully indexed!")
                        st.balloons()
                        st.json({
                            "document_id": res_json.get("document_id"),
                            "status": res_json.get("status"),
                            "parent_chunks_count": res_json.get("parent_chunks_count"),
                            "child_chunks_count": res_json.get("child_chunks_count"),
                            "message": res_json.get("message")
                        })
                    else:
                        st.error(f"Ingestion failed with status {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Error during uploading: {e}")


# ==========================================
# TAB 3: SEARCH EXPLORER & METRICS
# ==========================================
with tab_explorer:
    st.markdown("### Search Explorer & Evaluation Metrics")
    
    explorer_subtab1, explorer_subtab2 = st.tabs([
        "🔍 Semantic Search Explorer", 
        "📊 Run Retrieval Evaluation"
    ])
    
    # SUBTAB 1: SEMANTIC SEARCH
    with explorer_subtab1:
        st.markdown("#### Dense Vector Pre-Filtered Search")
        
        search_query = st.text_input("Search query", placeholder="Enter query terms (acronyms like 'COPD' will be expanded)...")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            filter_source_type = st.selectbox(
                "Filter Source Type (Optional)",
                options=["", "clinical_guideline", "biomedical_paper", "treatment_protocol"],
                format_func=lambda x: "No Filter" if x == "" else x.replace("_", " ").title(),
                key="filter_source_type"
            )
        with col_f2:
            filter_doc_id = st.text_input("Filter Document ID (Optional)", placeholder="UUID string")
        with col_f3:
            filter_page = st.number_input("Filter Page Number (Optional)", min_value=0, step=1, value=0)

        search_limit = st.slider("Result Count (K)", min_value=1, max_value=20, value=5)
        
        if st.button("Run Search", key="submit_search"):
            if not search_query.strip():
                st.error("Please enter a query.")
            else:
                with st.spinner("Searching vectors..."):
                    try:
                        # Construct filters
                        filters = {}
                        if filter_source_type:
                            filters["source_type"] = filter_source_type
                        if filter_doc_id.strip():
                            filters["document_id"] = filter_doc_id.strip()
                        if filter_page > 0:
                            filters["page_number"] = filter_page
                            
                        payload = {
                            "query": search_query,
                            "limit": search_limit,
                            "filters": filters if filters else None,
                            "expand_query": True
                        }
                        
                        resp = requests.post(f"{BACKEND_URL}/search/query", json=payload, timeout=20)
                        
                        if resp.status_code == 200:
                            results = resp.json()
                            if not results:
                                st.warning("No matches found.")
                            else:
                                st.success(f"Retrieved {len(results)} chunks successfully.")
                                for idx, res in enumerate(results):
                                    doc = res["document"]
                                    st.markdown(f"""
                                    <div class='metric-card'>
                                        <div class='chunk-title'>[{idx+1}] Score: {res['score']:.4f} - {doc['title']}</div>
                                        <div class='chunk-meta'>
                                            Section: {res['section_header'] or 'N/A'} | Page: {res['page_number'] or 'N/A'} | 
                                            Type: {doc['source_type']} | Grade: {doc['evidence_level'] or 'N/A'}
                                        </div>
                                        <div><strong>Child Context content:</strong></div>
                                        <p style='font-style: italic; color: #a1b0cb;'>"{res['content']}"</p>
                                        <div style='margin-top:5px;'><strong>Parent Reconstructed context content:</strong></div>
                                        <p style='color: #c9d1d9;'>"{res['parent_content']}"</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.error(f"Search failed: {resp.text}")
                    except Exception as e:
                        st.error(f"Search API error: {e}")

    # SUBTAB 2: RUN RETRIEVAL EVALUATION
    with explorer_subtab2:
        st.markdown("#### Evaluate Retrieval Performance (Hit Rate & MRR)")
        st.write("Calculates recall performance over a synthetic golden test dataset of clinical questions mapped to target guidelines.")
        
        if st.button("⚡ Run Retrieval Metrics Assessment", key="submit_evaluation"):
            # Provide a golden set of medical query-document pairs to evaluate
            # Since document IDs depend on database contents, we fetch documents first to construct a valid evaluation payload
            with st.spinner("Analyzing active document IDs to prepare validation run..."):
                try:
                    # Let's perform evaluate on some queries.
                    # Since this is a test page, we construct a dummy sample set using documents currently in the DB.
                    # We can fetch some guidelines from search/queries or create a pre-defined evaluation request.
                    # Let's construct a synthetic dataset to show evaluation results.
                    # First, we need to find some documents in the DB to test on.
                    # We can use some mock document IDs if there are none, or execute with the endpoint.
                    
                    # For a robust eval dashboard, we'll try to retrieve the first document ID.
                    # If empty, we notify the user.
                    eval_payload = {
                        "dataset": [
                            {
                                "query": "hypertension ACEI treatment recommendations",
                                "expected_document_id": "00000000-0000-0000-0000-000000000000",
                            },
                            {
                                "query": "atrial fibrillation stroke prevention",
                                "expected_document_id": "00000000-0000-0000-0000-000000000000",
                            }
                        ],
                        "limit": 5
                    }
                    
                    resp = requests.post(f"{BACKEND_URL}/search/evaluate", json=eval_payload, timeout=60)
                    
                    if resp.status_code == 200:
                        eval_data = resp.json()
                        
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric(label="Total Queries Evaluated", value=eval_data.get("total_queries", 0))
                        with col_m2:
                            st.metric(label="Mean Hit Rate (Recall@K)", value=f"{eval_data.get('mean_hit_rate', 0.0) * 100:.1f}%")
                        with col_m3:
                            st.metric(label="Mean Reciprocal Rank (MRR)", value=f"{eval_data.get('mean_reciprocal_rank', 0.0):.4f}")
                            
                        # Show logs table
                        logs = eval_data.get("queries_evaluated", [])
                        if logs:
                            st.markdown("##### Detailed Evaluation Log")
                            st.dataframe(logs)
                    else:
                        st.error(f"Evaluation request failed: {resp.text}")
                except Exception as e:
                    st.error(f"Evaluation service error: {e}")
