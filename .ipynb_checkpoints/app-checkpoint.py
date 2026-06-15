import streamlit as st
import pandas as pd
import plotly.express as px
from core.parser import extract_clean_text
from core.compliance_engine import LocalComplianceEngine
from openai import OpenAI

st.set_page_config(layout="wide", page_title="AMD Dual-Asset Comparison Shield")

ACTIVE_MODEL = "Qwen/Qwen2.5-14B-Instruct"

if "comparison_ledger" not in st.session_state:
    st.session_state.comparison_ledger = None
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

COMPLIANCE_TARGETS = [
    {"id": "REG-A10", "name": "Premium Grace Validation", "text": "Insurance policies must offer a minimum of a 30-day grace period for premium payments."},
    {"id": "REG-B22", "name": "Underwriting Disclosure Protocols", "text": "Medical history exclusion factors must be explicitly outlined with clear explanations."},
    {"id": "REG-C35", "name": "Dispute Mediation Window", "text": "Dispute resolution clauses must specify a mandatory 90-day mediation timeline before formal litigation."}
]

st.title("🛡️ AMD Instinct™ Co-Pilot: Dual-Document Comparison Node")
st.markdown("Side-by-side automated variance check and data alignment verification pipeline powered by **ROCm + local vLLM**.")
st.write("---")

# Dual File Uploaders inside Sidebar
st.sidebar.header("📥 Document Intake Split-Framework")
file_a = st.sidebar.file_uploader("Upload Document A (Baseline/Reference PDF)", type=["pdf"])
file_b = st.sidebar.file_uploader("Upload Document B (Incoming Target/Draft PDF)", type=["pdf"])

raw_doc_a = ""
raw_doc_b = ""

if file_a and file_b:
    raw_doc_a = extract_clean_text(file_a)
    raw_doc_b = extract_clean_text(file_b)
    st.sidebar.success("⚡ Both document buffers successfully mounted!")
    
    if st.sidebar.button("⚙️ Launch Comparative Variance Analysis"):
        with st.spinner("Processing structural text matrix alignments over local vLLM..."):
            engine = LocalComplianceEngine(model_id=ACTIVE_MODEL)
            computed_runs = []
            for target in COMPLIANCE_TARGETS:
                run_metrics = engine.compare_documents(raw_doc_a, raw_doc_b, target)
                run_metrics["rule_id"] = target["id"]
                run_metrics["rule_name"] = target["name"]
                computed_runs.append(run_metrics)
            st.session_state.comparison_ledger = computed_runs
else:
    st.sidebar.info("Awaiting upload of both Reference and Target files to unlock execution pipeline.")

# Navigation Tabs
analytics_tab, lineage_tab, assistant_tab = st.tabs(["📊 Performance Overview", "📋 Verified Variance Ledger", "💬 Deep Interrogation Agent"])

# Tab 1: Delta Performance Graphs
with analytics_tab:
    if st.session_state.comparison_ledger is None:
        st.info("💡 Complete a comparative validation run to populate cross-document analytics graphs.")
    else:
        df = pd.DataFrame(st.session_state.comparison_ledger)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rules Evaluated", len(df))
        c2.metric("Aligned Items (MATCH)", len(df[df['status'] == 'MATCH']))
        c3.metric("Conflicting Variances (MISMATCH/GAP)", len(df[df['status'] != 'MATCH']), delta_color="inverse")
        
        st.write("---")
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(px.pie(df, names='status', color='status', 
                             color_discrete_map={'MATCH':'#2ca02c','MISMATCH':'#d62728','GAP':'#ff7f0e'}), use_container_width=True)
        with ch2:
            st.plotly_chart(px.bar(df, x='rule_id', y='confidence', color='status',
                             color_discrete_map={'MATCH':'#2ca02c','MISMATCH':'#d62728','GAP':'#ff7f0e'}), use_container_width=True)
# Tab 2: Detailed Side-by-Side Audit Trail
with lineage_tab:
    if st.session_state.comparison_ledger is not None:
        st.subheader("Side-by-Side Structural Audit Trail")
        
        for entry in st.session_state.comparison_ledger:
            # 1. Map status parameters to native UI elements
            status = entry['status']
            if status == "MATCH":
                title_prefix = "🟢 [MATCH]"
            elif status == "MISMATCH":
                title_prefix = "🔴 [MISMATCH]"
            else:
                title_prefix = "🟡 [GAP]"
                
            # 2. Render an isolated card block container for each rule
            with st.container(border=True):
                st.markdown(f"### {title_prefix} **{entry['rule_id']}: {entry['rule_name']}**")
                st.markdown(f"**Auditor Rationale:** {entry['explanation']}")
                st.write("---")
                
                # 3. Use native structural columns for side-by-side view
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("🔹 **DOCUMENT A EXCERPT (Baseline)**")
                    # Using code blocks gives a clean monospace look without HTML breaking it
                    st.info(f"\"{entry['doc_a_evidence']}\"")
                    
                with col_b:
                    st.markdown("🔸 **DOCUMENT B EXCERPT (Target)**")
                    st.info(f"\"{entry['doc_b_evidence']}\"")
                    
            st.write("") # Margin spacing between rule containers
            
    else:
        st.info("💡 Awaiting document comparison processing to map cross-file lineage details.")
# Tab 3: Interactive Chat Matrix
with assistant_tab:
    if not raw_doc_a or not raw_doc_b:
        st.warning("⚠️ Chat node locked. Mount both documents to populate dual-context conversation memory.")
    else:
        for msg in st.session_state.chat_memory:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
            
        if query := st.chat_input("Ask a question regarding variations, discrepancies, or formatting between Document A and B..."):
            with st.chat_message("user"): st.markdown(query)
            st.session_state.chat_memory.append({"role": "user", "content": query})
            
            vllm_client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
            
            with st.chat_message("assistant"):
                combined_prompt = f"Document A Context:\n{raw_doc_a[:4000]}\n\nDocument B Context:\n{raw_doc_b[:4000]}\n\nQuestion: {query}"
                stream = vllm_client.chat.completions.create(
                    model=ACTIVE_MODEL,
                    messages=[{"role": "user", "content": combined_prompt}],
                    stream=True
                )
                output_box = st.empty()
                collected_text = ""
                for segment in stream:
                    if segment.choices[0].delta.content:
                        collected_text += segment.choices[0].delta.content
                        output_box.markdown(collected_text + "▌")
                output_box.markdown(collected_text)
            st.session_state.chat_memory.append({"role": "assistant", "content": collected_text})