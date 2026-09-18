import streamlit as st
import pandas as pd
import json
from predict import corriger_workflow_n8n

def reconstruire_workflow(correction, original_workflow_path):
    with open(original_workflow_path) as f:
        workflow = json.load(f)
    for i, node in enumerate(workflow["nodes"]):
        if i >= len(correction):
            break
        node_vec = correction[i]
        node["type"] = bool(node_vec[0])
        node["has_credentials"] = bool(node_vec[1])
        node["disabled"] = bool(node_vec[2])
        node["outgoing"] = bool(node_vec[3])
        node["incoming"] = bool(node_vec[4])
        node["is_broken"] = bool(node_vec[5])
        

    return workflow

st.title("AI Agent via DRL to correct n8n workflows, upload, wait then load your corrected workflow.")

uploaded_file = st.file_uploader("Your n8n workflow (.json) goes here", type="json")
columns = ["type", "has_credentials", "disabled", "outgoing", "incoming", "is_broken"]

if uploaded_file is not None:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if st.button("Correct me 🎀 "):
        correction = corriger_workflow_n8n(tmp_path)

        # 1. Affichage tableau des features
        df = pd.DataFrame(correction, columns=columns)
        st.subheader("Final state (features)")
        st.dataframe(df)

        # 2. Workflow JSON corrigé
        corrected_workflow = reconstruire_workflow(correction, tmp_path)
        st.subheader("Corrected JSON workflow")
        st.json(corrected_workflow)
