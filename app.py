"""
app.py - Main entry (Home / Landing Page)
Provides an overview of the app and guides users to start the workflow.
"""

import streamlit as st
from utils import inject_css, set_session_defaults

st.set_page_config(
    page_title="Excel Transformer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_css()
set_session_defaults({
    "uploaded_df": None,
    "transformations": {},
    "preview_df": None
})

st.markdown("<div class='header'>Excel Data Structuring Tool</div>", unsafe_allow_html=True)
st.markdown("<div class='lead'>Premium minimal UI — Upload → Transform → Preview → Output</div>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown(
        """
        **What this app does (UI only)**  
        - Upload an Excel file (.xlsx) and view a quick preview.  
        - Collect transformation logic via intuitive UI forms.  
        - Show a **safe preview** applying non-destructive transformations (frontend-only).  
        - Provide downloadable preview results (frontend-generated) for demo purposes.
        """
    )
with col2:
    st.markdown(
        """
        <div class='card'>
            <strong>Quick tips</strong><br/>
            <span class='meta'>
                Use the left sidebar to navigate pages.  
                Click 'Start' below to go to the Upload page.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# Direct navigation
if st.button("Start → Upload", key="start_btn"):
    st.switch_page("pages/Upload.py")

st.markdown("---")
st.markdown(
    "<div class='footer'>Built for Task 1 — Frontend UI/UX (Streamlit). Backend integration will be done separately.</div>",
    unsafe_allow_html=True
)
