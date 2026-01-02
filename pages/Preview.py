"""
Preview.py
Page: Preview
Applies safe transformations (UI-level) to show what the result will look like.
Frontend-only preview; backend handles full validation and heavy transforms.
"""

import streamlit as st
from utils import inject_css, safe_apply_preview_transformations

# Inject CSS
inject_css()

st.title("👀 Preview")
st.write("This page shows a **frontend-only preview** of the data after applying your selected transformations.")

# Fetch uploaded data and transformations
df = st.session_state.get("uploaded_df")
trans = st.session_state.get("transformations", {})

# No file uploaded
if df is None:
    st.warning("⚠️ Upload a file first on the 'Upload' page.")
    st.stop()

# Apply transformations or show plain data
if not trans:
    st.info("ℹ️ No transformations selected. Showing original data.")
    preview_df = df.copy()
else:
    with st.spinner("Applying safe preview transformations..."):
        preview_df = safe_apply_preview_transformations(df, trans)

# Save preview in session
st.session_state["preview_df"] = preview_df

# Display metadata
st.markdown(
    f"### 📊 Preview Summary\n"
    f"**Rows:** {preview_df.shape[0]} &nbsp;&nbsp; "
    f"**Columns:** {preview_df.shape[1]}"
)

# Handle empty data after filtering
if preview_df.empty:
    st.warning("⚠️ All rows were filtered out by your transformations. No data to display.")
else:
    st.markdown("**Preview (first 20 rows)**")
    st.dataframe(preview_df.head(20), use_container_width=True)

# -----------------------------
# Navigation buttons
# -----------------------------
st.write("")
c1, c2 = st.columns([1,1])

with c1:
    if st.button("⬅️ Back to Transformations"):
        st.success("Navigating to Transformation page...")
        st.switch_page("pages/Transformation.py")

with c2:
    if st.button("Proceed to Output →"):
        st.success("Navigating to Output page...")
        st.switch_page("pages/Output.py")
