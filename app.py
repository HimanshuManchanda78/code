import streamlit as st

# Use wide layout so the page stretches across the full browser width
st.set_page_config(layout="wide")

# --- Upper Part (20%) ---
# Create a fixed-height container for the upper section (20% of 900px = 180px).
# border=False hides the container's border so it looks seamless.
upper = st.container(height=int(0.20 * 900), border=False)

# Everything inside this 'with' block is rendered inside the upper container
with upper:
    # File uploader widget — supports drag-and-drop and browse.
    # 'type' restricts allowed file extensions to image formats only.
    # The returned object (uploaded_file) is None until the user uploads a file.
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "gif", "bmp", "webp"])

    # A button that returns True on the frame when clicked, False otherwise.
    evaluate = st.button("Evaluate")

# Draws a horizontal line to visually separate the upper and lower sections
st.divider()

# --- Lower Part (80%) ---
# Create a fixed-height container for the lower section (80% of 600px = 480px).
lower = st.container(height=int(0.80 * 600), border=False)

# Everything inside this 'with' block is rendered inside the lower container
with lower:
    # Only run when the Evaluate button has been clicked
    if evaluate:
        # Check that a file was actually uploaded before trying to use it
        if uploaded_file is not None:
            # Display the uploaded file's name in bold, followed by "Hello World"
            st.write(f"**{uploaded_file.name}** — Hello World")
        else:
            # Show a warning if user clicked Evaluate without uploading a file
            st.warning("Please upload an image first.")
