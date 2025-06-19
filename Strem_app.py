import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download, login

# Configuration
MODEL_REPO = "Multiclass_Fish_Image_Classification"
MODEL_FILE = "src/best_fish_model.keras"
TOKEN = "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"

# Class names — ensure this matches your training order
class_names = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

# Initialize session state for model caching
if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_resource
def load_model_from_hf():
    """Load model from Hugging Face Hub with caching"""
    try:
        # Authenticate with Hugging Face
        login(token=TOKEN)
        
        # Download model
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            token=TOKEN
        )
        
        # Load the Keras model
        model = tf.keras.models.load_model(model_path)
        return model
        
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub: {str(e)}")
        st.stop()

# App title and description
st.title("🐟 Fish Species Classifier")
st.markdown("""
Upload an image of a fish to classify its species. The model can identify various fish types including:
- Bass
- Black Sea Sprat
- Gilt Head Bream
- Hourse Mackerel
- Red Mullet
- And more!
""")

# Sidebar with additional info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app uses a deep learning model trained on fish images to classify species.
    The model is hosted on Hugging Face Hub.
    """)
    st.markdown("[View Model on Hugging Face](https://huggingface.co/Multiclass_Fish_Image_Classification)")

# File uploader
uploaded_file = st.file_uploader(
    "Upload a fish image", 
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

# Load model (only once)
if st.session_state.model is None:
    with st.spinner("Loading classification model..."):
        st.session_state.model = load_model_from_hf()

# Only run prediction after image is uploaded
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image with error handling
    try:
        image = image.convert("RGB")  # Ensure 3 channels
        image = image.resize((224, 224))  # Resize to match training input size
        img_array = np.array(image).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.stop()

    # Predict with loading indicator
    with st.spinner("Analyzing fish species..."):
        try:
            predictions = st.session_state.model.predict(img_array)[0]
            top_prediction_idx = np.argmax(predictions)
            predicted_class = class_names[top_prediction_idx]
            confidence_score = predictions[top_prediction_idx] * 100
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.stop()

    # Display prediction results
    st.success("Analysis complete!")
    
    # Main prediction
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🐠 Predicted Species")
        st.markdown(f"**{predicted_class}**")
    with col2:
        st.markdown("### 🔍 Confidence Score")
        st.markdown(f"**{confidence_score:.2f}%**")
    
    # Show top 3 predictions in an expandable section
    with st.expander("See detailed predictions"):
        st.markdown("### Top 3 Predictions:")
        top_3 = predictions.argsort()[-3:][::-1]
        
        for i in top_3:
            # Create a progress bar for each prediction
            pred_score = predictions[i] * 100
            st.markdown(f"**{class_names[i]}**")
            st.progress(int(pred_score))
            st.markdown(f"{pred_score:.2f}%")
            st.markdown("---")

    # Add some space at the bottom
    st.markdown("")
    st.markdown("*Try another image to see different results!*")