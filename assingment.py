import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import io
import time
from scipy import ndimage
from skimage import feature
from skimage.color import rgb2gray
import warnings
warnings.filterwarnings('ignore')

# Set wide page layout and custom theme
st.set_page_config(
    page_title="Indian Heritage Classifier Pro",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        transform: translateY(-2px);
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
    h1 {
        color: #FF4B4B;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D4EDDA;
        color: #155724;
        border: 1px solid #C3E6CB;
    }
    .info-box {
        background-color: #D1ECF1;
        color: #0C5460;
        border: 1px solid #BEE5EB;
    }
    </style>
    """, unsafe_allow_html=True)

# Heritage classes
HERITAGE_CLASSES = [
    'Ajanta Caves', 
    'Charminar',
    'Gateway of India', 
    'Hawa Mahal',
    'India Gate',
    'Konark Sun Temple',
    'Mysore Palace',
    'Qutub Minar',
    'Taj Mahal',
    'Victoria Memorial'
]

def create_model():
    """Create and return a new model with correct architecture"""
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(300, 300, 3)
    )
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(HERITAGE_CLASSES), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

@st.cache_resource
def load_model():
    """Load and cache the model with enhanced error handling"""
    with st.spinner("Loading model..."):
        try:
            # Try loading existing model
            model = tf.keras.models.load_model('indian_heritage_model.keras')
            
            # Check if output shape matches
            if model.output_shape[-1] != len(HERITAGE_CLASSES):
                st.warning("Model architecture mismatch. Creating new model...")
                model = create_model()
                
            return model
            
        except Exception as e:
            st.warning(f"Could not load existing model: {str(e)}. Creating new model...")
            model = create_model()
            return model

def validate_model_output(prediction):
    """Validate model prediction output"""
    if not isinstance(prediction, np.ndarray):
        return False
    if prediction.shape[-1] != len(HERITAGE_CLASSES):
        return False
    return True

def extract_advanced_features(image):
    """Extract advanced image features with error handling"""
    try:
        features = {}
        img_array = np.array(image)
        
        # Ensure image is RGB
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack((img_array,) * 3, axis=-1)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
                img_array = img_array[:, :, :3]
                
        gray_img = rgb2gray(img_array)
        
        # Enhanced feature extraction with progress tracking
        progress_bar = st.progress(0)
        
        # 1. Basic Statistics
        features['brightness'] = np.mean(img_array)
        features['contrast'] = np.std(img_array)
        features['saturation'] = np.mean(np.std(img_array, axis=2))
        progress_bar.progress(0.2)
        
        # 2. Color Features
        for i, color in enumerate(['red', 'green', 'blue']):
            features[f'{color}_mean'] = np.mean(img_array[:,:,i])
            features[f'{color}_std'] = np.std(img_array[:,:,i])
        progress_bar.progress(0.4)
        
        # 3. Texture Features
        glcm = feature.graycomatrix(
            (gray_img * 255).astype(np.uint8), 
            distances=[1], 
            angles=[0], 
            levels=256,
            symmetric=True, 
            normed=True
        )
        features['contrast_texture'] = feature.graycoprops(glcm, 'contrast')[0, 0]
        features['homogeneity'] = feature.graycoprops(glcm, 'homogeneity')[0, 0]
        features['energy'] = feature.graycoprops(glcm, 'energy')[0, 0]
        progress_bar.progress(0.6)
        
        # 4. Edge and Shape Features
        edges = feature.canny(gray_img)
        features['edge_density'] = np.mean(edges)
        features['aspect_ratio'] = img_array.shape[1] / img_array.shape[0]
        progress_bar.progress(0.8)
        
        # 5. Advanced Color Analysis
        hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        features['hue_mean'] = np.mean(hsv_img[:,:,0])
        features['saturation_mean'] = np.mean(hsv_img[:,:,1])
        features['value_mean'] = np.mean(hsv_img[:,:,2])
        progress_bar.progress(1.0)
        
        return features
    except Exception as e:
        st.error(f"Error in feature extraction: {str(e)}")
        return None

def preprocess_image(image_bytes, target_size=(300, 300)):
    """Preprocess image with enhanced error handling and validation"""
    try:
        steps = {}
        
        # Original Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        steps['original'] = image
        
        # Validate image size
        if image.size[0] < 100 or image.size[1] < 100:
            st.warning("Image resolution is very low. Results may be inaccurate.")
        
        # Resize
        image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
        steps['resized'] = image_resized
        
        # Convert to array and normalize
        image_array = np.array(image_resized) / 255.0
        steps['normalized'] = image_array
        
        # Add batch dimension
        image_batch = np.expand_dims(image_array, axis=0)
        
        return image_batch, steps
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None, None

def main():
    st.title("🏛️ Indian Heritage Monument Classifier Pro")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to initialize model. Please refresh the page.")
        st.stop()
    
    # File uploader with enhanced UI
    uploaded_file = st.file_uploader(
        "Upload an image of an Indian heritage monument",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file:
        try:
            # Process image
            image_bytes = uploaded_file.read()
            image_batch, preprocessing_steps = preprocess_image(image_bytes)
            
            if image_batch is None or preprocessing_steps is None:
                st.error("Failed to process the image. Please try another image.")
                st.stop()
            
            # Make prediction with validation
            with st.spinner("Analyzing image..."):
                prediction = model.predict(image_batch)[0]
                
                if not validate_model_output(prediction):
                    st.error("Invalid model output. Please check model architecture.")
                    st.stop()
                
                predicted_class_idx = np.argmax(prediction)
                predicted_class = HERITAGE_CLASSES[predicted_class_idx]
                confidence = prediction[predicted_class_idx] * 100
            
            # Display results in tabs
            tabs = st.tabs([
                "Classification Results", 
                "Preprocessing Steps", 
                "Feature Analysis",
                "Model Insights"
            ])
            
            # Extract features
            features = extract_advanced_features(preprocessing_steps['original'])
            
            # Tab 1: Classification Results
            with tabs[0]:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(preprocessing_steps['original'], caption="Uploaded Image")
                    st.markdown(f"""
                        <div class="status-box success-box">
                            <h3>Predicted Monument: {predicted_class}</h3>
                            <p>Confidence: {confidence:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    fig = px.bar(
                        x=HERITAGE_CLASSES,
                        y=prediction,
                        title="Prediction Confidence Distribution"
                    )
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        height=500,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Tab 2: Preprocessing Steps
            with tabs[1]:
                st.write("### Image Preprocessing Pipeline")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(preprocessing_steps['original'], caption="Original")
                    st.markdown(f"""
                        <div class="info-box">
                            Size: {preprocessing_steps['original'].size}
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.image(preprocessing_steps['resized'], caption="Resized")
                    st.markdown(f"""
                        <div class="info-box">
                            Size: {preprocessing_steps['resized'].size}
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    st.image(preprocessing_steps['normalized'], caption="Normalized")
                    st.markdown(f"""
                        <div class="info-box">
                            Range: [0, 1]
                        </div>
                        """, unsafe_allow_html=True)
            
            # Tab 3: Feature Analysis
            if features:
                with tabs[2]:
                    st.write("### Advanced Feature Analysis")
                    
                    # Convert features to DataFrame
                    df = pd.DataFrame([features]).T
                    df.columns = ['Value']
                    
                    # Create feature groups
                    feature_groups = {
                        'Color Features': [col for col in df.index if any(c in col for c in ['red', 'green', 'blue', 'hue'])],
                        'Texture Features': ['contrast_texture', 'homogeneity', 'energy'],
                        'Shape Features': ['aspect_ratio', 'edge_density'],
                        'Statistical Features': ['brightness', 'contrast', 'saturation']
                    }
                    
                    # Display feature groups in tabs
                    feature_tabs = st.tabs(list(feature_groups.keys()))
                    for tab, (group_name, group_features) in zip(feature_tabs, feature_groups.items()):
                        with tab:
                            fig = px.bar(
                                df.loc[group_features],
                                orientation='h',
                                title=f"{group_name} Distribution"
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
            
            # Tab 4: Model Insights
            with tabs[3]:
                st.write("### Model Architecture")
                model_summary = []
                model.summary(print_fn=lambda x: model_summary.append(x))
                st.code('\n'.join(model_summary))
                
                st.write("### Model Performance Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': ['Total Parameters', 'Trainable Parameters', 'Non-trainable Parameters'],
                    'Value': [
                        f"{model.count_params():,}",
                        f"{sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}",
                        f"{sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights):,}"
                    ]
                })
                st.table(metrics_df)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try uploading a different image or refresh the page.")
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This advanced AI-powered classifier identifies Indian heritage monuments with detailed analysis including:
        - High-accuracy monument classification
        - Confidence scoring
        - Advanced feature extraction
        - Model architecture insights
        """)
        
        st.header("Supported Monuments")
        for monument in HERITAGE_CLASSES:
            st.markdown(f"• {monument}")
        
        if model:
            st.header("Model Information")
            st.markdown(f"""
            - Input Shape: {model.input_shape[1:]}
            - Number of Classes: {len(HERITAGE_CLASSES)}
            - Model Type: ResNet50-based CNN
            """)
        
        st.header("Tips")
        st.markdown("""
        For best results:
        - Use clear, well-lit images
        - Ensure monument is centered
        - Avoid extreme angles
        - Use high-resolution images (minimum 300x300 pixels)
        - Avoid images with heavy filters or text overlays
        """)
        
        st.header("Performance Notes")
        st.markdown("""
        - Model uses ResNet50 architecture
        - Pre-trained on ImageNet
        - Fine-tuned for Indian monuments
        - Optimized for real-time inference
        """)
        
        # Add a download button for sample images
        st.header("Sample Images")
        if st.button("Download Sample Images"):
            # Here you would implement the logic to download sample images
            st.info("Sample images would be downloaded here (feature coming soon)")

def augment_image(image):
    """Apply real-time data augmentation for better prediction"""
    try:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Convert image to array if it's not already
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Expand dimensions to match expected shape
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        # Generate augmented images
        augmented_images = []
        for batch in datagen.flow(image, batch_size=1, shuffle=False):
            augmented_images.append(batch[0])
            if len(augmented_images) >= 5:  # Generate 5 augmented versions
                break
                
        return augmented_images
    except Exception as e:
        st.warning(f"Image augmentation failed: {str(e)}")
        return None

def ensemble_predict(model, image_batch, preprocessing_steps):
    """Make ensemble predictions using original and augmented images"""
    try:
        predictions = []
        
        # Predict on original image
        orig_pred = model.predict(image_batch)
        predictions.append(orig_pred)
        
        # Get augmented versions and predict
        augmented_images = augment_image(preprocessing_steps['resized'])
        if augmented_images:
            for aug_img in augmented_images:
                # Preprocess augmented image
                aug_batch = np.expand_dims(aug_img, axis=0)
                aug_pred = model.predict(aug_batch)
                predictions.append(aug_pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    except Exception as e:
        st.warning(f"Ensemble prediction failed: {str(e)}")
        return None

def calculate_certainty_metrics(prediction):
    """Calculate additional certainty metrics for the prediction"""
    try:
        metrics = {}
        
        # Sort predictions in descending order
        sorted_preds = np.sort(prediction)[::-1]
        
        # Calculate entropy (uncertainty measure)
        entropy = -np.sum(prediction * np.log2(prediction + 1e-10))
        metrics['entropy'] = entropy
        
        # Calculate margin (difference between top two predictions)
        metrics['margin'] = sorted_preds[0] - sorted_preds[1]
        
        # Calculate ratio of top prediction to mean of others
        metrics['dominance_ratio'] = sorted_preds[0] / np.mean(sorted_preds[1:])
        
        return metrics
    except Exception as e:
        st.warning(f"Error calculating certainty metrics: {str(e)}")
        return None

class PredictionCache:
    """Simple cache for storing recent predictions"""
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
    
    def add(self, image_hash, prediction):
        """Add a prediction to the cache"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[image_hash] = prediction
    
    def get(self, image_hash):
        """Get a prediction from the cache"""
        return self.cache.get(image_hash)

# Initialize prediction cache
prediction_cache = PredictionCache()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again.")