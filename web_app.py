"""
Streamlit Web Application for Currency Detection
User-friendly interface for the Currency Detection System
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import io
import logging
from pathlib import Path

from currency_detector import CurrencyDetectionSystem
from config import SUPPORTED_CURRENCIES, RESULTS_DIR

# Configure Streamlit page
st.set_page_config(
    page_title="Currency Detection System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging for web app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = CurrencyDetectionSystem()
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

def load_image(uploaded_file):
    """Load and process uploaded image"""
    try:
        # Read image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Resize for processing
        image_resized = cv2.resize(image_array, (224, 224))
        
        return image, image_resized
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None, None

def display_prediction_results(result):
    """Display prediction results in a formatted way"""
    # Main prediction
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if result['prediction'] == 'Real':
            st.success(f"✅ **{result['prediction']}**")
        else:
            st.error(f"❌ **{result['prediction']}**")
    
    with col2:
        confidence_color = "green" if result['confidence_score'] > 0.7 else "orange" if result['confidence_score'] > 0.5 else "red"
        st.markdown(f"**Confidence:** <span style='color:{confidence_color}'>{result['confidence_score']:.2%}</span>", 
                   unsafe_allow_html=True)
    
    with col3:
        level_color = {"High": "green", "Medium": "orange", "Low": "red"}[result['confidence_level']]
        st.markdown(f"**Level:** <span style='color:{level_color}'>{result['confidence_level']}</span>", 
                   unsafe_allow_html=True)
    
    # Detailed results
    with st.expander("📊 Detailed Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Individual Model Predictions")
            
            ml_pred = result['individual_predictions']['traditional_cv_ml']
            dl_pred = result['individual_predictions']['deep_learning']
            
            # Create comparison chart
            models = ['Traditional CV + ML', 'Deep Learning']
            confidences = [ml_pred['confidence'], dl_pred['confidence']]
            predictions = [ml_pred['prediction'], dl_pred['prediction']]
            
            df_models = pd.DataFrame({
                'Model': models,
                'Confidence': confidences,
                'Prediction': predictions
            })
            
            fig = px.bar(df_models, x='Model', y='Confidence', color='Prediction',
                        title='Model Comparison', color_discrete_map={'Real': 'green', 'Fake': 'red'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Analysis Details")
            st.write(f"**Currency:** {result['currency']}")
            st.write(f"**Processing Time:** {result['processing_time']:.3f} seconds")
            st.write(f"**Features Extracted:** {result['features_extracted']}")
            st.write(f"**Analysis Time:** {result['timestamp']}")

def create_confidence_gauge(confidence_score):
    """Create a confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 60], 'color': "gray"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    # Header
    st.title("💰 Advanced Currency Detection System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Controls")
        
        # Currency selection
        selected_currency = st.selectbox(
            "Select Currency",
            SUPPORTED_CURRENCIES,
            help="Choose the currency type for detection"
        )
        
        # System status
        st.header("📊 System Status")
        if st.session_state.detector.is_trained:
            st.success("✅ System Trained")
        else:
            st.warning("⚠️ System Not Trained")
        
        # Training section
        st.header("🚀 Training")
        if st.button("🎯 Train System", help="Train the detection models"):
            with st.spinner(f'Training models for {selected_currency}...'):
                try:
                    st.session_state.detector.train_system(selected_currency)
                    st.success(f"✅ System trained successfully for {selected_currency}!")
                except Exception as e:
                    st.error(f"❌ Training failed: {e}")
        
        # Model management
        st.header("💾 Model Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Models"):
                try:
                    st.session_state.detector.save_models(selected_currency)
                    st.success("Models saved!")
                except Exception as e:
                    st.error(f"Save failed: {e}")
        
        with col2:
            if st.button("📁 Load Models"):
                try:
                    st.session_state.detector.load_models(selected_currency)
                    st.success("Models loaded!")
                except Exception as e:
                    st.error(f"Load failed: {e}")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Detection", "📊 Analytics", "📈 System Info", "📋 History"])
    
    with tab1:
        st.header("Currency Authentication")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Currency Image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image of the currency note to analyze"
        )
        
        if uploaded_file is not None:
            # Load and display image
            original_image, processed_image = load_image(uploaded_file)
            
            if original_image is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Uploaded Image")
                    st.image(original_image, caption="Original Image", use_column_width=True)
                
                with col2:
                    st.subheader("Processed Image")
                    st.image(processed_image, caption="Processed for Analysis", use_column_width=True)
                
                # Detection button
                if st.button("🔍 Analyze Currency", type="primary"):
                    if not st.session_state.detector.is_trained:
                        st.error("❌ Please train the system first using the sidebar controls.")
                    else:
                        with st.spinner('Analyzing currency...'):
                            try:
                                # Perform detection
                                result = st.session_state.detector.predict_single_image(
                                    processed_image, selected_currency
                                )
                                
                                # Add to history
                                st.session_state.detection_history.append(result)
                                
                                # Display results
                                st.markdown("---")
                                st.subheader("🎯 Detection Results")
                                display_prediction_results(result)
                                
                                # Confidence gauge
                                st.subheader("📊 Confidence Gauge")
                                fig_gauge = create_confidence_gauge(result['confidence_score'])
                                st.plotly_chart(fig_gauge, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"❌ Detection failed: {e}")
                                logger.error(f"Detection error: {e}")
    
    with tab2:
        st.header("System Analytics")
        
        if st.session_state.detector.is_trained:
            if st.button("🔄 Run System Evaluation"):
                with st.spinner('Evaluating system performance...'):
                    try:
                        evaluation_results = st.session_state.detector.evaluate_system(selected_currency)
                        
                        # Display evaluation results
                        st.subheader("📊 Performance Metrics")
                        
                        # Ensemble performance
                        ensemble_metrics = evaluation_results['ensemble']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{ensemble_metrics['accuracy']:.2%}")
                        with col2:
                            st.metric("Precision", f"{ensemble_metrics['precision']:.2%}")
                        with col3:
                            st.metric("Recall", f"{ensemble_metrics['recall']:.2%}")
                        with col4:
                            st.metric("F1-Score", f"{ensemble_metrics['f1']:.2%}")
                        
                        # Model comparison
                        st.subheader("🔍 Model Comparison")
                        
                        # Prepare data for visualization
                        model_data = []
                        
                        # Add ML models
                        for model_name, metrics in evaluation_results.get('ml_models', {}).items():
                            model_data.append({
                                'Model': f"ML: {model_name}",
                                'Accuracy': metrics.get('accuracy', 0),
                                'Precision': metrics.get('precision', 0),
                                'Recall': metrics.get('recall', 0),
                                'F1': metrics.get('f1', 0),
                                'Type': 'Machine Learning'
                            })
                        
                        # Add DL models
                        for model_name, metrics in evaluation_results.get('dl_models', {}).items():
                            model_data.append({
                                'Model': f"DL: {model_name}",
                                'Accuracy': metrics.get('accuracy', 0),
                                'Precision': metrics.get('precision', 0),
                                'Recall': metrics.get('recall', 0),
                                'F1': metrics.get('f1', 0),
                                'Type': 'Deep Learning'
                            })
                        
                        # Add ensemble
                        model_data.append({
                            'Model': 'Ensemble',
                            'Accuracy': ensemble_metrics['accuracy'],
                            'Precision': ensemble_metrics['precision'],
                            'Recall': ensemble_metrics['recall'],
                            'F1': ensemble_metrics['f1'],
                            'Type': 'Ensemble'
                        })
                        
                        if model_data:
                            df_models = pd.DataFrame(model_data)
                            
                            # Create comparison chart
                            fig = px.bar(df_models, x='Model', y='F1', color='Type',
                                        title='Model Performance Comparison (F1-Score)',
                                        labels={'F1': 'F1-Score'})
                            fig.update_xaxis(tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Detailed metrics table
                            st.subheader("📋 Detailed Metrics")
                            st.dataframe(df_models)
                    
                    except Exception as e:
                        st.error(f"❌ Evaluation failed: {e}")
                        logger.error(f"Evaluation error: {e}")
        else:
            st.info("🔄 Please train the system first to view analytics.")
    
    with tab3:
        st.header("System Information")
        
        # Get system info
        system_info = st.session_state.detector.get_system_info()
        
        # Display system info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Basic Information")
            st.write(f"**System Name:** {system_info['system_name']}")
            st.write(f"**Version:** {system_info['version']}")
            st.write(f"**Status:** {'✅ Trained' if system_info['is_trained'] else '⚠️ Not Trained'}")
            st.write(f"**Supported Currencies:** {', '.join(system_info['supported_currencies'])}")
            
            st.subheader("🎯 Confidence Thresholds")
            for level, threshold in system_info['confidence_thresholds'].items():
                st.write(f"**{level.replace('_', ' ').title()}:** {threshold:.1%}")
        
        with col2:
            st.subheader("🔧 Detection Approaches")
            for approach in system_info['approaches']:
                st.write(f"• {approach}")
            
            st.subheader("🧠 Features by Approach")
            for approach, features in system_info['features'].items():
                with st.expander(f"{approach.replace('_', ' ').title()}"):
                    for feature in features:
                        st.write(f"- {feature}")
    
    with tab4:
        st.header("Detection History")
        
        if st.session_state.detection_history:
            # Summary statistics
            total_detections = len(st.session_state.detection_history)
            real_count = sum(1 for r in st.session_state.detection_history if r['prediction'] == 'Real')
            fake_count = total_detections - real_count
            avg_confidence = np.mean([r['confidence_score'] for r in st.session_state.detection_history])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Detections", total_detections)
            with col2:
                st.metric("Real Currency", real_count)
            with col3:
                st.metric("Fake Currency", fake_count)
            with col4:
                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
            
            # History chart
            history_df = pd.DataFrame(st.session_state.detection_history)
            
            fig = px.scatter(history_df, x=range(len(history_df)), y='confidence_score',
                           color='prediction', title='Detection History',
                           labels={'x': 'Detection Number', 'confidence_score': 'Confidence Score'},
                           color_discrete_map={'Real': 'green', 'Fake': 'red'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed history table
            st.subheader("📋 Detailed History")
            display_df = history_df[['timestamp', 'currency', 'prediction', 'confidence_score', 'confidence_level', 'processing_time']]
            st.dataframe(display_df)
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.detection_history = []
                st.experimental_rerun()
        
        else:
            st.info("🔍 No detection history available. Start by analyzing some currency images!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip:** For best results, ensure currency images are well-lit, clear, and focused. "
        "The system works best with high-quality images of complete currency notes."
    )

if __name__ == "__main__":
    main()