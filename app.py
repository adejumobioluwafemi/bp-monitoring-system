# app.py (in root folder)
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time
import json
import os
from typing import Optional
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(
    page_title="Hypertension Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 6px solid #28a745;
        margin: 1rem 0;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 6px solid #dc3545;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.25rem;
        border-radius: 0.75rem;
        border-left: 5px solid #6c757d;
        margin: 0.75rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stButton button {
        width: 100%;
        border-radius: 0.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class AppConfig:
    def __init__(self):
        self.api_url = self._get_api_url()
        
    def _get_api_url(self) -> str:
        """Get API URL from environment variable with fallback"""
        api_url = os.getenv("API_URL")

        if not api_url and hasattr(st, 'secrets') and 'API_URL' in st.secrets:
            self.api_url = st.secrets['API_URL']
        
        if not api_url:
            # Development mode - use localhost
            return "http://localhost:8000"
        
        # Ensure URL has proper protocol
        if not api_url.startswith(('http://', 'https://')):
            api_url = f"https://{api_url}"
            
        return api_url.rstrip('/') 
    
class HypertensionApp:
    def __init__(self, api_url: Optional[str] = None):
        self.config = AppConfig()
        # Allow override via parameter, otherwise use config
        self.api_url = api_url or self.config.api_url
        st.sidebar.info(f"🌐 API: {self.api_url}")
    
    def check_api_health(self):
        """Check if the API is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200, response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException as e:
            return False, {"error": str(e)}
    
    def predict_risk(self, patient_data):
        """Send prediction request to API"""
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json=patient_data,
                timeout=10
            )
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, response.json()
        except requests.exceptions.RequestException as e:
            return False, {"detail": f"Connection error: {str(e)}"}
    
    def get_model_info(self):
        """Get model information from API"""
        try:
            response = requests.get(f"{self.api_url}/model-info", timeout=5)
            return response.status_code == 200, response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException as e:
            return False, {"error": str(e)}
    
    def get_preprocessor_info(self):
        """Get preprocessor information from API"""
        try:
            response = requests.get(f"{self.api_url}/preprocessor-info", timeout=5)
            return response.status_code == 200, response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException as e:
            return False, {"error": str(e)}

def create_patient_form():
    """Create the patient data input form"""
    st.header("📋 Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Clinical Measurements")
        
        age = st.slider(
            "**Age** (years)", 
            min_value=18, 
            max_value=100, 
            value=45,
            help="Patient's age in years"
        )
        
        bmi = st.slider(
            "**BMI** (Body Mass Index)", 
            min_value=15.0, 
            max_value=50.0, 
            value=25.0, 
            step=0.1,
            help="Body Mass Index - healthy range: 18.5-24.9"
        )
        
        systolic_bp = st.slider(
            "**Systolic BP** (mmHg)", 
            min_value=90, 
            max_value=200, 
            value=120,
            help="Systolic blood pressure - normal: <120 mmHg"
        )
        
        diastolic_bp = st.slider(
            "**Diastolic BP** (mmHg)", 
            min_value=60, 
            max_value=130, 
            value=80,
            help="Diastolic blood pressure - normal: <80 mmHg"
        )
        
        heart_rate = st.slider(
            "**Heart Rate** (bpm)", 
            min_value=40, 
            max_value=120, 
            value=72,
            help="Resting heart rate - normal: 60-100 bpm"
        )
    
    with col2:
        st.subheader("👤 Lifestyle & History")
        
        gender = st.selectbox(
            "**Gender**",
            ["Female", "Male"],
            help="Biological sex"
        )
        
        medical_history = st.selectbox(
            "**Do you have a family history of hypertension?**",
            ["No", "Yes"],
            help="History of hypertension in immediate family"
        )
        
        smoking = st.selectbox(
            "**Do you smoke?**",
            ["No", "Yes"],
            help="Current smoking status"
        )
        
        sporting = st.selectbox(
            "**Do you exercise regularly (3-4 times/week)?**",
            ["No", "Yes"],
            help="Regular physical activity 3-4 times per week"
        )
    
    # Create patient data dictionary
    patient_data = {
        "Age": float(age),
        "BMI": float(bmi),
        "Systolic_BP": float(systolic_bp),
        "Diastolic_BP": float(diastolic_bp),
        "Heart_Rate": float(heart_rate),
        "Gender": gender,
        "Medical_History": medical_history,
        "Smoking": smoking,
        "Sporting": sporting
    }
    
    return patient_data

def display_results(result, patient_data):
    """Display prediction results with comprehensive visualization"""
    st.header("📈 Risk Assessment Results")
    
    # Risk level display with enhanced styling
    if result["prediction"] == 0:
        st.markdown('<div class="risk-low">', unsafe_allow_html=True)
        st.markdown("## ✅ Low Hypertension Risk")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="risk-high">', unsafe_allow_html=True)
        st.markdown("## ⚠️ High Hypertension Risk")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    risk_prob = result["probability"]
    confidence = risk_prob if result["prediction"] == 1 else 1 - risk_prob
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Risk Probability", 
            f"{risk_prob:.1%}",
            delta="High Risk" if result['prediction'] == 1 else "Low Risk",
            delta_color="inverse" if result['prediction'] == 1 else "normal"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Risk Level", result['risk_level'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Confidence", f"{confidence:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        risk_category = "High" if result['prediction'] == 1 else "Low"
        st.metric("Risk Category", risk_category)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Interactive risk gauge
    st.subheader("Risk Visualization")
    fig = create_risk_gauge(risk_prob, result["prediction"])
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed recommendation
    st.subheader("💡 Recommendations")
    st.markdown(f'<div class="info-box">', unsafe_allow_html=True)
    st.markdown(f"**{result['message']}**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional recommendations based on risk factors
    additional_recommendations = generate_additional_recommendations(patient_data, result["prediction"])
    if additional_recommendations:
        st.subheader("🎯 Personalized Suggestions")
        for rec in additional_recommendations:
            st.markdown(f"• {rec}")
    
    # Input summary
    with st.expander("📋 View Input Summary"):
        display_input_summary(patient_data)

def create_risk_gauge(risk_prob, prediction):
    """Create an interactive risk gauge chart"""
    risk_value = risk_prob * 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "Hypertension Risk Score",
            'font': {'size': 24, 'color': 'darkblue'}
        },
        delta={
            'reference': 50,
            'increasing': {'color': "red"},
            'decreasing': {'color': "green"}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': "darkblue"
            },
            'bar': {'color': "red" if prediction == 1 else "green"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=50, r=50, t=100, b=50),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def generate_additional_recommendations(patient_data, prediction):
    """Generate personalized recommendations based on input data"""
    recommendations = []
    
    # BMI-based recommendations
    if patient_data["BMI"] > 25:
        recommendations.append("Consider maintaining a healthy weight through balanced diet and exercise")
    
    # Blood pressure recommendations
    if patient_data["Systolic_BP"] > 130 or patient_data["Diastolic_BP"] > 85:
        recommendations.append("Monitor your blood pressure regularly and consider lifestyle modifications")
    
    # Lifestyle recommendations
    if patient_data["Smoking"] == "Yes":
        recommendations.append("Smoking cessation can significantly reduce cardiovascular risk")
    
    if patient_data["Sporting"] == "No":
        recommendations.append("Regular physical activity (30 minutes daily) can help manage blood pressure")
    
    # Age-based recommendations
    if patient_data["Age"] > 50:
        recommendations.append("Regular health screenings are important as risk increases with age")
    
    # High risk specific recommendations
    if prediction == 1:
        recommendations.append("Schedule an appointment with your healthcare provider for comprehensive evaluation")
        recommendations.append("Consider dietary changes to reduce sodium intake")
        recommendations.append("Stress management techniques like meditation may be beneficial")
    
    return recommendations

def display_input_summary(patient_data):
    """Display formatted input summary"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Clinical Measurements:**")
        st.write(f"• **Age:** {patient_data['Age']} years")
        st.write(f"• **BMI:** {patient_data['BMI']:.1f}")
        st.write(f"• **Systolic BP:** {patient_data['Systolic_BP']} mmHg")
        st.write(f"• **Diastolic BP:** {patient_data['Diastolic_BP']} mmHg")
        st.write(f"• **Heart Rate:** {patient_data['Heart_Rate']} bpm")
    
    with col2:
        st.write("**Lifestyle Factors:**")
        st.write(f"• **Gender:** {patient_data['Gender']}")
        st.write(f"• **Family History:** {patient_data['Medical_History']}")
        st.write(f"• **Smoking:** {patient_data['Smoking']}")
        st.write(f"• **Regular Exercise:** {patient_data['Sporting']}")
    
    # Health indicators
    st.write("**Health Indicators:**")
    bmi_status = "Normal" if 18.5 <= patient_data["BMI"] <= 24.9 else "Outside normal range"
    bp_status = "Normal" if patient_data["Systolic_BP"] < 120 and patient_data["Diastolic_BP"] < 80 else "Elevated"
    hr_status = "Normal" if 60 <= patient_data["Heart_Rate"] <= 100 else "Outside normal range"
    
    st.write(f"• **BMI Status:** {bmi_status}")
    st.write(f"• **Blood Pressure Status:** {bp_status}")
    st.write(f"• **Heart Rate Status:** {hr_status}")

def show_educational_content():
    """Show educational information about hypertension"""
    st.header("💡 Understanding Hypertension")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Risk Factors", "Healthy Ranges", "Prevention", "About This Tool"])
    
    with tab1:
        st.subheader("Major Risk Factors")
        st.markdown("""
        - **Age**: Risk increases with age
        - **Family History**: Genetic predisposition to hypertension
        - **High BMI**: Overweight and obesity are significant risk factors
        - **Lifestyle**: Smoking, excessive alcohol, physical inactivity
        - **Diet**: High sodium, low potassium intake
        - **Stress**: Chronic stress can contribute to elevated blood pressure
        - **Existing Conditions**: Diabetes, kidney disease, sleep apnea
        """)
    
    with tab2:
        st.subheader("Healthy Parameter Ranges")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Blood Pressure:**
            - Normal: <120/80 mmHg
            - Elevated: 120-129/<80 mmHg  
            - Hypertension Stage 1: 130-139/80-89 mmHg
            - Hypertension Stage 2: ≥140/≥90 mmHg
            
            **BMI Categories:**
            - Underweight: <18.5
            - Normal: 18.5-24.9
            - Overweight: 25-29.9
            - Obese: ≥30
            """)
        
        with col2:
            st.markdown("""
            **Heart Rate:**
            - Normal Resting: 60-100 bpm
            - Athletes: 40-60 bpm
            
            **Lifestyle Factors:**
            - Exercise: ≥150 minutes/week moderate activity
            - Smoking: Zero tobacco exposure
            - Alcohol: Moderate consumption only
            """)
    
    with tab3:
        st.subheader("Prevention Strategies")
        st.markdown("""
        **Lifestyle Modifications:**
        - Maintain healthy weight (BMI 18.5-24.9)
        - Regular aerobic exercise
        - Balanced diet rich in fruits and vegetables
        - Limit sodium intake to <2,300mg daily
        - Moderate alcohol consumption
        - Smoking cessation
        - Stress management techniques
        
        **Regular Monitoring:**
        - Annual blood pressure checks
        - Regular health screenings
        - Self-monitoring if at risk
        """)
    
    with tab4:
        st.subheader("About This Assessment Tool")
        st.markdown("""
        **Purpose:**
        This tool provides a preliminary assessment of hypertension risk based on 
        established clinical parameters and machine learning analysis.
        
        **Important Notes:**
        - This is a screening tool, not a medical diagnosis
        - Results should be discussed with healthcare professionals
        - Individual risk factors may vary
        - Regular medical checkups are essential
        
        **Technical Information:**
        - Powered by machine learning models trained on clinical data
        - Uses the same preprocessing as model training for consistency
        - Provides probabilistic risk assessment with confidence scores
        """)

def show_debug_info(app):
    """Show debug information for troubleshooting"""
    st.header("🔧 System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Get Model Information", use_container_width=True):
            with st.spinner("Fetching model info..."):
                success, model_info = app.get_model_info()
                if success:
                    st.json(model_info)
                else:
                    st.error("Failed to get model information")
    
    with col2:
        if st.button("Get Preprocessor Information", use_container_width=True):
            with st.spinner("Fetching preprocessor info..."):
                success, preprocessor_info = app.get_preprocessor_info()
                if success:
                    st.json(preprocessor_info)
                else:
                    st.error("Failed to get preprocessor information")
    
    # API status information
    st.subheader("API Status")
    success, health_info = app.check_api_health()
    if success:
        st.json(health_info)
    else:
        st.error("Unable to connect to API")

def main():
    # Initialize app
    app = HypertensionApp()
    
    # Header
    st.markdown('<h1 class="main-header">❤️ Hypertension Risk Assessment</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About This App")
        st.markdown("""
        This tool assesses hypertension risk using machine learning 
        based on clinical parameters and lifestyle factors.
        
        **Features:**
        - Comprehensive risk assessment
        - Personalized recommendations  
        - Educational resources
        - Professional-grade preprocessing
        
        **Note:** This is a screening tool for educational purposes.
        Always consult healthcare professionals for medical advice.
        """)
        
        st.markdown("---")
        
        if st.button("🔄 Reset Application", use_container_width=True):
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Need Help?**")
        if st.button("Debug Information", use_container_width=True):
            st.session_state.show_debug = True
        
        if st.button("Educational Content", use_container_width=True):
            st.session_state.show_education = True
    
    # Check API health
    with st.spinner("🔍 Checking system status..."):
        api_healthy, health_info = app.check_api_health()
    
    # System status display
    if not api_healthy:
        st.error("🚨 API Service Unavailable")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **To start the API service:**
            
            1. Open terminal in project root
            2. Run: `cd src/api && python app.py`
            3. Wait for "Application startup complete"
            4. Refresh this page
            
            **Expected output:**
            - Model loaded successfully
            - Preprocessor ready
            - API running on port 8000
            """)
        
        with col2:
            st.markdown("""
            **Troubleshooting:**
            - Check if port 8000 is available
            - Verify model file exists
            - Ensure all dependencies are installed
            - Check API logs for errors
            """)
            
            if st.button("🔄 Retry Connection", use_container_width=True):
                st.rerun()
        
        return
    
    # Show system status
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.success("✅ API Connected")
    
    with status_col2:
        if health_info.get('model_status') == 'loaded':
            st.success("✅ Model Ready")
        else:
            st.error("❌ Model Not Loaded")
    
    with status_col3:
        if health_info.get('preprocessor_status') == 'ready':
            st.success("✅ Preprocessor Ready")
        else:
            st.warning("⚠️ Basic Preprocessing")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Risk Assessment", "📚 Education Center", "🔧 System Info"])
    
    with tab1:
        # Get patient data from form
        patient_data = create_patient_form()
        
        st.markdown("---")
        
        # Prediction button
        if st.button("🔮 Assess Hypertension Risk", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing risk factors with professional preprocessing..."):
                # Simulate processing time for better UX
                time.sleep(1)
                
                success, result = app.predict_risk(patient_data)
                
                if success:
                    display_results(result, patient_data)
                else:
                    st.error(f"❌ Prediction failed: {result.get('detail', 'Unknown error')}")
                    
                    # Show troubleshooting tips
                    with st.expander("Troubleshooting Tips"):
                        st.markdown("""
                        - Check if the API service is running
                        - Verify the model file exists
                        - Check API logs for detailed error information
                        - Ensure all required parameters are provided
                        """)
    
    with tab2:
        show_educational_content()
    
    with tab3:
        show_debug_info(app)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <i>This hypertension risk assessment tool is for educational and screening purposes only. 
        It is not a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</i>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    # Initialize session state
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False
    if 'show_education' not in st.session_state:
        st.session_state.show_education = False
    
    main()