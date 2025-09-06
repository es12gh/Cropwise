import streamlit as st
import joblib
import os
from PIL import Image
import google.generativeai as genai

# ✅ Configure Gemini API key
genai.configure(api_key="AIzaSyAsEaVsf6LnoxPu27phiWroGYOYXfSxkhw")

# ✅ Generation config
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# ✅ Language options (all major Indian languages)
languages = {
    "English": "English",
    "Hindi": "Hindi",
    "Bengali": "Bengali",
    "Tamil": "Tamil",
    "Telugu": "Telugu",
    "Kannada": "Kannada",
    "Malayalam": "Malayalam",
    "Gujarati": "Gujarati",
    "Marathi": "Marathi",
    "Punjabi": "Punjabi",
    "Odia": "Odia",
    "Urdu": "Urdu"
}

# ✅ Cache Gemini model
@st.cache_resource
def get_gemini_model(system_instruction=None):
    return genai.GenerativeModel(
        model_name="gemini-2.5-pro",
        generation_config=generation_config,
        system_instruction=system_instruction
    )

# ✅ Cache model loader
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# ✅ Cache image loader
@st.cache_resource
def load_image():
    return Image.open(r'C:\Users\ghosh\Desktop\CropWise\data\cc.jpg')

# ✅ Function to translate text (English → Target Language)
def translate_text(text, target_language):
    gemini = get_gemini_model()
    response = gemini.generate_content(
        f"Translate the following crop details into {target_language}:\n\n{text}"
    )
    return response.text

# ✅ Main app
def main():
    st.title("🌱 CropWise - Smart Crop Recommender")

    # Show image
    image = load_image()
    st.image(image, use_container_width=True)

    # Header
    st.markdown("""
    <div style="background-color:teal; padding:10px">
        <h2 style="color:white; text-align:center;">Find The Most Suitable Crop</h2>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar model selection
    activities = [
        'Naive Bayes (The Best Model)',
        'Logistic Regression',
        'Decision Tree',
        'Random Forest'
    ]
    option = st.sidebar.selectbox("Select model:", activities)

    # 🌐 Language selection
    selected_language = st.sidebar.selectbox("🌐 Select Language", list(languages.keys()))

    st.subheader(option)

    # Inputs
    sn = st.slider('NITROGEN (N)', 0.0, 150.0)
    sp = st.slider('PHOSPHOROUS (P)', 0.0, 150.0)
    pk = st.slider('POTASSIUM (K)', 0.0, 210.0)
    pt = st.slider('TEMPERATURE (°C)', 0.0, 50.0)
    phu = st.slider('HUMIDITY (%)', 0.0, 100.0)
    pPh = st.slider('SOIL pH', 0.0, 14.0)
    pr = st.slider('RAINFALL (mm)', 0.0, 300.0)
    inputs = [[sn, sp, pk, pt, phu, pPh, pr]]

    # ✅ Model mapping
    base_path = os.path.dirname(__file__)
    model_map = {
        'Logistic Regression': load_model(os.path.join(base_path, 'LogisticRegression_model.pkl')),
        'Decision Tree': load_model(os.path.join(base_path, 'DecisionTree_model.pkl')),
        'Naive Bayes (The Best Model)': load_model(os.path.join(base_path, 'NaiveBayes_model.pkl')),
        'Random Forest': load_model(os.path.join(base_path, 'RandomForest_model.pkl'))
    }

    # Classification button
    if st.button('Classify'):
        with st.spinner("Classifying with selected model..."):
            selected_model = model_map[option]
            prediction = selected_model.predict(inputs)[0]
            st.success(f"✅ Recommended Crop: **{prediction}**")
            st.session_state['classification_result'] = prediction

    # Gemini crop details
    if 'classification_result' in st.session_state:
        result = st.session_state['classification_result']
        if st.button(f'Know more about {result}'):
            with st.spinner("Generating crop insights..."):
                if selected_language == "English":
                    # Direct English generation
                    system_instruction = """
                    Given the name of a crop, your task is to generate detailed insights about it in English.
                    Your response should include:
                    1. Climate Requirements
                    2. Sowing Time
                    3. Soil Preparation
                    4. Spacing and Planting Depth
                    5. Fertilizer Recommendations
                    6. Pesticides and Pest Control
                    7. Irrigation Requirements
                    8. Growth Stages and Care
                    9. Harvesting Time
                    10. Equipment Needed
                    11. Companion and Similar Crops
                    """
                    gemini = get_gemini_model(system_instruction)
                    response = gemini.generate_content("Crop Name: " + result)
                    st.write(response.text)
                else:
                    # Generate in English first → then translate
                    gemini = get_gemini_model()
                    response = gemini.generate_content("Crop Name: " + result)
                    translated_text = translate_text(response.text, selected_language)
                    st.write(translated_text)

# ✅ Run app
if __name__ == '__main__':
    main()

