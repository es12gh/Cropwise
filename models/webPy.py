import streamlit as st
import pickle
from PIL import Image
import joblib
import base64
import os
import google.generativeai as genai
# ✅ Configure Gemini API key
genai.configure(api_key="AIzaSyAsEaVsf6LnoxPu27phiWroGYOYXfSxkhw")# or hardcode for testing

# ✅ Define generation config
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# ✅ System prompt
system_instruction = """
Given the name of a crop, your task is to generate detailed insights about it. Your response should include the following information:
1. Climate Requirements: Ideal temperature, humidity, and rainfall levels.
2. Sowing Time: The best planting season or month(s) for optimal growth.
3. Soil Preparation: The type of soil needed, pH levels, and preparation tips.
4. Spacing and Planting Depth: Appropriate spacing between plants and recommended planting depth.
5. Fertilizer Recommendations: Types, quantities, and application schedule of fertilizers for each growth stage.
6. Pesticides and Pest Control: Common pests for this crop, recommended pesticides, and organic alternatives.
7. Irrigation Requirements: Frequency and amount of water required at different stages.
8. Growth Stages and Care: Key growth milestones, pruning, weeding, and other care tips.
9. Harvesting Time: Signs of maturity, best time to harvest, and post-harvest handling.
10. Equipment Needed: Essential tools and machinery for cultivation, maintenance, and harvesting.
11. Companion and Similar Crops: Suggestions for intercropping or similar crops with matching climate and soil needs.

Please also include any other relevant information or tips to help farmers maximize yield and ensure a healthy crop.

Strictly follow the format in the above example, not the points of the example.
"""

# ✅ Load the Gemini model
model = genai.GenerativeModel(
    model_name="gemini-2.5-pro",
    generation_config=generation_config,
    system_instruction=system_instruction
)

# ✅ Define local model paths
base_path = os.path.dirname(__file__)
model_path1 = os.path.join(base_path, 'LogisticRegression_model.pkl')
model_path2 = os.path.join(base_path, 'DecisionTree_model.pkl')
model_path3 = os.path.join(base_path, 'NaiveBayes_model.pkl')
model_path4 = os.path.join(base_path, 'RandomForest_model.pkl')

# ✅ Classification helper
def classify(pred):
    return pred[0]

# ✅ Streamlit main function
def main():
    st.title("CropWise (Crop Recommender)")
    
    # Load and show image
    base_path = os.path.dirname(r'C:\Users\ghosh\Desktop\CropWise')
    image_path = os.path.join(r'C:\Users\ghosh\Desktop\CropWise', '..', 'data', 'cc.jpg')
    image = Image.open(image_path)
    st.image(image, use_column_width=True)
    
    # Header
    html_temp = """
    <div style="background-color:teal; padding:10px">
    <h2 style="color:white;text-align:center;">Find The Most Suitable Crop</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Model selection
    activities = ['Naive Bayes (The Best Model)', 'Logistic Regression', 'Decision Tree', 'Random Forest']
    option = st.sidebar.selectbox("Which model would you like to use?", activities)
    st.subheader(option)
    
    # Inputs
    sn = st.slider('NITROGEN (N)', 0.0, 150.0)
    sp = st.slider('PHOSPHOROUS (P)', 0.0, 150.0)
    pk = st.slider('POTASSIUM (K)', 0.0, 210.0)
    pt = st.slider('TEMPERATURE', 0.0, 50.0)
    phu = st.slider('HUMIDITY', 0.0, 100.0)
    pPh = st.slider('Ph', 0.0, 14.0)
    pr = st.slider('RAINFALL', 0.0, 300.0)
    
    inputs = [[sn, sp, pk, pt, phu, pPh, pr]]

    # Classification button
    if st.button('Classify'):
        if option == 'Logistic Regression':
            model1 = joblib.load(model_path1)
            result = classify(model1.predict(inputs))
        elif option == 'Decision Tree':
            model2 = joblib.load(model_path2)
            result = classify(model2.predict(inputs))
        elif option == 'Naive Bayes (The Best Model)':
            model3 = joblib.load(model_path3)
            result = classify(model3.predict(inputs))
        else:
            model4 = joblib.load(model_path4)
            result = classify(model4.predict(inputs))
        
        st.success(f"Recommended Crop: {result}")
        st.session_state['classification_result'] = result

    # Gemini integration
    if 'classification_result' in st.session_state:
        result = st.session_state['classification_result']
        if st.button(f'Know more about {result}'):
            with st.spinner("Generating crop information..."):
                response = model.generate_content("Crop Name: " + result)
                st.markdown(response.text)

# ✅ Run the app
if __name__ == '__main__':
    main()
