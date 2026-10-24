import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Page Configuration
st.set_page_config(page_title="Nigeria National Emergency AI", page_icon="🇳🇬", layout="wide")

st.title("🇳🇬 Nigeria National AI Emergency Dispatch System")
st.caption("Serving all 36 States & FCT — Supporting English, Pidgin, Hausa, Yoruba & Igbo")

# All 36 States + FCT
NIGERIA_STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno",
    "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "FCT (Abuja)", "Gombe",
    "Imo", "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi", "Kwara",
    "Lagos", "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo", "Plateau",
    "Rivers", "Sokoto", "Taraba", "Yobe", "Zamfara"
]

# Primary Agency Mapping
AGENCY_MAP = {
    "Fire": "Federal / State Fire Service",
    "Traffic": "Federal Road Safety Corps (FRSC)",
    "Security": "Nigeria Police Force (NPF) / Civil Defence (NSCDC)",
    "Medical": "National Emergency Management Agency (NEMA) / State Health Services"
}

# Load and Train Multi-Lingual Engine
@st.cache_resource
def load_and_train_model():
    data = {
        'text': [
            # English
            "Fire outbreak in the residential building", 
            "Multiple car crash on the expressway",
            "Armed robbery ongoing near the commercial bank",
            "Person collapsed and needs immediate medical response",
            # Nigerian Pidgin
            "Fire deh burn house serious for street",
            "Motors don jam themselves for highway blood dey flow",
            "Bad boys dey drag bag and fire gun for junction",
            "Person don faint heavily he no dey breathe well",
            # Hausa
            "Ina bukatan taimako gobiya tana ci",
            "Hatsarin mota ya faru a kan hanya",
            "Yan fashi suna harbi a kasuwa",
            "Mutum ya yanke jiki ya fadi yana bukatan likita",
            # Yoruba
            "Ina njo ile ni agbegbe wa",
            "Oti wa ijamba oko lori titi",
            "Awon ole nbonilowo ni ile ifowopamo",
            "Eyan kan ti gbe emi mi leri aisan gbona",
            # Igbo
            "Akwukwo oku na-agba na ulo",
            "Ihe mberede ugbo ala mere n'okporo lei",
            "Amonye ochi na-agba egbe na ahia",
            "Mmadu adawo mba na-acho enyemaka emezie"
        ],
        'category': [
            "Fire", "Traffic", "Security", "Medical",
            "Fire", "Traffic", "Security", "Medical",
            "Fire", "Traffic", "Security", "Medical",
            "Fire", "Traffic", "Security", "Medical",
            "Fire", "Traffic", "Security", "Medical"
        ]
    }
    
    df = pd.DataFrame(data)
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['text'])
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['category'])
    
    model = XGBClassifier(eval_metric='mlogloss')
    model.fit(X, y)
    
    return vectorizer, label_encoder, model

vectorizer, label_encoder, model = load_and_train_model()

# Dispatch Portal Layout
st.subheader("Report an Emergency (Mberede / Gaggawa / Mberede)")

col1, col2 = st.columns(2)

with col1:
    selected_state = st.selectbox("Select State", NIGERIA_STATES)
    location_details = st.text_input("LGA / Specific Location Details", placeholder="e.g. Maitama, Lokoja Expressway, Ikeja")

with col2:
    caller_phone = st.text_input("Reporter Phone Number", placeholder="080XXXXXXXX")

emergency_text = st.text_area(
    "Describe the Emergency (English, Pidgin, Hausa, Yoruba, Igbo)",
    placeholder="e.g., Motors don jam for road OR Ina bukatan taimako gobiya..."
)

if st.button("Submit Dispatch Alert", type="primary"):
    if not emergency_text or not location_details:
        st.error("Please provide both emergency details and location.")
    else:
        X_input = vectorizer.transform([emergency_text])
        prediction = model.predict(X_input)[0]
        category = label_encoder.inverse_transform([prediction])[0]
        assigned_agency = AGENCY_MAP.get(category, "General Emergency Unit")
        
        st.success("Emergency Classified and Routed Successfully!")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Target State", selected_state)
        res_col2.metric("Threat Category", category)
        res_col3.metric("Assigned Agency", assigned_agency)
        
        st.info(f"**Live Dispatch Log:** Alert queued for {assigned_agency} command center in **{selected_state} ({location_details})**.")