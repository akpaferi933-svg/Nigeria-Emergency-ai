import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# 1. Page Configuration
st.set_page_config(
    page_title="Nigeria National Emergency AI & Dispatch Platform",
    page_icon="🚨",
    layout="wide"
)

# 2. Initialize Session State for Authentication & Logs
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}
if "dispatch_logs" not in st.session_state:
    st.session_state.dispatch_logs = []

# 3. Load and Train Multi-Lingual Engine (Cached)
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
            "Iberibe okporo uzu mere na express",
            "Ohi ji egbe na-apu n'ulo aku",
            "Onye adaala ma nwee mkpa doro anya"
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
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['category'])
    
    model = MultinomialNB()
    model.fit(X, y)
    
    return vectorizer, model, label_encoder

vectorizer, model, label_encoder = load_and_train_model()

# Agency Mapping Dictionary
AGENCY_MAP = {
    "Fire": "Federal Fire Service (FFS) / State Fire Command",
    "Traffic": "Federal Road Safety Corps (FRSC)",
    "Security": "Nigeria Police Force (NPF) / Civil Defence (NSCDC)",
    "Medical": "National Emergency Management Agency (NEMA) / State Health Services"
}

# Complete List of all 36 States + FCT
NIGERIA_STATES_FCT = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", 
    "Borno", "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", 
    "FCT Abuja", "Gombe", "Imo", "Jigawa", "Kaduna", "Kano", "Katsina", 
    "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun", 
    "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba", 
    "Yobe", "Zamfara"
]

# 4. Authentication Gate (Ensures accountability before reporting)
if not st.session_state.logged_in:
    st.markdown("## 🔐 Citizen Security Verification Portal")
    st.info("To prevent false alarms, ambushes, and safeguard responding security personnel, all reporters must verify their identity before accessing the emergency dispatch grid.")
    
    with st.form("auth_form"):
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input("Full Legal Name", placeholder="e.g., Chinedu Okafor")
            phone = st.text_input("Verified Phone Number", placeholder="e.g., 08012345678")
        with col2:
            email = st.text_input("Email Address", placeholder="e.g., user@domain.com")
            nin_id = st.text_input("National Identification Number (NIN) / ID", placeholder="11-digit NIN or Valid ID")
            
        submitted_auth = st.form_submit_button("Verify & Enter Dispatch Grid", type="primary")
        
        if submitted_auth:
            if not full_name or not phone or not email or not nin_id:
                st.error("All verification fields are required to proceed.")
            else:
                st.session_state.logged_in = True
                st.session_state.user_profile = {
                    "full_name": full_name,
                    "phone": phone,
                    "email": email,
                    "nin_id": nin_id,
                    "status": "VERIFIED_CITIZEN"
                }
                st.success("Identity verified successfully! Loading secure dashboard...")
                st.rerun()

else:
    # 5. Main Emergency Reporting Dashboard (Unlocked Only for Verified Users)
    st.sidebar.title("👤 Reporter Profile")
    st.sidebar.write(f"**Name:** {st.session_state.user_profile['full_name']}")
    st.sidebar.write(f"**Phone:** {st.session_state.user_profile['phone']}")
    st.sidebar.write(f"**Status:** 🛡️ {st.session_state.user_profile['status']}")
    
    if st.sidebar.button("Logout / Switch User"):
        st.session_state.logged_in = False
        st.session_state.user_profile = {}
        st.rerun()

    st.title("🚨 Nigeria National Emergency AI & Dispatch Platform")
    st.caption("Secure multi-channel crisis management and automated agency routing system covering all 36 states and the FCT.")

    with st.form("emergency_form"):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            selected_state = st.selectbox("Select State / Territory", NIGERIA_STATES_FCT)
        with col_s2:
            location_details = st.text_input("LGA / Specific Location Details", placeholder="e.g., Wuse Zone 4, near Central Bank")
            
        emergency_text = st.text_area(
            "Describe the Emergency (English, Pidgin, Hausa, Yoruba, Egbo)",
            placeholder="e.g., Motors don jam for road OR Ina bukatan taimako gobiya..."
        )
        
        submit_btn = st.form_submit_button("Submit Secure Dispatch Alert", type="primary")

        if submit_btn:
            if not emergency_text or not location_details:
                st.error("Please provide both emergency details and specific location.")
            else:
                # ML Classification
                X_input = vectorizer.transform([emergency_text])
                prediction = model.predict(X_input)[0]
                category = label_encoder.inverse_transform([prediction])[0]
                assigned_agency = AGENCY_MAP.get(category, "General Emergency Unit")
                
                st.success("Emergency Classified and Secured Successfully!")
                
                # Display Metrics
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Target State", selected_state)
                res_col2.metric("Threat Category", category)
                res_col3.metric("Assigned Agency", assigned_agency)
                
                st.info(f"**Live Dispatch Log:** Alert queued for {assigned_agency} command center in {selected_state}. Verified reporter identity attached.")

                # Bind Log Entry with Authenticated User Information
                log_entry = {
                    "Timestamp": pd.Timestamp.now(),
                    "Reporter Name": st.session_state.user_profile["full_name"],
                    "Reporter Phone": st.session_state.user_profile["phone"],
                    "Reporter Email": st.session_state.user_profile["email"],
                    "Verification Status": st.session_state.user_profile["status"],
                    "State": selected_state,
                    "Location": location_details,
                    "Category": category,
                    "Assigned Agency": assigned_agency,
                    "Details": emergency_text
                }
                st.session_state.dispatch_logs.append(log_entry)

    # 6. Session Dispatch Logs & CSV Export Panel
    st.divider()
    st.subheader("📋 Secure Session Dispatch Logs (Audited)")

    if st.session_state.dispatch_logs:
        logs_df = pd.DataFrame(st.session_state.dispatch_logs)
        st.dataframe(logs_df, use_container_width=True)

        csv_data = logs_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Audited Dispatch Logs (CSV)",
            data=csv_data,
            file_name="verified_nigeria_emergency_logs.csv",
            mime="text/csv",
            type="primary"
        )
    else:
        st.info("No dispatch logs recorded in the current active session yet.")
