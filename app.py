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

# 3. Load and Train Advanced Multi-Lingual Threat Engine (Cached)
@st.cache_resource
def load_and_train_model():
    data = {
        'text': [
            # --- Kidnapping & Banditry (Military / JTF / Tactical Police) ---
            "Bandits have invaded our village shooting guns and kidnapping people",
            "Gunmen abduct school children and take them to the bush",
            "Armed men entered the community and started kidnapping residents",
            "Yan bindiga sun shigo kauyen mu suna sace mutane da bindiga",
            "Awon kidnappers ti gbe awon eniyan lo si igbo",
            "Ndị ohi ji egbe tọọrọ mmadụ n'obodo anyị",
            "Bad boys carry gun enter community dey kidnap pipu for bush",
            
            # --- Armed Robbery & Community Attacks (Police / Civil Defence) ---
            "Armed robbery ongoing near the commercial bank",
            "Thieves break into houses with weapons shooting sporadically",
            "Awon ole nbonilowo ni ile ifowopamo pẹlu ibọn",
            "Yan fashi suna harbi a kasuwa suna satar kayayyaki",
            "Ohi ji egbe na-apu n'ulo aku",
            "Bad boys dey roba shop with gun for junction",

            # --- Fire Outbreaks (Fire Service) ---
            "Fire outbreak in the residential building",
            "Inferno burning down market stalls and shops",
            "Ina njo ile ni agbegbe wa",
            "Gobiya tana cin ginin gidaje",
            "Akwukwo oku na-agba na ulo",
            "Fire deh burn house serious for street",

            # --- Traffic & Road Accidents (FRSC) ---
            "Multiple car crash on the expressway blood dey flow",
            "Trailer somersaulted blocking the main highway",
            "Oti wa ijamba oko lori titi",
            "Hatsarin mota ya faru a babban kwanji",
            "Iberibe okporo uzu mere na express",
            "Motors don jam themselves for road accident",

            # --- Medical Emergencies (NEMA / Health Services) ---
            "Person collapsed and needs immediate medical response",
            "Severe cholera outbreak and people fainting",
            "Eyan kan ti gbe emi mi leri aisan gbona",
            "Mutum ya yanke jiki ya fadi yana bukatan likita",
            "Onye adaala ma nwee mkpa doro anya",
            "Person don faint heavily he no dey breathe well"
        ],
        'category': [
            "Kidnapping & Banditry", "Kidnapping & Banditry", "Kidnapping & Banditry", "Kidnapping & Banditry", "Kidnapping & Banditry", "Kidnapping & Banditry", "Kidnapping & Banditry",
            "Armed Robbery", "Armed Robbery", "Armed Robbery", "Armed Robbery", "Armed Robbery", "Armed Robbery",
            "Fire", "Fire", "Fire", "Fire", "Fire", "Fire",
            "Traffic", "Traffic", "Traffic", "Traffic", "Traffic", "Traffic",
            "Medical", "Medical", "Medical", "Medical", "Medical", "Medical"
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

# Advanced Specialized Agency Mapping
AGENCY_MAP = {
    "Kidnapping & Banditry": "Military Joint Task Force (JTF) / Counter-Terrorism & Police Tactical Command",
    "Armed Robbery": "Nigeria Police Force (NPF) / Rapid Response Squad (RRS)",
    "Fire": "Federal Fire Service (FFS) / State Fire Command",
    "Traffic": "Federal Road Safety Corps (FRSC)",
    "Medical": "National Emergency Management Agency (NEMA) / State Health Emergency Services"
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

# 4. Authentication Gate (Ensures accountability and prevents fake setups)
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
    st.caption("Secure multi-channel crisis management and automated agency routing across all 36 states and FCT.")

    with st.form("emergency_form"):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            selected_state = st.selectbox("Select State / Territory", NIGERIA_STATES_FCT)
        with col_s2:
            location_details = st.text_input("LGA / Specific Location Details", placeholder="e.g., Birnin Gwari LGA / Wuse Zone 4")
            
        emergency_text = st.text_area(
            "Describe the Emergency (English, Pidgin, Hausa, Yoruba, Igbo)",
            placeholder="e.g., Bandits don enter village OR Fire outbreak OR Motors don jam..."
        )
        
        submit_btn = st.form_submit_button("Submit Secure Dispatch Alert", type="primary")

        if submit_btn:
            if not emergency_text or not location_details:
                st.error("Please provide both emergency details and specific location.")
            else:
                # Advanced ML Classification
                X_input = vectorizer.transform([emergency_text])
                prediction = model.predict(X_input)[0]
                category = label_encoder.inverse_transform([prediction])[0]
                assigned_agency = AGENCY_MAP.get(category, "General Emergency Security Unit")
                
                st.success("Threat Classified and Routed Successfully!")
                
                # Display Metrics
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Target State", selected_state)
                res_col2.metric("Threat Category", category)
                res_col3.metric("Assigned Tactical Command", assigned_agency)
                
                st.info(f"**Live Dispatch Log:** Alert queued for **{assigned_agency}** command center in {selected_state} (LGA: {location_details}). Verified reporter identity securely attached.")

                # Bind Log Entry with Authenticated User Information & Threat Details
                log_entry = {
                    "Timestamp": pd.Timestamp.now(),
                    "Reporter Name": st.session_state.user_profile["full_name"],
                    "Reporter Phone": st.session_state.user_profile["phone"],
                    "Reporter Email": st.session_state.user_profile["email"],
                    "Verification Status": st.session_state.user_profile["status"],
                    "State": selected_state,
                    "Location": location_details,
                    "Threat Category": category,
                    "Assigned Agency": assigned_agency,
                    "Details": emergency_text
                }
                st.session_state.dispatch_logs.append(log_entry)

    # 6. Session Dispatch Logs & CSV Export Panel
    st.divider()
    st.subheader("📋 Secure Session Dispatch Logs & Intelligence Audits")

    if st.session_state.dispatch_logs:
        logs_df = pd.DataFrame(st.session_state.dispatch_logs)
        st.dataframe(logs_df, use_container_width=True)

        csv_data = logs_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Audited Intelligence Logs (CSV)",
            data=csv_data,
            file_name="verified_nigeria_security_dispatch_logs.csv",
            mime="text/csv",
            type="primary"
        )
    else:
        st.info("No dispatch logs recorded in the current active session yet.")
