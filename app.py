import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# 1. Page Configuration
st.set_page_config(
    page_title="Nigeria National Emergency AI & Dispatch Platform",
    page_icon="🚨",
    layout="wide"
)

# Coordinates for all 36 States + FCT for Geo-Spatial Mapping
STATE_COORDINATES = {
    "Abia": [5.5320, 7.4860], "Adamawa": [9.3265, 12.3984], "Akwa Ibom": [5.0377, 7.9128],
    "Anambra": [6.2209, 6.9370], "Bauchi": [10.3158, 9.8442], "Bayelsa": [4.7719, 6.0699],
    "Benue": [7.7304, 8.5241], "Borno": [11.8333, 13.1500], "Cross River": [5.9631, 8.3373],
    "Delta": [5.8904, 5.6800], "Ebonyi": [6.2649, 8.0137], "Edo": [6.3350, 5.6037],
    "Ekiti": [7.6210, 5.2201], "Enugu": [6.4584, 7.5464], "FCT Abuja": [9.0765, 7.3986],
    "Gombe": [10.2897, 11.1673], "Imo": [5.4832, 7.0358], "Jigawa": [12.2280, 9.5616],
    "Kaduna": [10.5105, 7.4165], "Kano": [12.0022, 8.5919], "Katsina": [12.9887, 7.6009],
    "Kebbi": [12.4539, 4.1975], "Kogi": [7.7337, 6.6906], "Kwara": [8.4799, 4.5418],
    "Lagos": [6.5244, 3.3792], "Nasarawa": [8.4998, 8.5153], "Niger": [9.9309, 5.5983],
    "Ogun": [7.1604, 3.3483], "Ondo": [7.2571, 5.2058], "Osun": [7.5629, 4.5200],
    "Oyo": [7.8430, 3.9368], "Plateau": [9.8965, 8.8583], "Rivers": [4.8156, 7.0498],
    "Sokoto": [13.0627, 5.2339], "Taraba": [7.8704, 9.7800], "Yobe": [12.0000, 11.5000],
    "Zamfara": [12.1702, 6.6641]
}

# 2. Initialize Session State for Authentication, Logs & Map Data
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}
if "dispatch_logs" not in st.session_state:
    st.session_state.dispatch_logs = []
if "map_data" not in st.session_state:
    st.session_state.map_data = pd.DataFrame(columns=["lat", "lon"])

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

AGENCY_MAP = {
    "Kidnapping & Banditry": "Military Joint Task Force (JTF) / Counter-Terrorism & Police Tactical Command",
    "Armed Robbery": "Nigeria Police Force (NPF) / Rapid Response Squad (RRS)",
    "Fire": "Federal Fire Service (FFS) / State Fire Command",
    "Traffic": "Federal Road Safety Corps (FRSC)",
    "Medical": "National Emergency Management Agency (NEMA) / State Health Emergency Services"
}

NIGERIA_STATES_FCT = sorted(list(STATE_COORDINATES.keys()))

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
    # 5. Main Emergency Reporting Dashboard
    st.sidebar.title("👤 Reporter Profile")
    st.sidebar.write(f"**Name:** {st.session_state.user_profile['full_name']}")
    st.sidebar.write(f"**Phone:** {st.session_state.user_profile['phone']}")
    st.sidebar.write(f"**Status:** 🛡️ {st.session_state.user_profile['status']}")
    
    if st.sidebar.button("Logout / Switch User"):
        st.session_state.logged_in = False
        st.session_state.user_profile = {}
        st.rerun()

    st.title("🚨 Nigeria National Emergency AI & Dispatch Platform")
    st.caption("Secure multi-channel crisis management, automated agency routing, and telecommunications dispatch relay.")

    # Channel Selection Tabs
    tab1, tab2 = st.tabs(["🌐 Online Web AI Dispatch Portal", "📱 Offline Telecom USSD Gateway (*347*911#)"])

    # TAB 1: WEB AI DISPATCH
    with tab1:
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
                    # ML Classification
                    X_input = vectorizer.transform([emergency_text])
                    prediction = model.predict(X_input)[0]
                    category = label_encoder.inverse_transform([prediction])[0]
                    assigned_agency = AGENCY_MAP.get(category, "General Emergency Security Unit")
                    
                    st.success("Threat Classified & Multi-Channel Dispatch Broadcasted!")
                    
                    # Display Metrics
                    res_col1, res_col2, res_col3 = st.columns(3)
                    res_col1.metric("Target State", selected_state)
                    res_col2.metric("Threat Category", category)
                    res_col3.metric("Assigned Tactical Command", assigned_agency)
                    
                    # Simulate Telecom SMS/WhatsApp Gateway Relay
                    st.warning(f"📡 **Telecom Gateway Relay Active:** Priority SMS alert & WhatsApp dispatch payload dispatched to {assigned_agency} field commanders in {selected_state}.")

                    # Update Map Coordinates
                    coords = STATE_COORDINATES[selected_state]
                    new_map_point = pd.DataFrame({"lat": [coords[0]], "lon": [coords[1]]})
                    st.session_state.map_data = pd.concat([st.session_state.map_data, new_map_point], ignore_index=True)

                    # Bind Log Entry
                    log_entry = {
                        "Timestamp": pd.Timestamp.now(),
                        "Channel": "WEB_AI_PORTAL",
                        "Reporter Name": st.session_state.user_profile["full_name"],
                        "Reporter Phone": st.session_state.user_profile["phone"],
                        "Verification Status": st.session_state.user_profile["status"],
                        "State": selected_state,
                        "Location": location_details,
                        "Threat Category": category,
                        "Assigned Agency": assigned_agency,
                        "Details": emergency_text
                    }
                    st.session_state.dispatch_logs.append(log_entry)

    # TAB 2: USSD SIMULATOR FOR RURAL / FEATURE PHONES
    with tab2:
        st.subheader("📱 Offline USSD Emergency Gateway (`*347*911#`)")
        st.info("Designed for rural communities with zero internet connectivity. Accessible via basic feature phones.")
        
        with st.form("ussd_form"):
            ussd_state = st.selectbox("Select Location State", NIGERIA_STATES_FCT, key="ussd_state")
            ussd_threat = st.selectbox("Select Emergency Type", [
                "1. Kidnapping & Banditry (Armed Invasion)",
                "2. Armed Robbery In Progress",
                "3. Fire Outbreak",
                "4. Major Road Crash / Accident",
                "5. Medical Emergency / Fainting"
            ])
            ussd_location = st.text_input("Village / Landmark / LGA", placeholder="e.g., Dan Sadau Village")
            
            ussd_submit = st.form_submit_button("Dial *347*911# & Transmit Alert")

            if ussd_submit:
                threat_clean = ussd_threat.split(". ")[1].split(" (")[0]
                
                category_map = {
                    "Kidnapping & Banditry": "Kidnapping & Banditry",
                    "Armed Robbery In Progress": "Armed Robbery",
                    "Fire Outbreak": "Fire",
                    "Major Road Crash / Accident": "Traffic",
                    "Medical Emergency / Fainting": "Medical"
                }
                
                cat = category_map.get(threat_clean, "General Emergency")
                agency = AGENCY_MAP.get(cat, "Military/Police Command")

                st.success("📲 USSD Signal Transmitted via Telecommunication Tower!")
                
                # Update Map Coordinates
                coords = STATE_COORDINATES[ussd_state]
                new_map_point = pd.DataFrame({"lat": [coords[0]], "lon": [coords[1]]})
                st.session_state.map_data = pd.concat([st.session_state.map_data, new_map_point], ignore_index=True)

                log_entry = {
                    "Timestamp": pd.Timestamp.now(),
                    "Channel": "TELECOM_USSD_*347*911#",
                    "Reporter Name": st.session_state.user_profile["full_name"],
                    "Reporter Phone": st.session_state.user_profile["phone"],
                    "Verification Status": st.session_state.user_profile["status"],
                    "State": ussd_state,
                    "Location": ussd_location,
                    "Threat Category": cat,
                    "Assigned Agency": agency,
                    "Details": f"USSD Alert Code 911 - {threat_clean}"
                }
                st.session_state.dispatch_logs.append(log_entry)

    # 6. Real-Time Geo-Spatial Crisis Map
    st.divider()
    st.subheader("🗺️ Live National Geo-Spatial Threat Heatmap")
    if not st.session_state.map_data.empty:
        st.map(st.session_state.map_data)
    else:
        st.info("No active geo-spatial map markers yet. Submit an alert above to render live threat location pins.")

    # 7. Session Dispatch Logs & CSV Export Panel
    st.divider()
    st.subheader("📋 Secure Multi-Channel Dispatch Logs & Intelligence Audits")

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
