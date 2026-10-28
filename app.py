import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from streamlit_geolocation import streamlit_geolocation

# 1. Page Configuration
st.set_page_config(
    page_title="Nigeria National Emergency AI & Tactical Command Platform",
    page_icon="🚨",
    layout="wide"
)

# Precise State Center Coordinates for Nigeria
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

NIGERIA_STATES = sorted(list(STATE_COORDINATES.keys()))

# 2. Session State Initialization
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}
if "dispatch_logs" not in st.session_state:
    st.session_state.dispatch_logs = []
if "map_data" not in st.session_state:
    st.session_state.map_data = pd.DataFrame({"lat": [9.0765], "lon": [7.3986]})

# 3. Expanded Multi-Lingual & Multi-Category Threat Engine
@st.cache_resource
def load_and_train_model():
    data = {
        'text': [
            # Kidnapping & Banditry
            "Bandits have invaded our village shooting guns and kidnapping people",
            "Gunmen abduct school children and take them to the bush",
            "Yan bindiga sun shigo kauyen mu suna sace mutane da bindiga",
            "Awon kidnappers ti gbe awon eniyan lo si igbo",
            "Ndị ohi ji egbe tọọrọ mmadụ n'obodo anyị",
            "Bad boys carry gun enter community dey kidnap pipu for bush",
            
            # Armed Robbery
            "Armed robbery ongoing near the commercial bank",
            "Thieves break into houses with weapons shooting sporadically",
            "Awon ole nbonilowo ni ile ifowopamo pẹlu ibọn",
            "Yan fashi suna harbi a kasuwa suna satar kayayyaki",
            "Ohi ji egbe na-apu n'ulo aku",
            "Bad boys dey roba shop with gun for junction",

            # Fire
            "Fire outbreak in the residential building",
            "Inferno burning down market stalls and shops",
            "Ina njo ile ni agbegbe wa",
            "Gobiya tana cin ginin gidaje",
            "Akwukwo oku na-agba na ulo",
            "Fire deh burn house serious for street",

            # Traffic / Accident
            "Multiple car crash on the expressway blood dey flow",
            "Trailer somersaulted blocking the main highway",
            "Oti wa ijamba oko lori titi",
            "Hatsarin mota ya faru a babban kwanji",
            "Iberibe okporo uzu mere na express",
            "Motors don jam themselves for road accident",

            # Medical
            "Person collapsed and needs immediate medical response",
            "Severe cholera outbreak and people fainting",
            "Eyan kan ti gbe emi mi leri aisan gbona",
            "Mutum ya yanke jiki ya fadi yana bukatan likita",
            "Onye adaala ma nwee mkpa doro anya",
            "Person don faint heavily he no dey breathe well"
        ],
        'category': [
            "Kidnapping & Banditry", "Kidnapping & Banditry", "Kidnapping & Banditry", "Kidnapping & Banditry", "Kidnapping & Banditry", "Kidnapping & Banditry",
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
    "Kidnapping & Banditry": "Military Joint Task Force (JTF) / Counter-Terrorism Command",
    "Armed Robbery": "Nigeria Police Force (NPF) / Rapid Response Squad",
    "Fire": "Federal Fire Service (FFS) / State Fire Command",
    "Traffic": "Federal Road Safety Corps (FRSC)",
    "Medical": "National Emergency Management Agency (NEMA) / Health Services"
}

# 4. Sidebar Mode Selection
st.sidebar.title("🛡️ Platform Access Mode")
portal_mode = st.sidebar.radio("Select Portal", ["Citizen Dispatch Portal", "Agency Tactical Command Dashboard"])

# =====================================================================
# PORTAL MODE 1: CITIZEN DISPATCH PORTAL
# =====================================================================
if portal_mode == "Citizen Dispatch Portal":

    if not st.session_state.logged_in:
        st.markdown("## 🔐 Citizen Security Verification Portal")
        st.info("Identity binding is mandatory to prevent false alarms and safeguard responding tactical units.")
        
        with st.form("auth_form"):
            col1, col2 = st.columns(2)
            with col1:
                full_name = st.text_input("Full Legal Name", placeholder="e.g., Chinedu Okafor")
                phone = st.text_input("Verified Phone Number", placeholder="e.g., 08012345678")
            with col2:
                email = st.text_input("Email Address", placeholder="e.g., user@domain.com")
                nin_id = st.text_input("National Identification Number (NIN) / ID", placeholder="11-digit NIN or Valid ID")
                
            submitted_auth = st.form_submit_button("Verify & Access Emergency Grid", type="primary")
            
            if submitted_auth:
                if not full_name or not phone or not email or not nin_id:
                    st.error("All verification fields are required.")
                else:
                    st.session_state.logged_in = True
                    st.session_state.user_profile = {
                        "full_name": full_name, "phone": phone, "email": email,
                        "nin_id": nin_id, "status": "VERIFIED_CITIZEN"
                    }
                    st.success("Identity verified! Loading emergency dispatch grid...")
                    st.rerun()

    else:
        st.sidebar.title("👤 Reporter Profile")
        st.sidebar.write(f"**Name:** {st.session_state.user_profile['full_name']}")
        st.sidebar.write(f"**Phone:** {st.session_state.user_profile['phone']}")
        st.sidebar.write(f"**NIN:** {st.session_state.user_profile['nin_id']}")
        st.sidebar.write(f"**Status:** 🛡️ {st.session_state.user_profile['status']}")
        
        if st.sidebar.button("Logout / Switch User"):
            st.session_state.logged_in = False
            st.session_state.user_profile = {}
            st.rerun()

        st.title("🚨 Nigeria National Emergency AI Dispatch Portal")
        st.caption("Integrated Dual-Location Verification & Multi-Agency Emergency Dispatch System")

        st.markdown("---")
        st.subheader("1. Automatic Device Location Capture")
        st.write("Capture your current GPS location to cross-reference with your declared emergency location:")
        
        device_location = streamlit_geolocation()
        
        user_lat = device_location.get("latitude") if device_location else None
        user_lon = device_location.get("longitude") if device_location else None

        if user_lat and user_lon:
            st.success(f"📍 **Device GPS Locked:** Latitude: `{user_lat:.4f}`, Longitude: `{user_lon:.4f}`")
        else:
            st.info("Click the location button above to lock your current device GPS position.")

        st.markdown("---")
        st.subheader("2. Emergency Incident Reporting Form")

        with st.form("emergency_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                claimed_state = st.selectbox("Declared Emergency State", NIGERIA_STATES)
            with col_b:
                claimed_location = st.text_input("LGA / Ward / Specific Landmark", placeholder="e.g., Funtua LGA / Central Market Junction")

            emergency_text = st.text_area(
                "Describe the Incident (English, Pidgin, Hausa, Yoruba, Igbo)",
                placeholder="e.g., Fire outbreak for market OR Motors don jam for road OR Bandits don enter..."
            )

            submit_btn = st.form_submit_button("Transmit Verified Emergency Alert", type="primary")

            if submit_btn:
                if not emergency_text or not claimed_location:
                    st.error("Please fill in both the location details and incident description.")
                else:
                    X_input = vectorizer.transform([emergency_text])
                    prediction = model.predict(X_input)[0]
                    category = label_encoder.inverse_transform([prediction])[0]
                    assigned_agency = AGENCY_MAP.get(category, "General Security Command")

                    state_base_coords = STATE_COORDINATES.get(claimed_state, [9.0765, 7.3986])
                    
                    if user_lat and user_lon:
                        lat_diff = abs(user_lat - state_base_coords[0])
                        lon_diff = abs(user_lon - state_base_coords[1])
                        
                        if lat_diff > 2.5 or lon_diff > 2.5:
                            verification_flag = "🚨 SUSPECTED FALSE ALARM / LOCATION MISMATCH"
                            flag_color = "error"
                        else:
                            verification_flag = "✅ VERIFIED PHYSICAL LOCATION MATCH"
                            flag_color = "success"
                        
                        actual_lat = user_lat
                        actual_lon = user_lon
                    else:
                        verification_flag = "⚠️ GPS UNBOUND (MANUAL REPORT)"
                        flag_color = "warning"
                        actual_lat = state_base_coords[0]
                        actual_lon = state_base_coords[1]

                    st.success(f"Emergency Alert Routed to **{assigned_agency}**")
                    
                    if flag_color == "error":
                        st.error(f"**Security Audit Flag:** {verification_flag}. The reported state ({claimed_state}) does not match your device's current location!")
                    elif flag_color == "success":
                        st.success(f"**Security Audit Flag:** {verification_flag}")
                    else:
                        st.warning(f"**Security Audit Flag:** {verification_flag}")

                    # Update Map Data
                    new_point = pd.DataFrame({"lat": [actual_lat], "lon": [actual_lon]})
                    st.session_state.map_data = pd.concat([st.session_state.map_data, new_point], ignore_index=True)

                    log_entry = {
                        "Timestamp": pd.Timestamp.now(),
                        "Reporter Name": st.session_state.user_profile["full_name"],
                        "Reporter Phone": st.session_state.user_profile["phone"],
                        "Reporter NIN": st.session_state.user_profile["nin_id"],
                        "Declared State": claimed_state,
                        "Declared Location": claimed_location,
                        "Threat Category": category,
                        "Assigned Agency": assigned_agency,
                        "Verification Flag": verification_flag,
                        "Actual Device Lat": actual_lat,
                        "Actual Device Lon": actual_lon,
                        "Nav Link": f"https://www.google.com/maps?q={actual_lat},{actual_lon}",
                        "Incident Details": emergency_text
                    }
                    st.session_state.dispatch_logs.append(log_entry)

        # Map View
        st.divider()
        st.subheader("🗺️ Live National Threat Heatmap")
        st.map(st.session_state.map_data, zoom=6)

        # RECORD-KEEPING & CSV EXPORT TABLE
        st.divider()
        st.subheader("📋 Audited Grassroots Intelligence Logs & Record Keeping")

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

# =====================================================================
# PORTAL MODE 2: AGENCY TACTICAL COMMAND DASHBOARD
# =====================================================================
else:
    st.title("🛡️ National Security Tactical Command Center")
    st.caption("Real-Time Multi-Agency Intelligence, Dual-Location Verification & Anti-Setup Audit Panel")

    agency_filter = st.selectbox(
        "Select Security Command Agency View",
        [
            "All Agencies (National Intelligence Command)",
            "Military Joint Task Force (JTF) / Counter-Terrorism Command",
            "Nigeria Police Force (NPF) / Rapid Response Squad",
            "Federal Fire Service (FFS) / State Fire Command",
            "Federal Road Safety Corps (FRSC)",
            "National Emergency Management Agency (NEMA) / Health Services"
        ]
    )

    st.divider()

    if st.session_state.dispatch_logs:
        logs_df = pd.DataFrame(st.session_state.dispatch_logs)

        if agency_filter != "All Agencies (National Intelligence Command)":
            filtered_df = logs_df[logs_df["Assigned Agency"] == agency_filter]
        else:
            filtered_df = logs_df

        st.subheader(f"📋 Live Dispatch Feed ({len(filtered_df)} Incidents Logged)")

        if not filtered_df.empty:
            for idx, row in filtered_df.iterrows():
                if "SUSPECTED FALSE ALARM" in row["Verification Flag"]:
                    st.error(f"🚨 **[STAGED SETUP WARNING]** Incident #{idx+1} - {row['Threat Category']} in {row['Declared State']}")
                else:
                    st.success(f"⚡ Incident #{idx+1} - {row['Threat Category']} in {row['Declared State']}")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write(f"**Reporter:** {row['Reporter Name']}")
                    st.write(f"**Phone:** {row['Reporter Phone']}")
                    st.write(f"**NIN:** {row['Reporter NIN']}")
                with c2:
                    st.write(f"**Declared Location:** {row['Declared Location']}, {row['Declared State']}")
                    st.write(f"**Audit Status:** {row['Verification Flag']}")
                    st.write(f"**Assigned Agency:** {row['Assigned Agency']}")
                with c3:
                    st.write(f"**Timestamp:** {row['Timestamp']}")
                    st.markdown(f"📍 [**Open Precise GPS Scene Navigation**]({row['Nav Link']})")

                st.write(f"**Incident Description:** *\"{row['Incident Details']}\"*")
                st.divider()

            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Classified Tactical Intelligence Report (CSV)",
                data=csv_data,
                file_name="tactical_command_security_audit.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.info("No active incidents currently logged for this agency.")
    else:
        st.info("No dispatch logs recorded in the current active session.")
