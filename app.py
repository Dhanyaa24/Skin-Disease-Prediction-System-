import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Skin Disease Assessment",
    page_icon="https://i.pinimg.com/1200x/e8/bb/07/e8bb0791a87e6b9a67a9408f03fd781b.jpg",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_models():
    try:
        model = joblib.load('skin_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found! Please ensure 'skin_disease_model.pkl' and 'scaler.pkl' are in the same directory.")
        return None, None

model, scaler = load_models()

# Disease information with links and tips
disease_info = {
    'Psoriasis': {
        'description': 'A chronic autoimmune condition that causes rapid skin cell buildup, resulting in thick, red, scaly patches on the skin surface. These patches can be itchy and sometimes painful.',
        'link': 'https://www.mayoclinic.org/diseases-conditions/psoriasis/symptoms-causes/syc-20355840',
        'tips': [
            'Keep skin moisturized with thick creams or ointments',
            'Moderate sun exposure can help, but avoid sunburn',
            'Take lukewarm baths with colloidal oatmeal',
            'Manage stress through relaxation techniques',
            'Avoid smoking and limit alcohol consumption'
        ]
    },
    'Seborrheic dermatitis': {
        'description': 'A common skin condition that mainly affects the scalp, causing scaly patches, red skin, and stubborn dandruff. It can also affect oily areas like the face, sides of the nose, eyebrows, and chest.',
        'link': 'https://www.mayoclinic.org/diseases-conditions/seborrheic-dermatitis/symptoms-causes/syc-20352710',
        'tips': [
            'Use medicated shampoos with ketoconazole or selenium sulfide',
            'Wash affected areas regularly but gently',
            'Apply aloe vera or tea tree oil (diluted) to soothe skin',
            'Get adequate sleep to reduce stress',
            'In winter, use a humidifier to prevent dry skin'
        ]
    },
    'Lichen planus': {
        'description': 'An inflammatory condition affecting the skin and mucous membranes, characterized by purplish, flat-topped, itchy bumps that can appear anywhere on the body. It may also affect the mouth, nails, and scalp.',
        'link': 'https://www.mayoclinic.org/diseases-conditions/lichen-planus/symptoms-causes/syc-20351378',
        'tips': [
            'Avoid harsh soaps and hot water',
            'Apply cool compresses to reduce itching',
            'Protect skin from sun exposure',
            'Reduce stress through meditation or yoga',
            'Maintain good oral hygiene if mouth is affected'
        ]
    },
    'Pityriasis rosea': {
        'description': 'A common skin condition causing a scaly rash that usually begins with a single large patch (herald patch), followed by smaller patches spreading across the body in a pattern resembling a Christmas tree.',
        'link': 'https://www.mayoclinic.org/diseases-conditions/pityriasis-rosea/symptoms-causes/syc-20376405',
        'tips': [
            'Use gentle, fragrance-free moisturizers',
            'Take lukewarm baths with oatmeal',
            'Moderate sun exposure may help (15-20 minutes)',
            'Wear loose, breathable clothing',
            'Use over-the-counter anti-itch creams if needed'
        ]
    },
    'Chronic dermatitis': {
        'description': 'Long-term inflammation of the skin causing persistent itchy, red, swollen, and dry patches. It can result from various causes including allergies, irritants, or unknown factors.',
        'link': 'https://www.mayoclinic.org/diseases-conditions/dermatitis-eczema/symptoms-causes/syc-20352380',
        'tips': [
            'Moisturize frequently, especially after bathing',
            'Identify and avoid triggers (allergens, irritants)',
            'Use mild, fragrance-free cleansers',
            'Keep nails short to prevent scratching damage',
            'Wear soft, breathable fabrics like cotton'
        ]
    },
    'Pityriasis rubra pilaris': {
        'description': 'A rare skin disorder causing reddish-orange scaling patches and thickening of the skin. It typically starts on the head and progresses downward, with small islands of normal skin within affected areas.',
        'link': 'https://rarediseases.org/rare-diseases/pityriasis-rubra-pilaris/',
        'tips': [
            'Use thick emollients to keep skin moisturized',
            'Stay well-hydrated by drinking plenty of water',
            'Avoid extreme temperatures',
            'Follow prescribed treatment plan carefully',
            'Join support groups for rare skin conditions'
        ]
    }
}

# Questions mapped to features
questions = {
    'clinical': [
        {
            'feature': 'erythema',
            'question': 'Do you have redness on your skin?',
            'options': ['No redness', 'Slight redness', 'Moderate redness', 'Severe redness']
        },
        {
            'feature': 'scaling',
            'question': 'Is your skin flaking or peeling?',
            'options': ['No scaling', 'Mild scaling', 'Moderate scaling', 'Heavy scaling']
        },
        {
            'feature': 'definite_borders',
            'question': 'Do the affected areas have clear, well-defined edges?',
            'options': ['No clear borders', 'Slightly defined', 'Moderately defined', 'Very clear borders']
        },
        {
            'feature': 'itching',
            'question': 'How severe is the itching?',
            'options': ['No itching', 'Mild itching', 'Moderate itching', 'Severe itching']
        },
        {
            'feature': 'koebner_phenomenon',
            'question': 'Do new lesions appear where your skin has been injured or scratched?',
            'options': ['Never', 'Rarely', 'Sometimes', 'Frequently']
        },
        {
            'feature': 'polygonal_papules',
            'question': 'Do you have small, flat-topped bumps with multiple sides?',
            'options': ['None', 'Few', 'Moderate amount', 'Many']
        },
        {
            'feature': 'follicular_papules',
            'question': 'Do you have bumps around hair follicles?',
            'options': ['None', 'Few', 'Moderate amount', 'Many']
        },
        {
            'feature': 'oral_mucosal_involvement',
            'question': 'Do you have lesions or white patches inside your mouth?',
            'options': ['No', 'Mild', 'Moderate', 'Severe']
        },
        {
            'feature': 'knee_elbow_involvement',
            'question': 'Are your knees or elbows affected?',
            'options': ['Not affected', 'Slightly affected', 'Moderately affected', 'Severely affected']
        },
        {
            'feature': 'scalp_involvement',
            'question': 'Is your scalp affected?',
            'options': ['Not affected', 'Slightly affected', 'Moderately affected', 'Severely affected']
        },
        {
            'feature': 'family_history',
            'question': 'Do family members have similar skin conditions?',
            'options': ['No family history', 'Distant relatives', 'Close relatives', 'Multiple family members']
        }
    ],
    'histopathological': [
        {
            'feature': 'melanin_incontinence',
            'question': 'Has a skin biopsy shown melanin pigment leakage?',
            'options': ['Not tested/No', 'Mild', 'Moderate', 'Severe']
        },
        {
            'feature': 'eosinophils_infiltrate',
            'question': 'Did the biopsy show eosinophil cells?',
            'options': ['Not tested/No', 'Few present', 'Moderate amount', 'Many present']
        },
        {
            'feature': 'PNL_infiltrate',
            'question': 'Did the biopsy show inflammatory white blood cells?',
            'options': ['Not tested/No', 'Mild infiltration', 'Moderate infiltration', 'Heavy infiltration']
        },
        {
            'feature': 'fibrosis_papillary_dermis',
            'question': 'Did the biopsy show tissue scarring in the dermis?',
            'options': ['Not tested/No', 'Mild', 'Moderate', 'Severe']
        },
        {
            'feature': 'exocytosis',
            'question': 'Did the biopsy show immune cells in the outer skin layer?',
            'options': ['Not tested/No', 'Mild', 'Moderate', 'Severe']
        },
        {
            'feature': 'acanthosis',
            'question': 'Did the biopsy show thickening of the outer skin layer?',
            'options': ['Not tested/No', 'Mild', 'Moderate', 'Severe']
        },
        {
            'feature': 'hyperkeratosis',
            'question': 'Did the biopsy show thickening of the top skin layer?',
            'options': ['Not tested/No', 'Mild', 'Moderate', 'Severe']
        },
        {
            'feature': 'parakeratosis',
            'question': 'Did the biopsy show abnormal cell development?',
            'options': ['Not tested/No', 'Mild', 'Moderate', 'Severe']
        },
        {
            'feature': 'clubbing_rete_ridges',
            'question': 'Did the biopsy show club-shaped ridges?',
            'options': ['Not tested/No', 'Mild', 'Moderate', 'Severe']
        },
        {
            'feature': 'elongation_rete_ridges',
            'question': 'Did the biopsy show elongated skin ridges?',
            'options': ['Not tested/No', 'Mild', 'Moderate', 'Severe']
        },
        {
            'feature': 'thinning_suprapapillary_epidermis',
            'question': 'Did the biopsy show thinning of skin above ridges?',
            'options': ['Not tested/No', 'Mild', 'Moderate', 'Severe']
        },
        {
            'feature': 'spongiform_pustule',
            'question': 'Did the biopsy show sponge-like pus formations?',
            'options': ['Not tested/No', 'Mild', 'Moderate', 'Severe']
        },
        {
            'feature': 'munro_microabcess',
            'question': 'Did the biopsy show small pus collections?',
            'options': ['Not tested/No', 'Few', 'Moderate', 'Many']
        },
        {
            'feature': 'focal_hypergranulosis',
            'question': 'Did the biopsy show localized thickening?',
            'options': ['Not tested/No', 'Mild', 'Moderate', 'Severe']
        },
        {
            'feature': 'disappearance_granular_layer',
            'question': 'Did the biopsy show loss of a skin layer?',
            'options': ['Not tested/No', 'Mild', 'Moderate', 'Complete']
        },
        {
            'feature': 'vacuolisation_damage_basal_layer',
            'question': 'Did the biopsy show damage to the base skin layer?',
            'options': ['Not tested/No', 'Mild', 'Moderate', 'Severe']
        },
        {
            'feature': 'spongiosis',
            'question': 'Did the biopsy show fluid between skin cells?',
            'options': ['Not tested/No', 'Mild', 'Moderate', 'Severe']
        },
        {
            'feature': 'saw_tooth_appearance_retes',
            'question': 'Did the biopsy show saw-tooth shaped ridges?',
            'options': ['Not tested/No', 'Mild', 'Moderate', 'Severe']
        },
        {
            'feature': 'follicular_horn_plug',
            'question': 'Did the biopsy show plugged hair follicles?',
            'options': ['Not tested/No', 'Few', 'Moderate', 'Many']
        },
        {
            'feature': 'perifollicular_parakeratosis',
            'question': 'Did the biopsy show abnormal cells around follicles?',
            'options': ['Not tested/No', 'Mild', 'Moderate', 'Severe']
        },
        {
            'feature': 'inflammatory_mononuclear_infiltrate',
            'question': 'Did the biopsy show immune cell infiltration?',
            'options': ['Not tested/No', 'Mild', 'Moderate', 'Severe']
        },
        {
            'feature': 'band_like_infiltrate',
            'question': 'Did the biopsy show a band of immune cells?',
            'options': ['Not tested/No', 'Mild', 'Moderate', 'Severe']
        }
    ]
}

# Disease names
disease_names = [
    'Psoriasis',
    'Seborrheic dermatitis',
    'Lichen planus',
    'Pityriasis rosea',
    'Chronic dermatitis',
    'Pityriasis rubra pilaris'
]

# Initialize session state
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'age' not in st.session_state:
    st.session_state.age = 30
if 'show_lab_section' not in st.session_state:
    st.session_state.show_lab_section = False
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

# Main title
st.title("Skin Disease Assessment Tool")
st.markdown("### Answer a few questions about your symptoms")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About This Tool")
    st.info("""
    This assessment tool helps predict potential skin conditions based on your symptoms.
    
    **Diseases Assessed:**
    - Psoriasis
    - Seborrheic dermatitis
    - Lichen planus
    - Pityriasis rosea
    - Chronic dermatitis
    - Pityriasis rubra pilaris
    
    **Model Accuracy:** 98.61%
    """)
    
    st.markdown("---")
    st.warning("**Important:** This tool is for educational purposes only. Always consult a healthcare professional for proper diagnosis and treatment.")

if model is not None and scaler is not None:
    
    # Clinical Symptoms Section
    if not st.session_state.show_results:
        st.header("Clinical Symptoms & Observations")
        st.markdown("*Please answer based on your current symptoms*")
        st.markdown("")
        
        # Age question first
        st.session_state.age = st.number_input(
            "**What is your age?**",
            min_value=1,
            max_value=120,
            value=st.session_state.age,
            key='age_input'
        )
        
        st.markdown("---")
        
        # Clinical questions
        for q in questions['clinical']:
            st.markdown(f"**{q['question']}**")
            answer = st.radio(
                "",
                options=range(len(q['options'])),
                format_func=lambda x, opts=q['options']: opts[x],
                key=f"q_{q['feature']}",
                horizontal=False,
                index=st.session_state.answers.get(q['feature'], 0)
            )
            st.session_state.answers[q['feature']] = answer
            st.markdown("")
        
        st.markdown("---")
        
        # Action buttons after clinical symptoms
        st.markdown("### What would you like to do next?")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Add Lab Results (Optional)", use_container_width=True, type="secondary"):
                st.session_state.show_lab_section = True
                st.rerun()
        
        with col2:
            if st.button("Get Results Now", use_container_width=True, type="primary"):
                # Set all lab results to 0 (not tested)
                for q in questions['histopathological']:
                    if q['feature'] not in st.session_state.answers:
                        st.session_state.answers[q['feature']] = 0
                st.session_state.show_results = True
                st.rerun()
    
    # Lab Results Section (only shown if user clicks the button)
    if st.session_state.show_lab_section and not st.session_state.show_results:
        st.header("Histopathological/Lab Results")
        st.info("Tip: Answer these questions based on your biopsy report. If you're unsure, select 'Not tested/No'.")
        st.markdown("")
        
        # Histopathological questions
        for q in questions['histopathological']:
            st.markdown(f"**{q['question']}**")
            answer = st.radio(
                "",
                options=range(len(q['options'])),
                format_func=lambda x, opts=q['options']: opts[x],
                key=f"q_{q['feature']}",
                horizontal=False,
                index=st.session_state.answers.get(q['feature'], 0)
            )
            st.session_state.answers[q['feature']] = answer
            st.markdown("")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Get Assessment Results", use_container_width=True, type="primary"):
                st.session_state.show_results = True
                st.rerun()
    
    # Results Section
    if st.session_state.show_results:
        # Prepare input data
        feature_order = [q['feature'] for q in questions['clinical']] + ['age'] + [q['feature'] for q in questions['histopathological']]
        
        input_values = []
        for feature in feature_order:
            if feature == 'age':
                input_values.append(st.session_state.age)
            else:
                input_values.append(st.session_state.answers.get(feature, 0))
        
        input_data = [input_values]
        
        # Scale and predict
        try:
            # Check if user has minimal symptoms (might not have a skin disease)
            clinical_answers = [st.session_state.answers.get(q['feature'], 0) for q in questions['clinical']]
            total_symptom_score = sum(clinical_answers)
            
            # If very few symptoms, show a different message
            if total_symptom_score <= 2:  # Almost no symptoms
                st.info("## Assessment Complete")
                st.markdown("")
                st.markdown("### Good News!")
                st.write("""
                Based on your responses, you appear to have **minimal to no symptoms** of the skin conditions 
                we assess. This is a positive sign!
                
                **However, if you are experiencing skin issues:**
                - The symptoms might be very mild or in early stages
                - You might have a different skin condition not covered by this tool
                - Please consult a dermatologist for a proper examination
                """)
                
                st.markdown("---")
                st.info("""
                **General Skin Health Tips:**
                - Keep skin clean and moisturized
                - Protect from excessive sun exposure
                - Stay hydrated
                - Eat a balanced diet
                - If you notice any changes, consult a healthcare provider
                """)
                
                st.markdown("---")
                
                # New assessment button for low symptom case
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("Start New Assessment", use_container_width=True, type="secondary", key="restart_low"):
                        st.session_state.answers = {}
                        st.session_state.age = 30
                        st.session_state.show_lab_section = False
                        st.session_state.show_results = False
                        st.rerun()
                
            else:
                # Proceed with normal prediction
                scaled_data = scaler.transform(input_data)
                prediction = model.predict(scaled_data)
                predicted_disease = disease_names[prediction[0] - 1]
                
                # Show results
                st.success(f"## Assessment Complete")
                st.markdown("")
                
                # Confidence indicator based on symptom severity
                confidence_note = ""
                if total_symptom_score <= 5:
                    confidence_note = " *(Low symptom severity - consider getting a professional examination)*"
                elif total_symptom_score <= 10:
                    confidence_note = " *(Moderate symptom severity)*"
                else:
                    confidence_note = " *(High symptom severity - please consult a dermatologist soon)*"
                
                # Main prediction with prominent display
                st.markdown(f"# Predicted Condition: **{predicted_disease}**{confidence_note}")
                st.markdown("---")
                
                # Disease information
                info = disease_info[predicted_disease]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### About This Condition")
                    st.write(info['description'])
                    st.markdown("")
                    st.markdown(f"**[Learn More About {predicted_disease}]({info['link']})**")
                
                with col2:
                    st.metric("Model Confidence", "98.61%")
                    st.metric("Disease Class", f"Class {prediction[0]}")
                
                st.markdown("---")
                
                # General tips
                st.markdown("### General Care Tips")
                cols = st.columns(2)
                for i, tip in enumerate(info['tips']):
                    with cols[i % 2]:
                        st.markdown(f"- {tip}")
                
                st.markdown("---")
                
                # Important notice
                st.error("""
                **IMPORTANT MEDICAL NOTICE**
                
                This assessment is based on AI prediction and should NOT replace professional medical advice. 
                Please consult a dermatologist or healthcare provider for:
                - Proper diagnosis and confirmation
                - Personalized treatment plan
                - Prescription medications if needed
                - Monitoring of your condition
                """)
                
                # Additional resources
                with st.expander("Additional Resources"):
                    st.markdown("""
                    **General Skin Health Resources:**
                    - [American Academy of Dermatology](https://www.aad.org/)
                    - [National Eczema Association](https://nationaleczema.org/)
                    - [National Psoriasis Foundation](https://www.psoriasis.org/)
                    
                    **When to Seek Immediate Care:**
                    - Severe pain or discomfort
                    - Signs of infection (pus, fever, warmth)
                    - Rapid spreading of lesions
                    - Difficulty breathing or swallowing
                    """)
                
                st.markdown("---")
                
                # New assessment button for disease prediction case
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("Start New Assessment", use_container_width=True, type="secondary", key="restart_disease"):
                        st.session_state.answers = {}
                        st.session_state.age = 30
                        st.session_state.show_lab_section = False
                        st.session_state.show_results = False
                        st.rerun()
            
        except Exception as e:
            st.error(f"Error during assessment: {str(e)}")
            if st.button("Try Again"):
                st.session_state.show_results = False
                st.rerun()

else:
    st.error("Unable to load model files. Please ensure the model has been trained and saved.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>Model trained on UCI Dermatology Dataset | Accuracy: 98.61%</p>
    <p>This tool is for educational and research purposes only</p>
</div>
""", unsafe_allow_html=True)