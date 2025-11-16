import streamlit as st
from PIL import Image

# Set up the main landing page
st.set_page_config(page_title="Healthcare Risk Intelligence", layout="wide")

# Centered logo at the very top
try:
    logo = Image.open("C:/Users/dkeya/Documents/projects/insurance/logo.png")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo, width=250)  # Centered and resized logo
except:
    st.warning("Logo not found")

# Title
st.title("🏥 Healthcare Risk Intelligence Platform")

# Welcome message
st.markdown("""
Welcome to the **Healthcare Risk Intelligence Platform** — your unified environment for managing and analyzing both **insured** and **fund-managed** healthcare schemes.

Choose a solution to begin:
""")

# Solution navigation
option = st.radio(
    "🔍 Select Your Health Financing Model",
    ["🧾 Fund Management", "📑 Insured Scheme (Loss Ratio Predictor)"],
    horizontal=True
)

# Dynamic navigation message
if option == "📑 Insured Scheme (Loss Ratio Predictor)":
    st.success("You selected the Insured Scheme Solution. Please navigate to the **Insured App Page** to continue.")
    st.markdown("👉 [Go to Insured Scheme Module](insured_scheme_app.py)")
elif option == "🧾 Fund Management":
    st.success("You selected the Fund Management Solution. Please navigate to the **Fund Management App Page** to continue.")
    st.markdown("👉 [Go to Fund Management Module](fund_management_app.py)")

# Footer
st.markdown("---")
st.markdown("**Version:** 1.0.0 | © 2025 Virtual Analytics | All Rights Reserved")


