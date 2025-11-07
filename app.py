import streamlit as st
import pandas as pd
import numpy as np

st.title(\"🚀 Test App - Dependencies Working\")
st.write(f\"Streamlit version: {st.__version__}\")
st.write(f\"Pandas version: {pd.__version__}\") 
st.write(f\"Numpy version: {np.__version__}\")

st.success(\"✅ If you see this, basic dependencies are working!\")

# Simple functionality test
if st.button(\"Test Calculation\"):
    result = np.mean([1, 2, 3, 4, 5])
    st.write(f\"Numpy calculation: {result}\")