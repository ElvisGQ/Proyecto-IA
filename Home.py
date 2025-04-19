import streamlit as st
import importlib

import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

image_path = "images/carrito2.png"  # Adjust this to your image's path
img_base64 = get_base64_image(image_path)

st.markdown(f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_base64}" alt="Logo" width="120" style="margin-right: 25px;"/>
    </div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>PARKUCC</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Aplicación web para ver disponibilidad del estacionamiento de la universidad UCC.</h3>", unsafe_allow_html=True)

st.markdown("""
    <style>
        .card-container {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 30px;
            flex-wrap: wrap;
        }
        .card {
            background-color: #f8f9fa;
            padding: 20px;
            width: 300px;
            border-radius: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
            transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease;
            cursor: pointer;
        }
        .card:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 16px rgba(0,0,0,0.2);
        }
        .card h3 {
            margin-bottom: 10px;
            color: #333;
        }
        .card p {
            color: #666;
        }
    </style>

    <div class="card-container">
        <div class="card">
            <h3>Clasificación</h3>
            <p>Clasificar espacios disponibles.</p>
        </div>
        <div class="card">
            <h3>Regresión</h3>
            <p>Predecir cantidad de espacios libres.</p>
        </div>
    </div>
""", unsafe_allow_html=True)