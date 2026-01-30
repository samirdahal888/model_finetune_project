import streamlit as st
import requests
from config.config import Config
import traceback

EMOJIS = {"World": "🌍", "Sports": "🏈", "Business": "💼", "Sci/Tech": "🔬"}


def check_api() -> bool:
    try:
        response = requests.get(f"{Config.API_URL}/health", timeout=5)
        return response.status_code == 200 and response.json().get("model_loaded")
    except requests.exceptions.RequestException:
        return False


def predict(text: str) -> dict | None:
    try:
        response = requests.post(
            f"{Config.API_URL}/predict", json={"text": text}, timeout=30
        )
        if response.status_code == 200:
            res = response.json()
        else:
            res = response.text
        return res
    except requests.exceptions.RequestException:
        print(traceback.format_exc())
        return None


def main():
    st.title("Text classification")
    st.caption("Classify news into World, Sports, Business, or Sci/Tech")

    # check api connection
    if not check_api():
        st.error("API not connected")
        return

    text = st.text_area("Enter here : ")

    if st.button("Classify"):
        if not text or len(text) < 10:
            st.warning("Please enter atleast 10 character")

        with st.spinner("classifying....."):
            result = predict(text)

        st.markdown(
            f"""
        <div style="
            padding: 1rem;
            border-radius: 12px;
            background-color: #f5f7fa;
            border-left: 6px solid #4CAF50;
        ">
            <h4>Model Prediction</h4>
            <p><strong>Label:</strong> {result["predicted_label"]}</p>
            <p><strong>Confidence:</strong> {result["confidence"]:.2%}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()


