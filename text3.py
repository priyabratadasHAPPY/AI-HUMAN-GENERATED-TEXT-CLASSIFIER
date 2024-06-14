import streamlit as st
from transformers import pipeline

# Initialize the model
unmasker = pipeline('text-classification', model='priyabrat/New_AI_or_Humantext_categorisation', truncation=True, padding=True, max_length=50, add_special_tokens=True)

# Function to classify text
def classify_text(text):
    data = unmasker(text)
    label = [d['label'] for d in data if 'label' in d]
    score = [d['score'] for d in data if 'score' in d]
    for num in score:
        score1 = num * 100
    for val in label:
        if val == "LABEL_0":
            label1 = "AI text"
        else:
            label1 = "HUMAN TEXT"
    if label1 == "HUMAN TEXT":
        label2 = "AI text"
        score2 = 100 - score1
    else:
        label2 = "HUMAN TEXT"
        score2 = 100 - score1
    result = (label1, round(score1), label2, round(score2))
    return result

# Streamlit app layout
st.title("AI or Human Blog Classifier")
st.subheader("Determine if a blog is AI-generated or human-generated")

# Input text area for blog content
st.write("Enter your blog content below to classify whether it is AI-generated or human-generated.")
blog_content = st.text_area("Enter your blog content here:")

if st.button('Classify'):
    if blog_content:
        result = classify_text(blog_content)
        st.subheader("Classification Result:")
        st.write(f"**{result[0]}** with a confidence score of **{result[1]}%**")
        st.write(f"**{result[2]}** with a confidence score of **{result[3]}%**")
    else:
        st.error("Please enter some blog content to classify!")
else:
    st.info("Enter your blog content and click 'Classify' to determine if it is AI-generated or human-generated.")
