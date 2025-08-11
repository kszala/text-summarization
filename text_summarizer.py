import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load final model
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

st.title("🧠 AI Text Summarizer (Final - BART Large CNN)")
st.write("Paste a long paragraph or article below:")

user_input = st.text_area("Enter text here...", height=200)

if st.button("Summarize"):
    if len(user_input.strip()) < 30:
        st.warning("Please enter at least 30 characters.")
    else:
        inputs = tokenizer.encode(user_input, return_tensors="pt", max_length=1024, truncation=True)

        summary_ids = model.generate(
            inputs,
            max_length=120,     # enough space for a complete summary
            min_length=40,      # avoid too-short outputs
            length_penalty=2.0, # discourage overly long text
            num_beams=5,        # better search for optimal summary
            early_stopping=True,
            no_repeat_ngram_size=3  # avoid repeating phrases
        )


        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.subheader("Summary:")
        st.write(summary)
