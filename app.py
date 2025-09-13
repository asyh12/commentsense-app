import streamlit as st
import pandas as pd
import joblib
import os
import re
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = "."

# spam & category rules
def apply_rules(comment, pred):
    if pred == "spam" or pd.isna(comment):
        return "spam"

    comment = comment.lower()

    emoji_pattern = r"(❤{2,}|😍{2,}|😂{2,}|🔥{2,}|💖{2,}|💯{2,}|✨{2,}|💎{2,})"
    link_pattern = r"(http|www|\.com|\.\w{2,4})"
    promo_pattern = r"(check my channel|subscribe|follow me|buy now|make money|free gift)"
    short_or_empty = len(comment.strip()) < 3 or re.fullmatch(r"[\W_]+", comment)
    repeated_chars = re.search(r"(.)\1{4,}", comment)

    if re.search(emoji_pattern, comment): return "spam"
    if re.search(link_pattern, comment): return "spam"
    if re.search(promo_pattern, comment): return "spam"
    if short_or_empty: return "spam"
    if repeated_chars: return "spam"

    return "ham"


def classify_category(comment, is_spam):
    if is_spam == "spam":
        return "spam"

    comment = comment.lower()

    skincare_keywords = ["skin", "skincare", "moisturizer", "serum", "cream", "lotion", "facewash", "cleanser"]
    fragrance_keywords = ["perfume", "fragrance", "cologne", "scent", "eau de", "aroma"]
    makeup_keywords = ["lipstick", "foundation", "mascara", "eyeshadow", "makeup", "concealer", "blush", "primer","lip balm"]

    if any(w in comment for w in skincare_keywords):
        return "skincare"
    elif any(w in comment for w in fragrance_keywords):
        return "fragrance"
    elif any(w in comment for w in makeup_keywords):
        return "makeup"
    else:
        return "other"

# Vectorizer + spam classifier
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
spam_model = joblib.load(os.path.join(BASE_DIR, "spam_model.pkl"))

# Sentiment models
#XLM_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification

XLM_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Use the official XLM-R tokenizer for CardiffNLP
xlm_tokenizer = XLMRobertaTokenizer.from_pretrained(
    XLM_MODEL_NAME,
    use_fast=True,
    add_prefix_space=True
)

xlm_model = AutoModelForSequenceClassification.from_pretrained(
    XLM_MODEL_NAME
).to("cpu")

ROBERTA_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

#xlm_tokenizer = AutoTokenizer.from_pretrained(XLM_MODEL_NAME, use_fast=False)
#xlm_model = AutoModelForSequenceClassification.from_pretrained(XLM_MODEL_NAME).to("cpu")

roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME, use_fast=False)
roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_NAME).to("cpu")

labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

comments = pd.read_parquet(os.path.join(BASE_DIR, "comments_final_labels.parquet"))
videos = pd.read_csv(os.path.join(BASE_DIR, "videos_sampled.csv"))

# sentiment prediction
def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cpu")
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).cpu().item()
    return labels.get(pred, "Unknown")

# streamlit
st.title("CommentSense: YouTube Comment Analyzer")
tab1, tab2 = st.tabs(["Video Dashboard", "Live Comment Demo"])

# tab1 video dashboard
with tab1:
    st.header("Video Comments Insights")

    if st.button("Choose Random Video"):
        valid_videos = comments["videoId"].unique()
        row = videos[videos["videoId"].isin(valid_videos)].sample(1).iloc[0]

        st.subheader(row["title"])
        st.write(row["description"])

        vid_id = row["videoId"]
        vid_comments = comments[comments["videoId"] == vid_id]

        if vid_comments.empty:
            st.warning("No processed comments found for this video.")
        else:
            st.write(f"Found {len(vid_comments)} processed comments for this video.")

            # Spam or not
            spam_quality = vid_comments["final_label"].apply(
                lambda x: "spam" if x == "spam" else "ham"
            )
            spam_counts = spam_quality.value_counts()

            st.subheader("Spam vs Quality Comments")
            if spam_counts.empty:
                st.info("No comments available for spam vs quality breakdown.")
            else:
                try:
                    fig, ax = plt.subplots()
                    ax.bar(spam_counts.index, spam_counts.values, color=["red", "green"])
                    ax.set_ylabel("Count")
                    ax.set_title("Spam vs Quality")
                    ax.yaxis.get_major_locator().set_params(integer=True)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not generate spam vs quality chart: {e}")


            # Category
            non_spam = vid_comments[vid_comments["final_label"] != "spam"]
            category_counts = non_spam["final_label"].value_counts()

            st.subheader("Category Distribution")
            if category_counts.empty:
                st.info("No non-spam comments available for category distribution.")
            else:
                fig, ax = plt.subplots()
                category_counts.plot(kind="bar", ax=ax, color="skyblue")
                ax.set_ylabel("Count")
                ax.set_title("Categories of Non-Spam Comments")
                ax.set_xticklabels(category_counts.index, rotation=0)
                ax.yaxis.get_major_locator().set_params(integer=True)
                st.pyplot(fig)

            # Sentiment percentage
            st.subheader("Sentiment Breakdown")

            if non_spam.empty:
                st.info("No non-spam comments available for sentiment analysis.")
            else:
                try:
                    non_spam["sentiment"] = non_spam["clean_comment"].apply(
                        lambda x: predict_sentiment(x, roberta_model, roberta_tokenizer)
                    )
                    sentiment_counts = non_spam["sentiment"].value_counts()

                    if sentiment_counts.empty:
                        st.info("Sentiment analysis did not return any results.")
                    else:
                        fig, ax = plt.subplots()
                        ax.pie(
                            sentiment_counts,
                            labels=sentiment_counts.index,
                            autopct="%1.1f%%",
                            startangle=90
                        )
                        ax.axis("equal")
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Sentiment analysis failed: {e}")


            # Engagement
            st.subheader("Engagement Metrics")
            views = int(row.get("viewCount", 0))
            likes = int(row.get("likeCount", 0))
            comments_count = int(row.get("commentCount", 0))

            like_rate = (likes / views * 100) if views > 0 else 0
            comment_rate = (comments_count / views * 100) if views > 0 else 0
            soe = ((likes + comments_count) / views * 100) if views > 0 else 0

            st.metric("Views", f"{views:,}")
            st.metric("Likes", f"{likes:,} ({like_rate:.2f}%)")
            st.metric("Comments", f"{comments_count:,} ({comment_rate:.2f}%)")
            st.metric("SoE", f"{soe:.2f}%")

            # processed comment
            st.caption(
                f"Note: Found {len(vid_comments)} processed comments "
                f"(vs {comments_count} total comments reported by YouTube)."
            )

            st.subheader("Processed Comments Preview")
            st.dataframe(
                vid_comments[["clean_comment", "final_label"]].head(10).reset_index(drop=True)
            )

            with st.expander("See all processed comments for this video"):
                st.dataframe(
                    vid_comments[["clean_comment", "final_label"]].reset_index(drop=True)
                )

# tab2 live comment prediction
with tab2:
    st.header("Comment Analyzer")

    comment = st.text_area("Enter a comment:")

    sentiment_choice = st.selectbox(
        "Choose Sentiment Model:",
        ["XLM-RoBERTa", "RoBERTa English"]
    )

    if st.button("Analyze Comment"):
        if comment.strip():
            X = vectorizer.transform([comment])
            spam = spam_model.predict(X)[0]

            category = classify_category(comment, spam)

            if sentiment_choice == "XLM-RoBERTa":
                sentiment_label = predict_sentiment(comment, xlm_model, xlm_tokenizer)
            else:
                sentiment_label = predict_sentiment(comment, roberta_model, roberta_tokenizer)

            st.write("**Spam or Not:**", spam)
            st.write("**Category:**", category)
            st.write(f"**Sentiment ({sentiment_choice}):**", sentiment_label)
        else:
            st.warning("Please enter a comment.")
