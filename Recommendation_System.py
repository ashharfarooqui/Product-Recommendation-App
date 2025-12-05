import pandas as pd
import numpy as np
import streamlit as st
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix 
from scipy.sparse.linalg import svds

# ----------------------------
# Streamlit App Config
# ----------------------------
st.set_page_config(page_title="Recommendation System", layout="wide")
st.title("📊 Recommendation System - Matrix Factorization SVD")
# ----------------------------
# Upload Data
# ----------------------------
uploaded_file = st.file_uploader("📂 Upload your file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    required_cols = {'userid', 'productid', 'rating'}
    if not required_cols.issubset(df.columns):
        st.error(f"Dataset must contain columns: {required_cols}")
    else:

        tab1, tab2 = st.tabs(["📉 Visualization", "🎯 Recommendation"])
    
        with tab1:
            st.subheader("📑 Data Preview")
            st.write(df.head())
    
            st.subheader("📉 Rating Distribution")
            fig, ax = plt.subplots(figsize=(7, 3))
            sns.histplot(df['rating'], bins=10, kde=True, color='skyblue', ax=ax)
            ax.set_xlabel("Rating", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title("Rating Distribution", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
    
            
            st.subheader("📉 Number of Ratings Per User")
            user_counts = df['userid'].value_counts().head(20)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=user_counts.index, y=user_counts.values, palette='viridis', ax=ax)
            plt.xticks(rotation=90)
            ax.set_xlabel("User ID", fontsize=12)
            ax.set_ylabel("Number of Ratings", fontsize=12)
            ax.set_title("Top 20 Users by Rating Count", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
    
            st.subheader("📉 Number of Ratings Per product")
            product_counts = df['productid'].value_counts().head(20)
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(x=product_counts.index, y=product_counts.values, palette='viridis', ax=ax)
            plt.xticks(rotation=90)
            ax.set_xlabel("Product ID", fontsize=12)
            ax.set_ylabel("Number of Ratings", fontsize=12)
            ax.set_title("Top 20 Products by Rating Count", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
    
            # Convert date to datetime and extract year/month
            df['date'] = pd.to_datetime(df['date'], unit='s')
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
    
            st.subheader("📉 Average Rating by Year")
            yearly_avg = df.groupby('year')['rating'].mean()
            fig, ax = plt.subplots(figsize=(7, 4))
            yearly_avg.plot(kind='bar', color='coral', ax=ax)
            ax.set_ylabel("Average Rating", fontsize=12)
            ax.set_title("Average Rating by Year", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
    
        # ----------------------------
        # TAB 2: New Customer Prediction
        # ----------------------------
        with tab2:
            st.header("🎯 Personalized Recommendations using SVD")
            # Encode user and product IDs

             # Precompute average rating for each product
            product_avg_rating = df.groupby('productid')['rating'].mean().to_dict()
            # Cache SVD model to avoid recomputation
            @st.cache_resource
            def build_svd(df, k=20):
                # Encode user and product IDs
                user_codes = df['userid'].astype('category').cat.codes
                product_codes = df['productid'].astype('category').cat.codes

                # Build user-item matrix
                user_item_sparse = csr_matrix(
                    (df['rating'], (user_codes, product_codes)),
                    shape=(user_codes.max() + 1, product_codes.max() + 1)
                )

                # Apply SVD
                U, sigma, Vt = svds(user_item_sparse, k=k)
                sigma = np.diag(sigma)
                svd_pred = np.dot(np.dot(U, sigma), Vt)

                # Mapping between indices and IDs
                user_id_map = dict(enumerate(df['userid'].astype('category').cat.categories))
                product_id_map = dict(enumerate(df['productid'].astype('category').cat.categories))

                return svd_pred, user_id_map, product_id_map

            with st.spinner("Training SVD model (only once per dataset)..."):
                svd_pred, user_id_map, product_id_map = build_svd(df, k=20)

            # Recommendation function (No predicted rating shown)
            def recommend_svd(user_id, n=5):
                if user_id not in df['userid'].values:
                    return pd.DataFrame({'productid': []})

                # Get index for user
                user_idx = df['userid'].astype('category').cat.categories.get_loc(user_id)
                user_ratings = svd_pred[user_idx]

                # Get products user already rated
                rated_products = df[df['userid'] == user_id]['productid'].unique()
                unrated_indices = [i for i, pid in product_id_map.items() if pid not in rated_products]

                # Sort unrated products by predicted score
                top_indices = np.argsort(user_ratings[unrated_indices])[::-1][:n]
                recommended_products = [product_id_map[unrated_indices[i]] for i in top_indices]

                return pd.DataFrame({'Recommended_Products': recommended_products})
            
                # Add average rating from dataset
                avg_ratings = [round(product_avg_rating.get(pid, np.nan), 2) for pid in recommended_products]

                return pd.DataFrame({
                    'Recommended_Product': recommended_products,
                    'Average_Rating': avg_ratings
                })

            # ----------------------------
            # Streamlit Controls
            # ----------------------------
            st.subheader("👤 Select a User to Get Recommendations")

            user_list = df['userid'].unique()
            selected_user = st.selectbox("Select User ID:", user_list)
            n_recs = st.slider("Number of Recommendations", 1, 20, 5)

            # Generate recommendations
            if st.button("🎁 Get Recommendations"):
                with st.spinner("Generating product recommendations..."):
                    recs = recommend_svd(selected_user, n=n_recs)
                    if not recs.empty:
                        st.success(f"Top {n_recs} recommended products for user {selected_user}")
                        st.dataframe(recs)
                    else:
                        st.warning("No recommendations available for this user.")

else:
    st.info("👆 Please upload a dataset to get started.")