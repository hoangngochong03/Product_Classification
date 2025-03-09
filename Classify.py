from transformers import BertForSequenceClassification, BertTokenizer
import streamlit as st
import torch
import io
import os
import zipfile
import gdown
import pandas as pd
# ---------------------------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# import pandas as pd
# model_path = "./saved_model"
# model = BertForSequenceClassification.from_pretrained(model_path)
# tokenizer = BertTokenizer.from_pretrained(model_path)
# model.to(device)
# ----------------------------------------------


device = torch.device("cpu")

model_path = "./model"


url = 'https://drive.google.com/file/d/1OqibMYh7pC0hF1EO3Utim0fOI_IUGW4H/edit'  
output = 'saved_model.zip'  

if not os.path.exists(model_path):
    print("Downloading the model from Google Drive...")
    gdown.download(url, output, quiet=False,fuzzy=True)

    if os.path.exists(output):
        print("Extracting the model...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(model_path)  
        print("Model extracted to:", model_path)

    os.remove(output)

print("Loading model and tokenizer...")
model = BertForSequenceClassification.from_pretrained(os.path.join(model_path,"saved_model"))
tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path,"saved_model"))

model.to(device)


print(f'Model is loaded and moved to: {device}')
#----------------------------------------------------------------
categories={0: 'Cà phê', 1: 'Fast food', 2: 'Học phí', 3: 'Nước ép', 4: 'Quần áo', 5: 'Rau củ', 6: 'Taxi', 7: 'Thuốc men', 8: 'Thức ăn thú cưng', 9: 'Trà', 10: 'Vé máy bay'}

st.title("Classify Product")
st.markdown("Danh mục: **" + " | ".join(categories.values()) + "**")
uploaded_file = st.file_uploader("Upload a excel, csv file( Có chứa cột |Tên sản phẩm|)", type=["csv","xlsx"])
product=st.text_input("Tên sản phẩm: ")
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    if 'Tên sản phẩm' not in df.columns:
        st.error("CSV file must have a column named 'Tên sản phẩm'")
    else:
        def classify_product(product):
            inputs = tokenizer(product, return_tensors="pt").to(device)
            outputs = model(**inputs)
            prediction = outputs.logits.argmax(dim=-1)
            return categories[prediction[0].item()]

        df['Prediction'] = df['Tên sản phẩm'].apply(classify_product)

        st.write(df)


    output = io.BytesIO()


    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')


    output.seek(0)


    st.download_button(
        label="Download Result",
        data=output,
        file_name="classified_products.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
elif product: 
    inputs = tokenizer(product, return_tensors="pt").to(device)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=-1)
    st.write("Phân loại: ",categories[prediction[0].item()])
