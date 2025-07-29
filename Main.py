import os
import shutil
import streamlit as st

st.set_page_config(
    page_title="Data Sensei",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/naveenkumarj25/Knowledge_Representation/blob/main/README.md',
        'About': "# One spot to know everything about your CSV!"
    }
)

if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

# Lazy imports
@st.cache_resource(show_spinner=False)
def lazy_imports():
    import src.KnowRep as KnowRep
    import src.Tools as Tools
    import src.Model as Model
    import src.Processing as Processing
    import src.chat_with_csv.chat_with_csv as chat_with_csv
    import src.chat_with_csv.ui_template as ui
    return KnowRep, Tools, Model, Processing, chat_with_csv, ui

KnowRep, Tools, Model, Processing, chat_with_csv, ui = lazy_imports()
Tools.make_folders()

# Sidebar
with st.sidebar:
    st.image("https://res.cloudinary.com/dgrhfkocl/image/upload/v1753768271/f2ee25161a920fe0cbbbefdda8c0f934-removebg-preview_lwno7p.png", width=200)
    st.title("Data Sensei")
    st.session_state.api_key = st.text_input("Enter your API Key", type="password", value=st.session_state.api_key)
    isExampleFileSelected = st.toggle("Use Example File", value=False, key="example_file_toggle")
    uploaded_file = None

    if isExampleFileSelected:
        selectedExample = st.selectbox("Select Example", Tools.AVAILABLE_EXAMPLES.keys(), index=0)
    else:
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if (uploaded_file or isExampleFileSelected) and st.session_state.api_key and (not st.session_state.file_uploaded or isExampleFileSelected):
        if st.button("Process File"):
            with st.spinner("Processing..."):
                try:
                    if isExampleFileSelected:
                        source_path = Tools.AVAILABLE_EXAMPLES[selectedExample]
                        destination_path = f'{Tools.ORIGINAL_PATH}{selectedExample}.csv'
                        shutil.copy(source_path, destination_path)
                    elif Tools.save_file(uploaded_file, Tools.ORIGINAL_PATH) != 1:
                        raise Exception("Failed to save file")

                    Processing.preprocess_dataset()
                    st.session_state.file_uploaded = True
                    KnowRep.make_llm(st.session_state.api_key)
                    st.success("File processed successfully!")

                except Exception as e:
                    st.error(f"Error: {e}")

    if st.button("Reset Application"):
        with st.spinner("Resetting..."):
            try:
                Tools.delete_files()
                for key in ['file_uploaded', 'insights', 'display_insights', 'result', 'chat_started', 'model_trained', 'model_accuracy', 'labelEncoder']:
                    st.session_state[key] = False if isinstance(st.session_state.get(key), bool) else None
                st.rerun()
            except Exception as e:
                st.error(f"Error during reset: {e}")

# Main content
st.markdown("# **KNOWLEDGE REPRESENTATION ON STRUCTURED DATASETS**")
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Insights Generation", "Chat with CSV", "ML Prediction"])

with tab1:
    st.header("Welcome to Data Sensei")
    st.markdown("""
        1. Enter your Gemini API key in the sidebar. [Click here to obtain one.](https://aistudio.google.com/app/apikey)
        2. Upload a CSV file.
        3. Click :orange[Process File].
        4. Use the tabs to explore features.
        5. Click :orange[Reset Application] to start fresh.
    """)
    if st.session_state.file_uploaded:
        st.markdown("##### Uploaded DataFrame")
        st.dataframe(Tools.load_csv_files(Tools.PATH, key='dataframe').head(5), use_container_width=True)

with tab2:
    st.header("Insights Generation")
    if 'insights' not in st.session_state:
        st.session_state['insights'] = ''
    if 'display_insights' not in st.session_state:
        st.session_state.display_insights = False

    if st.session_state.file_uploaded:
        if st.button("Generate Insights", key="generate_insights"):
            with st.spinner("Analyzing data..."):
                try:
                    df_string = Tools.load_csv_files(Tools.PATH, key='string')
                    st.session_state.insights = KnowRep.generate_insights(df_string)
                    st.session_state.display_insights = True
                    df = Tools.load_csv_files(Tools.PATH, key='dataframe')
                    charts = KnowRep.generate_and_extract_charts(df)
                    Processing.Visualize_charts(charts)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.warning("Please upload and process a CSV file first.")

    if st.session_state.display_insights:
        st.markdown("### 📊 Insights")
        st.markdown(st.session_state.insights)
        st.markdown("### 📈 Visualizations")
        for file in os.listdir(Tools.VISUALIZE_PATH):
            if file.endswith(".png"):
                st.image(os.path.join(Tools.VISUALIZE_PATH, file), use_container_width=True)

with tab3:
    st.header("Chat with CSV")
    if 'chat_started' not in st.session_state:
        st.session_state['chat_started'] = False
    if st.session_state.file_uploaded:
        if st.button("Start Chat", disabled=st.session_state.chat_started):
            st.session_state.chat_started = True
            chat_with_csv.initChat()

        if st.session_state.chat_started:
            user_question = st.chat_input("Ask a question about your data:")
            if user_question:
                with st.spinner("Processing question..."):
                    try:
                        st.write(ui.CSS, unsafe_allow_html=True)
                        chat_with_csv.handle_userinput(user_question)
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.write(ui.bot_template("Sorry, something went wrong."), unsafe_allow_html=True)
    else:
        st.warning("Please upload and process a CSV file first.")

with tab4:
    for key, default in {
        'model_accuracy': '',
        'submitted': None,
        'target_column': 'Auto',
        'prediction_type': 'Auto',
        'model_trained': None,
        'result': None
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    st.header("ML Prediction")
    if st.session_state.file_uploaded:
        st.markdown("##### Sample DataFrame")
        st.dataframe(Tools.load_csv_files(Tools.PATH, key='dataframe').head(5), use_container_width=True)

        if not st.session_state.model_trained:
            columns_display = ['Auto'] + [col for col in Tools.fetch_columns()]
            st.session_state.target_column = st.selectbox('Select Target Column', columns_display, index=0)
            st.session_state.prediction_type = st.selectbox('Select Prediction Type', ['Auto', 'Classification', 'Regression'], index=0)

        if not st.session_state.model_trained and st.button('Train ML Model'):
            with st.spinner("Training the model..."):
                try:
                    df = Tools.load_csv_files(Tools.PATH, key='dataframe')
                    if st.session_state.target_column == 'Auto':
                        st.session_state.target_column = KnowRep.get_target(df.head(5).to_string())
                    if st.session_state.prediction_type == 'Auto':
                        st.session_state.prediction_type = KnowRep.dataset_type(df.head(5).to_string())

                    accuracy, le = Model.create_model(df, st.session_state.target_column, st.session_state.prediction_type)
                    st.session_state.model_accuracy = accuracy
                    st.session_state.labelEncoder = le
                    st.session_state.model_trained = True
                except Exception as e:
                    st.error(f"Error during training: {e}")

        if st.session_state.model_accuracy:
            st.success(f"Model trained successfully with accuracy: {st.session_state.model_accuracy}")

        if st.session_state.model_trained:
            with st.form("Prediction Input Form"):
                input_data = {}
                prediction_columns = Tools.fetch_columns()
                prediction_columns.remove(st.session_state.target_column)
                column_types = Tools.column_dtype(prediction_columns)

                for col in prediction_columns:
                    if col in column_types['Numerical']:
                        input_data[col] = st.number_input(f"{col}:", key=f"num_{col}")
                    elif col in column_types['DateTime']:
                        input_data[col] = st.date_input(f"{col}:", key=f"date_{col}")
                    else:
                        input_data[col] = st.text_input(f"{col}:", key=f"text_{col}")
                submitted = st.form_submit_button("Predict")

            if submitted:
                with st.spinner("Running prediction..."):
                    try:
                        st.session_state.result = Model.predict_model(
                            user_input=input_data,
                            column_dropped=st.session_state.target_column,
                            data_type=st.session_state.prediction_type,
                            le=st.session_state.labelEncoder,
                            columns=prediction_columns
                        )
                        st.success("Prediction completed successfully!")
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")

        if st.session_state.result:
            st.markdown("### Prediction Result")
            st.markdown(f"**Prediction:** {st.session_state.result}")
    else:
        st.warning("Please upload and process a CSV file first.")
