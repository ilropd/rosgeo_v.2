import streamlit as st
import tensorflow
import openpyxl
from keras.models import model_from_json

import pandas as pd
import numpy as np

st.title('ОПРЕДЕЛЕНИЕ КОНВЕКТОРА')
st.write('---')

st.sidebar.header('Ввод данных для обработки')
uploaded_file = st.sidebar.file_uploader(label='Выберите файл в формате xls для обработки')
cols = ['GGKP_korr', 'GK_korr', 'PE_korr', 'DS_korr', 'DTP_korr', 'Wi_korr', 'BK_korr', 'BMK_korr']
if uploaded_file is not None:
        st.write('*Файл загружен успешно*')

        df = pd.read_excel(uploaded_file, engine='openpyxl', header=1)

        df = df.dropna(axis='index', how='any')
        df = df.drop([0])
        df.reset_index(drop=True, inplace=True)
        for col in df.columns:
            if col not in cols:
                df.pop(col)
        df = df.reindex(columns=cols)

        def get_x_data(dataframe, columns=cols):
            get_x = dataframe[columns].values.astype(np.float32)
            print(f'Размер: {get_x.shape}')
            return get_x

        data_for_predict = get_x_data(df)
else:
    def accept_user_data():
        st.sidebar.write('')
        st.sidebar.write('или введите данные для вручную')
        ggkp_korr = st.sidebar.number_input('GGKP_korr')
        gk_korr = st.sidebar.number_input('GK_korr')
        pe_korr = st.sidebar.number_input('PE_korr')
        ds_korr = st.sidebar.number_input('DS_korr')
        dtp_korr = st.sidebar.number_input('DTP_korr')
        wi_korr = st.sidebar.number_input('Wi_korr')
        bk_korr = st.sidebar.number_input('BK_korr')
        bmk_korr = st.sidebar.number_input('BMK_korr')
        data = {'GGKP_korr': ggkp_korr,
                'GK_korr': gk_korr,
                'PE_korr': pe_korr,
                'DS_korr': ds_korr,
                'DTP_korr': dtp_korr,
                'Wi_korr': wi_korr,
                'BK_korr': bk_korr,
                'BMK_korr': bmk_korr
                }
        user_prediction_data = pd.DataFrame(data,index=[0])
        return user_prediction_data

    input_df = accept_user_data()

st.subheader('Введенные данные')

def args_to_types(output):
    res = []
    for i in output:
        if i == 0:
            res.append(2)
        elif i == 1:
            res.append(4)
        elif i == 2:
            res.append(80)
        else:
            res.append(0)
    res = pd.DataFrame(res)
    return res

if uploaded_file is not None:
    st.write(df)
else:
    st.write('*Загрузите файл или введите данные вручную*')
    st.write(input_df)

json_file = open('venv/Models/js_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('venv/Models/weights.h5')
print('Loaded model from disk')

result = st.button('Классифицировать')
if result:
    st.write('Результат классификации')
    if uploaded_file is not None:
        preds = loaded_model.predict(data_for_predict)
        pred_args = np.argmax(preds, axis=1)
        out = args_to_types(pred_args)
        st.write(out)
        out_csv = out.to_csv()
        download_file = st.download_button('Сохранить', data=out_csv)

    else:
        preds = loaded_model.predict(input_df)
        pred_args = np.argmax(preds, axis=1)
        out = args_to_types(pred_args)
        st.write(f'Коллектор: {out[0][0]}')
