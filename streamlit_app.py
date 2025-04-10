
import streamlit as st
import pandas as pd
import requests
import json
import time
from openai import OpenAI
from io import BytesIO
import anthropic
from yandex_cloud_ml_sdk import YCloudML
from mistralai import Mistral
from time import sleep
import xlsxwriter, openpyxl
import os

os.environ['nebius_api'] = st.secrets['nebius_api']
os.environ['antropic_api'] = st.secrets['antropic_api']
os.environ['yandex_api'] = st.secrets['yandex_api']
os.environ['mistral_api'] = st.secrets['mistral_api']
os.environ['openrouter_api'] = st.secrets['openrouter_api']
os.environ['folder_id1'] = st.secrets['folder_id1']

# Инициализация клиентов
def init_clients():
    return {
        "anthropic": anthropic.Client(api_key=os.environ['antropic_api']),
        "nebius": OpenAI(
            base_url="https://api.studio.nebius.ai/v1/",

            api_key=os.environ['nebius_api']
        ),
        "yandex": YCloudML(
            folder_id=os.environ['folder_id1'],
            auth=os.environ['yandex_api']
        ),
        "mistral": Mistral(api_key=os.environ['mistral_api'] ),
        "openrouter": OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ['openrouter_api']
        )
    }

clients = init_clients()

# Модели для тестирования
MODELS = {
    "Claude3.5": {"provider": "anthropic", "model": "claude-3-sonnet-20240229"},
    "Claude3.7": {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
    "DeepSeek-V3": {"provider": "nebius", "model": "deepseek-ai/DeepSeek-V3"},
    "DeepSeek-R1": {"provider": "nebius", "model": "deepseek-ai/DeepSeek-R1"},
    "Grok-2": {"provider": "openrouter", "model": "x-ai/grok-2-1212"},
    "YandexGPT5": {"provider": "yandex", "model": "yandexgpt"},
    "Gemini": {"provider": "openrouter", "model": "google/gemini-2.0-flash-001"},
    "Llama-3-70B": {"provider": "openrouter", "model": "meta-llama/Meta-Llama-3.1-70B-Instruct"},
    "Hermes": {"provider": "openrouter", "model": "NousResearch/Hermes-3-Llama-3.1-70B"},
    "Qwen": {"provider": "openrouter", "model": "Qwen/Qwen2.5-72B-Instruct"},
    "Mistral-Large": {"provider": "mistral", "model": "mistral-large-latest"}
}

# ======================================
# Вкладка 1: Тестирование датасетов
# ======================================
def dataset_testing_tab():
    st.header("Тестирование LLM на датасетах")

    # Загрузка датасетов
    @st.cache_data
    def load_dataset(dataset_name):
        try:
            if dataset_name == "CISM":
                return pd.read_excel("cism_eng_dataset.xlsx")
            elif dataset_name == "CISSP":
                return pd.read_excel("cissp_eng_dataset.xlsx")
            elif dataset_name == "CISSP5":
                return pd.read_excel("cissp_eng_5.xlsx")
            elif dataset_name == "тест":
                return pd.read_excel("тест.xlsx")
        except Exception as e:
            st.error(f"Ошибка загрузки датасета: {e}")
            return None

    dataset_name = st.selectbox("Выберите датасет", ["CISM", "CISSP", "CISSP5","тест"])
    dataset = load_dataset(dataset_name)

    if dataset is not None:
        st.subheader("Первые 5 вопросов:")
        st.dataframe(dataset.head())

        model_choice = st.selectbox(
            "Выберите модель для тестирования",
            list(MODELS.keys())
        )

        if st.button("🚀 Начать тестирование"):
            test_model(model_choice, dataset)

def test_model(model_name, dataset):
    model_info = MODELS[model_name]
    responses = []
    df = dataset.copy()

    with st.spinner(f"Тестирование {model_name}..."):
        progress_bar = st.progress(0)
        total_questions = len(df)

        for i, row in df.iterrows():
            answer = "ERROR"
            try:
                system_prompt = str(row['System_prompt'])
                question = str(row['Question'])

                if model_info["provider"] == "anthropic":
                    response = clients["anthropic"].messages.create(
                        model=model_info["model"],
                        messages=[{"role": "user", "content": f"{system_prompt}\n{question}"}],
                        max_tokens=50
                    )
                    answer = response.content[0].text.strip()

                elif model_info["provider"] == "nebius":
                    response = clients["nebius"].chat.completions.create(
                        model=model_info["model"],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question},
                        ],
                        max_tokens=500,
                        temperature=0.3,
                    )
                    answer = response.choices[0].message.content.strip()

                elif model_info["provider"] == "openrouter":
                    response = clients["openrouter"].chat.completions.create(
                        model=model_info["model"],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question},
                        ],
                        max_tokens=500,
                        temperature=0.7
                    )
                    answer = response.choices[0].message.content.strip()
                    sleep(1)

                elif model_info["provider"] == "yandex":
                    result = clients["yandex"].models.completions("yandexgpt").run(
                        messages=[
                            {"role": "system", "text": system_prompt},
                            {"role": "user", "text": question}
                        ]
                    )
                    if hasattr(result, 'alternatives') and result.alternatives:
                        answer = result.alternatives[0].message.text.strip()

                elif model_info["provider"] == "mistral":
                    response = clients["mistral"].chat(
                        model=model_info["model"],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question}
                        ],
                        temperature=0.7
                    )
                    answer = response.choices[0].message.content.strip()

                responses.append(answer)
                progress_bar.progress((i + 1) / total_questions)
                time.sleep(0.5)

            except Exception as e:
                st.error(f"Ошибка в вопросе {i+1}: {str(e)}")
                responses.append(answer)
                progress_bar.progress((i + 1) / total_questions)

    df[model_name] = responses
    st.success("✅ Тестирование завершено!")
    st.dataframe(df)

    # Кнопки скачивания
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Скачать CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f'results_{model_name}.csv',
            mime='text/csv'
        )
    with col2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        st.download_button(
            label="📥 Скачать Excel",
            data=output.getvalue(),
            file_name=f'results_{model_name}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

# ======================================
# Вкладка 2: Чат с LLM
# ======================================
def chat_tab():
    st.header("Чат с LLM")

    # Инициализация истории чата
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Выбор модели
    model_choice = st.selectbox(
        "Выберите модель для чата",
        list(MODELS.keys())
    )

    # Системный промпт
    system_prompt = st.text_area(
        "Роль системы:",
        "You are a helpful AI assistant. Provide concise and accurate answers.",
        height=100
    )

    # Отображение истории чата
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Обработка пользовательского ввода
    if prompt := st.chat_input("Введите ваш вопрос..."):
        # Добавление сообщения пользователя
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Формирование контекста
        messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                try:
                    model_info = MODELS[model_choice]

                    if model_info["provider"] == "anthropic":
                        response = clients["anthropic"].messages.create(
                            model=model_info["model"],
                            messages=messages,
                            max_tokens=1000
                        )
                        answer = response.content[0].text

                    elif model_info["provider"] == "nebius":
                        response = clients["nebius"].chat.completions.create(
                            model=model_info["model"],
                            messages=messages,
                            max_tokens=1000
                        )
                        answer = response.choices[0].message.content

                    elif model_info["provider"] == "openrouter":
                        response = clients["openrouter"].chat.completions.create(
                            model=model_info["model"],
                            messages=messages,
                            max_tokens=1000
                        )
                        answer = response.choices[0].message.content

                    elif model_info["provider"] == "yandex":
                        result = clients["yandex"].models.completions("yandexgpt").run(
                            messages=messages
                        )
                        answer = result.alternatives[0].message.text

                    elif model_info["provider"] == "mistral":
                        response = clients["mistral"].chat(
                            model=model_info["model"],
                            messages=messages,
                            temperature=0.7
                        )
                        answer = response.choices[0].message.content

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")

# ======================================
# Главное меню
# ======================================
st.sidebar.title("LLM Sandbox")
tab = st.sidebar.radio("Выберите режим", ["Тестирование датасетов", "Чат с LLM"])

if tab == "Тестирование датасетов":
    dataset_testing_tab()
else:
    chat_tab()
