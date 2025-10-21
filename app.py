import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import os
from io import BytesIO
import instructor
from pycaret.regression import load_model, predict_model
from matplotlib.lines import Line2D
from langfuse.decorators import observe
from langfuse.openai import BaseModel, OpenAI
import boto3
import joblib
from io import BytesIO, StringIO
import datetime
import pandera as pa


# Tabs
USING_MODEL="Opisz siebie i swoje wyniki sportowe, aby przewidzieć twój czas w półmaratonie✍️"
SURVEY="Wypełnij formularz, aby przewidzieć twój czas w półmaratonie📝"
CHECK_SCORE="Sprawdź czy twoje wyniki pozwolą ci pobiec jak sobie wymarzyłęś/aś✅📈"
# Models
TEXT_TO_TEXT="gpt-4o"

if "page" not in st.session_state:
    st.session_state["page"]="intro"

if "user_text" not in st.session_state:
    st.session_state["user_text"]=""

if "date_of_birth" not in st.session_state:
    st.session_state["date_of_birth"]=0

if "sex" not in st.session_state:
    st.session_state["sex"]=""

if "speed_5km" not in st.session_state:
    st.session_state["speed_5km"]=0

if "time_5km" not in st.session_state:
    st.session_state["time_5km"]=""
    
if "speed_10km" not in st.session_state:
    st.session_state["speed_10km"]=0

if "time_10km" not in st.session_state:
    st.session_state["time_10km"]=""

if "time_half_maraton" not in st.session_state:
    st.session_state["time_half_maraton"]=0

if "is_ok_clicked" not in st.session_state:
    st.session_state["is_ok_clicked"]=False
# Functions

# OEPENAI LANGFUSE
def llm_key_get():
    api_client=OpenAI(api_key=st.secrets("OPENAI_API_KEY"))
    llm_key=instructor.from_openai(client=api_client)
    return llm_key

# Digital Ocean storage

def get_client():
    session=boto3.session.Session()
    try:
        client=session.client(
            's3',
            region_name=st.secrets("REGION_NAME"),
            endpoint_url=st.secrets("ENPOINT_URL_KEY"),  # zmień na swoje
            aws_access_key_id=st.secrets("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=st.secrets("AWS_SECRET_ACCESS_KEY")
        )
    except:
        st.error("Błąd wczytywania danych spróbuj ponownie")
        st.stop()
    return client

def get_compare_data():

    client=get_client()
    try:
        response=client.get_object(
            Bucket=st.secrets("BUCKET"),
            Key=st.secrets("MARATON_DATA")
        )

        stream = response["Body"]
        body = stream.read()
        csv_buffer = BytesIO(body)
        df = pd.read_csv(csv_buffer, sep=";")
    except:
        st.error("Błąd wczytywania danych spróbuj ponownie")
        st.stop()
    return df

def get_pipeline_model():
    client=get_client()

    try:
        response = client.get_object(
        Bucket=st.secrets("BUCKET"),
        Key=st.secrets("MARATON_MODEL")
    )

        model_buffer = BytesIO(response["Body"].read())
        model_buffer.seek(0)

        # Load model
        model = joblib.load(model_buffer)

    except:
        st.error("Błąd wczytywania danych modelu spróbuj ponownie")
        st.stop()
    return model
    

class UserInfo(BaseModel):
    sex:int
    time_5km:str
    time_10km:str

@observe
def text_to_dict_lang(prompt,model) -> UserInfo:
    system_content='''
            Jesteś specjalistą w szukaniu informacji w tekście dotyczących użytkownika 

            Twoim  zadaniem jest wyciągnięcie z tekstu przekazanego przez użytkownika informacji, które podaje na temat
            roku urodzenia, płci, czasu biegu na 5 km, czasu biegu na 10km, tempie biegu na 5km, tempie biegu na 10 km 
            i przedstawienie tych informacji. 

            <sex>: jeżeli użytkownik powie:
            - jest mężczyzną -> wypisz 1
            - jest kobietą -> wypisz 0
            - inne -> 2

            <time_5km>: Jeżlei użytkownik napisze, o czsie na 5 km wypisz ten czas,
            jeżeli nie ->0
            zawsze w formacie hh:mm:ss

            <time_10km>: Jeżlei użytkownik napisze, o czsie na 10 km wypisz ten czas,
            jeżeli nie ->0
            zawsze w formacie hh:mm:ss

            Odpowiedź zwróć w formacie JSON:
            {
                "sex": 1,
                "time_5km": "00:25:00",
                "time_10km": "00:55:00"
            }
            
            '''   
    messages=[
        {
            "role": "system",
            "content":system_content
        },
        {
            "role": "user",
            "content":prompt
        }
    ]

    try:
        llm_client=llm_key_get()
        chat_completion = llm_client.chat.completions.create(
        messages=messages,
        model=model,
        response_model=UserInfo,
        response_format={"type": "json_object"}
        )
        response=chat_completion.model_dump()
        
    except Exception as e:
        raise e
    return response

        
def veryfications_of_info(sex,time_5km, time_10km):
    return ((sex=="0")|(time_10km=="0")|(time_5km=="0"))
    
def veryfications_of_info_to_error(sex,time_5km, time_10km):
    response_string="Zapomniałeś dodać informacji o: "
    if sex==2:
        response_string+="płci "
    if time_5km=="0":
        response_string+="czasie na 5km "
    if time_10km=="0":
        response_string+="czasie na 10km "
    return response_string


def sec_to_time(seconds):
    #seconds = round(seconds)  # zaokrąglamy i konwertujemy na int
    hours = seconds // 3600
    time_between = seconds % 3600
    minutes = time_between // 60
    seconds_time = time_between % 60
    # formatowanie z zerami: 02:04:09
    return f"{hours:02}:{minutes:02}:{seconds_time:02}"


def time_to_sec(time_str):
    # h,m,s=[0,0,0]
    h=0
    m=0
    s=0
    try:
        parts = time_str.strip().split(":")
        if len(parts) == 2:
            m, s =map(int, parts)
            if m<3:
                h=m
                m=s
                s=0
        else:
            h, m, s = map(int, parts)
        if ((m>60 and m<0) or (s>60 and s<0)):
            return 0
        return h * 3600 + m * 60 + s
    except (ValueError, AttributeError):
        return 0


def string_to_speed(time_str):
    """Konwertuje czas w formacie HH:MM:SS na liczbę sekund (float)."""
    try: 
        result_time=float(time_str)
    except:
        return 0
    return result_time

def sec_to_time_diff(seconds):
    #seconds = round(seconds)  # zaokrąglamy i konwertujemy na int
    hours = seconds // 3600
    time_between = seconds % 3600
    minutes = time_between // 60
    seconds_time = time_between % 60
    # formatowanie z zerami: 02:04:09

    response_str=""
    if hours>0:
        if minutes>0 or seconds_time>0:
            response_str+=str(hours) + " godz. i "
        else:
            response_str+=str(hours) + " godz."
    if minutes>0:
        if seconds_time>0:
            response_str+=str(minutes) + " min. i "
        else:
            response_str+=str(minutes) + " min."
    if seconds_time>0:
        response_str+=str(seconds_time) + " sek."

    return response_str

def format_speed_diff(speed_dif):
    try:
        minutes = int(speed_dif)
        seconds = round((speed_dif - minutes) * 60)

        parts = []
        if minutes > 0:
            parts.append(f"{minutes} min")
        if seconds > 0:
            parts.append(f"{seconds} sek")

        return " ".join(parts) if parts else "0 sek"
    except Exception:
        return "Błąd danych"


def time_pred(predicted_score):

    full_resp=sec_to_time_diff(time_to_sec(predicted_score))
    st.markdown("""
        <style>
        .track-box {
            background: linear-gradient(135deg, #4CAF50, #81C784);
            border-radius: 25px;
            padding: 30px;
            text-align: center;
            color: white;
            font-size: 32px;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            border: 4px dashed white;
            margin-top: 20px;
            margin-bottom: 2px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Wyświetlanie przewidywanego czasu w stylizowanej bieżni
    st.markdown(f"""
        <div class="track-box">
            🏃‍♂️ Twój przewidywany czas: {predicted_score}, <br>
                Czyli: {full_resp}
        </div>
    """, unsafe_allow_html=True)
def time_exp_user(expected_score):
    resp_full=sec_to_time_diff(time_to_sec(expected_score))
    st.markdown("""
    <style>
    .exp-box {
        background: linear-gradient(135deg, #FFD700, #FFC300); /* mocny złoty gradient */
        border-radius: 25px;
        padding: 30px;
        text-align: center;
        color: white;
        font-size: 32px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 1);
        border: 4px dashed white;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    </style>
    """, unsafe_allow_html=True)

    # Wyświetlanie przewidywanego czasu w stylizowanej bieżni
    st.markdown(f"""
        <div class="exp-box">
            🏆 Czas jaki chcesz osiągnąć: {expected_score}, <br>
                    Czyli: {resp_full}
        </div>
    """, unsafe_allow_html=True)

def description_of_score(marathon_time:bool, time_diference:float):
    resp_full=sec_to_time_diff(time_diference)
    if marathon_time:
        st.markdown("""
                <style>
                .pred_lower-box {
                    background: linear-gradient(135deg, #E53935, #EF5350);
                    border-radius: 25px;
                    padding: 30px;
                    text-align: center;
                    color: white;
                    font-size: 20px;
                    font-weight: bold;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    border: 4px dashed white;
                    margin-top: 20px;
                    margin-bottom: 20px;
                }
                </style>
            """, unsafe_allow_html=True)

            # Wyświetlanie przewidywanego czasu w stylizowanej bieżni
        st.markdown(f"""
            <div class="pred_lower-box">
                🏃‍♂️❌ Przewidujemy, że pobiegniesz gorzej od oczekiwanego przez 
                    ciebie czasu o: {time_diference} sekund <br>
                Czyli o: {resp_full}
                
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
                <style>
                .pred_higher-box {
                    background: linear-gradient(135deg, #4CAF50, #81C784);
                    border-radius: 25px;
                    padding: 30px;
                    text-align: center;
                    color: white;
                    font-size: 20px;
                    font-weight: bold;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    border: 4px dashed white;
                    margin-top: 20px;
                    margin-bottom: 20px;
                }
                </style>
            """, unsafe_allow_html=True)

            # Wyświetlanie przewidywanego czasu w stylizowanej bieżni
        st.markdown(f"""
            <div class="pred_higher-box">
                🏃‍♂️✅ Przewidujemy, że pobiegniesz lepiej od oczekiwanego przez 
                    ciebie czasu o: {time_diference} sekund <br>
                Czyli o: {resp_full}
            </div>
        """, unsafe_allow_html=True)

def desp_of_speed(marathon_time:bool, speed_diference:float, exp_time):

    full_resp=format_speed_diff(speed_diference)
    if marathon_time:
        st.markdown("""
                <style>
                .speed_lower-box {
                    background: linear-gradient(135deg, #E53935, #EF5350);
                    border-radius: 25px;
                    padding: 30px;
                    text-align: center;
                    color: white;
                    font-size: 20px;
                    font-weight: bold;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    border: 4px dashed white;
                    margin-top: 1px;
                    margin-bottom: 10px;
                }
                </style>
            """, unsafe_allow_html=True)

            # Wyświetlanie przewidywanego czasu w stylizowanej bieżni
        st.markdown(f"""
            <div class="speed_lower-box">
                🏃‍♂️❌ Patrząc na wykres możemy wywyniokskowac, że biegniesz tempem
                o: {speed_diference} min/km wolniejszym niż średnie tempo potrzebna do osiągnięcia czasu {expected_score},<br>
                Czyli biegasz dystans 1km o: {full_resp} wolniej niż minimalny czas na 1km potrzebny do osiągnięcia oczekiwanego rezultatu w półmaratonie
                
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
                <style>
                .speed_higher-box {
                    background: linear-gradient(135deg, #4CAF50, #81C784);
                    border-radius: 25px;
                    padding: 30px;
                    text-align: center;
                    color: white;
                    font-size: 20px;
                    font-weight: bold;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    border: 4px dashed white;
                    margin-top: 1px;
                    margin-bottom: 10px;
                }
                </style>
            """, unsafe_allow_html=True)

            # Wyświetlanie przewidywanego czasu w stylizowanej bieżni
        st.markdown(f"""
            <div class="speed_higher-box">
                🏃‍♂️✅ Patrząc na wykres możemy wywyniokskowac, że biegniesz tempem
                o: {speed_diference} min/km szybszym niż średnie tempo potrzebna do osiągnięcia czasu {expected_score}, <br>
                Czyli biegasz dystans 1km o: {full_resp} szybciej niż minimalny czas na 1km potrzebny do osiągnięcia oczekiwanego rezultatu w półmaratonie
            </div>
        """, unsafe_allow_html=True)

def desp_of_5km(marathon_time:bool, time_diference_5km:float, exp_time):
    time_diference_5km=sec_to_time_diff(time_diference_5km)
    if marathon_time:
        st.markdown("""
                <style>
                .time_5km_lower-box {
                    background: linear-gradient(135deg, #E53935, #EF5350);
                    border-radius: 25px;
                    padding: 30px;
                    text-align: center;
                    color: white;
                    font-size: 20px;
                    font-weight: bold;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    border: 4px dashed white;
                    margin-top: 20px;
                    margin-bottom: 20px;
                }
                </style>
            """, unsafe_allow_html=True)

            # Wyświetlanie przewidywanego czasu w stylizowanej bieżni
        st.markdown(f"""
            <div class="time_5km_lower-box">
                🏃‍♂️❌ Aby przebiec półmaraton w czasie {exp_time} <br>
                musisz pobiec szybciej o: {time_diference_5km} dystans na 5 km
                
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
                <style>
                .time_5km_higher-box {
                    background: linear-gradient(135deg, #4CAF50, #81C784);
                    border-radius: 25px;
                    padding: 30px;
                    text-align: center;
                    color: white;
                    font-size: 20px;
                    font-weight: bold;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    border: 4px dashed white;
                    margin-top: 1px;
                    margin-bottom: 20px;
                }
                </style>
            """, unsafe_allow_html=True)

            # Wyświetlanie przewidywanego czasu w stylizowanej bieżni
        st.markdown(f"""
            <div class="time_5km_higher-box">
                🏃‍♂️✅ Prędkość, którą biegasz na 5km jest wystarczająca do <br>
                    przebiegnięcia półmaratonu w czasie {exp_time}
            </div>
        """, unsafe_allow_html=True)

def desp_of_10km(marathon_time:bool, time_diference_10km:float, exp_time):
    time_diference_10km=sec_to_time_diff(time_diference_10km)
    if marathon_time:
        st.markdown("""
                <style>
                .time_10km_lower-box {
                    background: linear-gradient(135deg, #E53935, #EF5350);
                    border-radius: 25px;
                    padding: 30px;
                    text-align: center;
                    color: white;
                    font-size: 20px;
                    font-weight: bold;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    border: 4px dashed white;
                    margin-top: 5px;
                    margin-bottom: 5px;
                }
                </style>
            """, unsafe_allow_html=True)

            # Wyświetlanie przewidywanego czasu w stylizowanej bieżni
        st.markdown(f"""
            <div class="time_10km_lower-box">
                🏃‍♂️❌ Aby przebiec półmaraton w czasie {exp_time} <br>
                musisz pobiec szybciej o: {time_diference_10km} dystans na 10 km
                
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
                <style>
                .time_10km_higher-box {
                    background: linear-gradient(135deg, #4CAF50, #81C784);
                    border-radius: 25px;
                    padding: 30px;
                    text-align: center;
                    color: white;
                    font-size: 20px;
                    font-weight: bold;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    border: 4px dashed white;
                    margin-top: 1px;
                }
                </style>
            """, unsafe_allow_html=True)

            # Wyświetlanie przewidywanego czasu w stylizowanej bieżni
        st.markdown(f"""
            <div class="time_10km_higher-box">
                🏃‍♂️✅ Prędkość, którą biegasz na 10km jest wystarczająca do <br>
                przebiegnięcia półmaratonu w czasie {exp_time}
            </div>
        """, unsafe_allow_html=True)


#
## MAIN
#

if st.session_state["page"]=="intro":
    st.markdown('''
                ## Witaj w aplikacji CheckYourTime⏱️
                Przewidzimy dla ciebie jaki czas możesz osiągnac w półmaratonie''')
    
    if st.button("Kontunuuj"):
        st.session_state["page"]="main"
        st.rerun()

if st.session_state["page"]=="main":
    st.markdown("# CheckYourTime⏱️")
    tab=st.selectbox("# Wybierz opcję, która cię interesuje", options=[USING_MODEL, SURVEY])

    if tab==USING_MODEL:
        
        st.markdown("""
        ### 📝 Podaj swoje dane.

        Wpisz poniżej w jednym tekście następujące informacje:

        - 🚻 Płeć   
        - ⏱️ Czas biegu na **5 km** (np. 00:27:30) \n
          🔸 Zakres: **00:17:00 – 00:39:00**
        - ⏱️ Czas biegu na **10 km** (np. 1:00:00)\n
          🔸 Zakres: **00:35:00 – 01:17:00**

        Na podstawie tego tekstu automatycznie wyciągniemy dane do analizy
        """)

        user_text=st.text_area(
            label="Podaj informacje w polu poniżej🔻",
            placeholder="np. Jestem mężczyzną na 5 km biegam 00:30:10 i na 10km biegam 00:54:10"

        )
        st.session_state["user_text"]=user_text

        #user_info,placeholder, label="Powiedz coś o sobie"
        st.info("Jeżeli jesteś zadowolony/a z podanych przez ciebie informacji naciśnij zatwierdź ✅")
        if st.button("Zatwierdź"):
            st.session_state["is_ok_clicked"]=False
            st.session_state["page"]="confirm"
            st.rerun()
        


    if tab==SURVEY:
        st.markdown("## Wypełnij ankietę, pamiętaj żeby wypełnić każdą komórkę 📝")

        # date_of_birth=st.text_input("Podaj datę urodzenia 🎂")
        # st.session_state["date_of_birth"]=date_of_birth

        sex=st.radio("Podaj płeć",["Mężczyzna 👨", "Kobieta 👩"])
        sex =1 if sex =="Mężczyzna 👨" else 0
        st.session_state["sex"]=sex

        time_5km = st.text_input(
            "⏱️ Podaj swój czas na 5 km \n 🔸 Zakres: **00:17:00 – 00:39:00**",
            placeholder="np. 00:36:22"
            )

        st.session_state["time_5km"]=time_5km

        time_10km=st.text_input(" ⏱️ Podaj twój czas na 10km \n 🔸 Zakres: **00:35:00 – 01:17:00** " \
        "", placeholder="np. 00:55:22")
        st.session_state["time_10km"]=time_10km


        st.info("Jeżeli jesteś zadowolony/a z podanych przez ciebie informacji naciśnij zatwierdź ✅")
        if st.button("Zatwierdź"):
            st.session_state["is_ok_clicked"]=False
            st.session_state["page"]="confirm"
            st.rerun()

if st.session_state["page"] == "confirm":
    st.markdown("### Chcesz sprawdzić, jak Twój wymarzony czas w półmaratonie wypada na tle naszego przewidywanego wyniku?")
    st.markdown("##### Wpisz czas jaki chcesz osiągnąć w poniższym polu i wciśnij Zatwierdź ✅")

    expected_score=st.text_input("Podaj czas, w jakim chciałbyś przebiec półmaraton⏱️", placeholder="np. 01:30:22")
    st.session_state["time_half_maraton"]=expected_score
    col1, col2,col3, col4,col5, col6 = st.columns(6)
    with col5:
        if st.button("Pomiń"):
            st.session_state["time_half_maraton"]=0
            st.session_state["page"]="result_marathon_time"
            st.rerun()
    with col6:
        if st.button("Zatwierdź"):
            st.session_state["is_ok_clicked"]=True
    if st.session_state["is_ok_clicked"]==True:
        if st.session_state["time_half_maraton"]:

            if time_to_sec(st.session_state["time_half_maraton"])==0:
                st.info("Sprawdź czy podałeś prawidłowe informacje o oczekiwanym rezultacie i czy są w odpowiednim formacie")
                st.stop()
            else:
                st.session_state["page"]="result_marathon_time"
                st.rerun()
        else:
            if  st.session_state["is_ok_clicked"]==True:
                st.info("Sprawdź czy podałeś prawidłowe informacje o oczekiwanym rezultacie i czy są w odpowiednim formacie")
                st.stop()
            
if st.session_state["page"]=="result_marathon_time":
    if st.button("Wróć"):
        st.session_state["page"]="main"
        st.rerun()
    st.markdown("## 🏁 Twój przewidywany czas półmaratonu: ")
    st.write("---")  # linia oddzielająca
    user_text=st.session_state["user_text"]
    
    if st.session_state["user_text"]!="":
        next=False

        user_text=st.session_state["user_text"]
        dict_user=text_to_dict_lang(user_text, TEXT_TO_TEXT)

        if veryfications_of_info(dict_user["sex"], dict_user["time_5km"], dict_user["time_10km"]):
            info=veryfications_of_info_to_error(dict_user["sex"], dict_user["time_5km"], dict_user["time_10km"])
            st.info(info)
            st.stop()
            st.session_state["page"]="main"
            st.rerun()

        st.session_state["sex"]=dict_user["sex"]
        st.session_state["time_5km"]=dict_user["time_5km"]
        st.session_state["time_10km"]=dict_user["time_10km"]
        next=True
    next=True
    if next==True:
       
        if time_to_sec(st.session_state["time_5km"])==0 or time_to_sec(st.session_state["time_10km"])==0:
            st.info("Sprawdź czy podałeś wszytskie informacje lub czy podałeś je w dobrym formacie")
            st.stop()
            st.session_state["page"]="main"
            st.rerun()
        # sex=st.session_state["sex"]
        df=pd.DataFrame(
        [
            {
             "sex":st.session_state["sex"],
             "time_5km": time_to_sec(st.session_state["time_5km"]),
             "time_10km":time_to_sec(st.session_state["time_10km"]),
            }
        ])

        schema=pa.DataFrameSchema(
            {
             "sex":pa.Column(int),
             "time_5km": pa.Column(int ,pa.Check.in_range(1030,2350)),
             "time_10km":pa.Column(int, pa.Check.in_range(2100,4620)),
            }
        )

        try:
            schema.validate(df)
        except:
            st.info("Podaj prawidłowe informacje, stosując również przedstawione zakresy czasowe")
            st.stop()

        model = get_pipeline_model()

        # przewidujemy tylko dla JEDNEGO wiersza
        pred_df = predict_model(model, data=df)

        # pobieramy dokładnie jedną wartość z prediction_label
        pred_seconds = round(pred_df["prediction_label"].iloc[0])  # np. 3874.23

        # zamieniamy na czytelny format czasu
        converted_time = sec_to_time(pred_seconds)  # np. "1:04:34"

        # wyświetlamy
       # CSS + HTML

        time_pred(converted_time)
        
        expected_score=st.session_state["time_half_maraton"]
        if expected_score !=0:
            time_exp_user(expected_score)
            try:
                expected_time=time_to_sec(expected_score)
                # st.markdown(expected_time)
                # st.markdown(pred_seconds)
            except:
                st.error("Błąd sprubuj ponownie")
            
            if pred_seconds>expected_time:
                marathon_time_bool=True
            else:
                marathon_time_bool=False

            time_diff=int(abs(pred_seconds-expected_time))
            description_of_score(marathon_time_bool,time_diff)

            df_data=get_compare_data()

            vis_df=df_data[df_data["sex"]==st.session_state["sex"]]

            df_model=vis_df[(vis_df["time"]>= pred_seconds - 50) & (vis_df["time"]<=pred_seconds +50)]
            chceck_df=vis_df[(vis_df["time"]>= expected_time - 50) & (vis_df["time"]<=expected_time +50)]


            if chceck_df.shape[0] >= 1 and df_model.shape[0] >= 1:

                exp_t_model=df_model.iloc[0]
                exp_time=chceck_df.iloc[0]
                #st.dataframe(exp_time)
                st.markdown("# Możesz sprawdzić swoje wyniki na wykresach poniżej 📈 🔻")
                with st.expander("Sprawdź swoje wyniki na wykresach 📈"):
                    fig, ax = plt.subplots(1, 1, figsize=(8, 4))  # szeroki, ale nie za wysoki
                    ax.plot(vis_df["time"], vis_df["tempo"])
                    # Punkty start i koniec
                    x1, y1 = exp_time["time"], exp_time["tempo"]
                    x2, y2 = pred_seconds,exp_t_model["tempo"]
                    # Dodanie strzałki
                    ax.annotate("",
                        xy=(x1, y1), xytext=(x2, y2),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2)
                    )
                    # st.pyplot(fig)
                    ax.text(x1, y1, ".",color="yellow", fontsize="40")  # s to wielkość punktu
                    ax.text(x2, y2, ".",color="red", fontsize="40")
                    ax.set_title("Porównanie tempa biegu do osiągniętych czasów")
                    ax.set_xlabel("Czas na całym dystansie w sekundach")
                    ax.set_ylabel("Tempo na całym dystansie w min/km")
                    legend_elements = [
                        Line2D([0], [0], marker='o', color='w', label='Wynik jaki chcesz osiągnąć (żółta kropka)',
                            markerfacecolor='yellow', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='Twój wynik (czerwona kropka)',
                            markerfacecolor='red', markersize=10)
                    ]
                    ax.legend(handles=legend_elements)
                    st.pyplot(fig)
                    
                    speed_diff=(abs(exp_t_model["tempo"]-exp_time["tempo"])).round(2)
                    desp_of_speed(marathon_time_bool, speed_diff,expected_score)
                    
                    fig, ax=plt.subplots(1,1, figsize=(8,4))
                    ax.plot(vis_df["time"], vis_df["time_5km"])
                    x1_2, y1_2 = exp_time["time"], exp_time["time_5km"]
                    x2_2, y2_2 = pred_seconds,exp_t_model["time_5km"]
                    ax.annotate("",
                        xy=(x1_2, y1_2), xytext=(x2_2, y2_2),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2)
                    )
                    ax.text(x1_2, y1_2, ".",color="yellow", fontsize="40")  # s to wielkość punktu
                    ax.text(x2_2, y2_2, ".",color="red", fontsize="40")
                    ax.set_xlim(min(vis_df["time"].min(), x1_2, x2_2) - 10, max(vis_df["time"].max(), x1_2, x2_2) + 10)
                    ax.set_title("Porównanie czasu na 5km do osiągniętych rezultatów")
                    ax.set_xlabel("Czas na całym dystansie w sekundach")
                    ax.set_ylabel("Czas na 5km w sekundach")
                    legend_elements = [
                        Line2D([0], [0], marker='o', color='w', label='Wynik jaki chcesz osiągnąć (żółta kropka)',
                            markerfacecolor='yellow', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='Twój wynik (czerwona kropka)',
                            markerfacecolor='red', markersize=10)
                    ]
                    ax.legend(handles=legend_elements)
                    
                    st.pyplot(fig)
                    
                    time_5km_diff=int((abs(exp_t_model["time_5km"]-exp_time["time_5km"])))
                    desp_of_5km(marathon_time_bool, time_5km_diff,expected_score)
                   

                    fig, ax=plt.subplots(1,1, figsize=(8,4))
                    ax.plot(vis_df["time"], vis_df["time_10km"])
                    x1_2, y1_2 = exp_time["time"], exp_time["time_10km"]
                    x2_2, y2_2 = pred_seconds,exp_t_model["time_10km"]
                    ax.annotate("",
                        xy=(x1_2, y1_2), xytext=(x2_2, y2_2),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2)
                    )
                    ax.text(x1_2, y1_2, ".",color="yellow", fontsize="40")  # s to wielkość punktu
                    ax.text(x2_2, y2_2, ".",color="red", fontsize="40")
                    ax.set_xlim(min(vis_df["time"].min(), x1_2, x2_2) - 10, max(vis_df["time"].max(), x1_2, x2_2) + 10)
                    ax.set_title("Porównanie czasu na 10km do osiągniętych rezultatów")
                    ax.set_xlabel("Czas na całym dystansie w sekundach")
                    ax.set_ylabel("Czas na 10km w sekundach")

                    legend_elements = [
                        Line2D([0], [0], marker='o', color='w', label='Wynik jaki chcesz osiągnąć (żółta kropka)',
                            markerfacecolor='yellow', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='Twój wynik (czerwona kropka)',
                            markerfacecolor='red', markersize=10)
                    ]
                    ax.legend(handles=legend_elements)
                    
                    st.pyplot(fig)

                    time_10km_diff=int((abs(exp_t_model["time_10km"]-exp_time["time_10km"])))
                    desp_of_10km(marathon_time_bool, time_10km_diff,expected_score)

            else:
                st.info("Nie jesteśmy w stanie porównać twoich wyników do innych")
        




    





    


