import numpy as np
from numpy import sum, isrealobj, sqrt
from numpy.random import standard_normal
import pandas as pd
import re
import random
import os
import threading
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
from scipy.io import wavfile
import torch
import time
import datetime
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from multiprocessing import Process
import warnings
warnings.filterwarnings('ignore')

# ## Определение констант и глобальных переменных
global MAIN_PATH
MAIN_PATH = os.getcwd()
global TXT_PATH
TXT_PATH = os.path.join(MAIN_PATH, 'txt')
if not os.path.isdir(TXT_PATH):
    print('Директория "txt" не обнаружена в текущем каталоге {}!'.format(MAIN_PATH))    
global result
result = []
global fs
fs = 8000 # частота дискретизации 
global ln
ln = 100

# Для синтеза
text = 'ЬБОХЗ'
f = [650, 700, 750] # имитируем допплеровские сдвиги вокруг 700
f_el = 50 # помеха 50 Гц
pts = 1000 * ln / fs
SNR = [np.inf, 20, 10, 0, -5, -10, -15, -20] 
RS = 333 # не надо менять!
random.seed(RS)
np.random.seed(RS)
PH_el = 8 # сдвиг по фазе для 50 Гц

global AMP
AMP = 30000

#source_samplerate = 48000 # что выдает панель
source_samplerate = fs # на чем учится нейронка
base_dot_lenght = .05 # 0.05 секунды
base_mask_len = np.int64(source_samplerate * base_dot_lenght) # базовый размер элемента маски

global rus_to_morse    
rus_to_morse = {'а': '.-',
                    'б': '-...',
                    'в': '.--',
                    'г': '--.',
                    'д': '-..',
                    'е': '.',
                    'ж': '...-',
                    'з': '--..',
                    'и': '..',
                    'й': '.---',
                    'к': '-.-',
                    'л': '.-..',
                    'м': '--',
                    'н': '-.',
                    'о': '---',
                    'п': '.--.',
                    'р': '.-.',
                    'с': '...',
                    'т': '-',
                    'у': '..-',
                    'ф': '..-.',
                    'х': '....',
                    'ц': '-.-.',
                    'ч': '---.',
                    'ш': '----',
                    'щ': '--.-',
                    'ъ': '.--.-.',
                    'ы': '-.--',
                    'ь': '-..-',
                    'э': '..-..',
                    'ю': '..--',
                    'я': '.-.-',
                    '0': '-----',
                    '1': '.----',
                    '2': '..---',
                    '3': '...--',
                    '4': '....-',
                    '5': '.....',
                    '6': '-....',
                    '7': '--...',
                    '8': '---..',
                    '9': '----.',
                    '.': '......',
                    ',': '.-.-.-',
                    ':': '---...',
                    ';': '-.-.-',
                    '(': '-.--.-', 
                    ')': '-.--.-', 
                    "'": '.----.',
                    '"': '.-..-.',
                    '-': '-....-',
                    '/': '-..-.',
                    '?': '..--..',
                    '!': '--..--',
                    '=': '-...-', 
                    '@': '.--.-.'}

global morse_to_rus       
morse_to_rus = {'.-': 'а',
                        '-...': 'б',
                        '.--': 'в',
                        '--.': 'г',
                        '-..': 'д',
                        '.': 'е',
                        '...-': 'ж',
                        '--..': 'з',
                        '..': 'и',
                        '.---': 'й',
                        '-.-': 'к',
                        '.-..': 'л',
                        '--': 'м',
                        '-.': 'н',
                        '---': 'о',
                        '.--.': 'п',
                        '.-.': 'р',
                        '...': 'с',
                        '-': 'т',
                        '..-': 'у',
                        '..-.': 'ф',
                        '....': 'х',
                        '-.-.': 'ц',
                        '---.': 'ч',
                        '----': 'ш',
                        '--.-': 'щ',
                        '.--.-.': 'ъ',
                        '-.--': 'ы',
                        '-..-': 'ь',
                        '..-..': 'э',
                        '..--': 'ю',
                        '.-.-': 'я',
                        '-----': '0',
                        '.----': '1',
                        '..---': '2',
                        '...--': '3',
                        '....-': '4',
                        '.....': '5',
                        '-....': '6',
                        '--...': '7',
                        '---..': '8',
                        '----.': '9',
                        '......': '.',
                        '.-.-.-': ',',
                        '---...': ':',
                        '-.-.-': ';',
                        '-.--.-': '|', # скобка, нет различия правая или левая
                        '.----.': "'",
                        '.-..-.': '"',
                        '-....-': '-',
                        '-..-.': '/',
                        '..--..': '?',
                        '--..--': '!',
                        '-...-': '=', # знак раздела
                        '........': 'ERR', # ошибка/перебой, число точек может варьироваться, они могут идти не в одном символе а с межсимвольными паузами
                        '.--.-.': '@',
                        '...-.-': 'END CONN', # конец связи "СК"
                        '.-.-.': 'END RADIO', # конец радио-обмена "АР"
                        '...---...': 'SOS',
                        ' ': ' '}

# ## Определение основных функций и процедур
# Функция добавления в сигнал белого гауссовского шума 
def awgn(s, SNRdB, L=1):
    gamma = 10**(SNRdB/10) #SNR to linear scale
    if s.ndim==1:# if s is single dimensional vector
        P=np.float64(L*sum(abs(s)**2)/len(s)) #Actual power in the vector
    else: # multi-dimensional signals like MFSK
        P=np.float64(L*sum(sum(abs(s)**2))/len(s)) # if s is a matrix [MxN]
    N0=P/gamma # Find the noise spectral density
    if isrealobj(s):# check if input is real/complex object type
        n = np.float64(sqrt(N0/2)*standard_normal(s.shape)) # computed noise
    else:
        n = np.float64(sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape)))
    r = np.float64(s + n) # received signal
    
    if SNRdB == np.inf:
        r = s # это я добавил для варианта чистого сигнала
        
    return r

# ### Примечание:
# В стандартном коде Морзе за единицу времени принимается длительность самого короткого сигнала — точки. Длительность тире равна трём точкам. Пауза между элементами одного знака — одна точка, между знаками в слове — 3 точки, между словами — 7 точек. Код может передаваться с любой доступной скоростью
# 
# <br>В азбуке Морзе отсутствуют такие символы как:
# * Точка
# * Тире
# * Запятая
# * Восклицательный знак
# * Вопросительный знак
# * Другие специальные символы 
# <br>Этот нюанс необходимо учесть. Программа должна уметь обрабатывать такие символы. Сделаем максимально просто — будем их игнорировать. Также, понятие строчной и заглавной буквы растворяются.

# Функция очистки слова от некодируемых в Морзе символов
def clean_stroke(content):
    #result = []
    content = str(content).lower()

    # удалил чтобы проходили спец-смволы
    #for element in content:
    #    if (ord(element) >= 1072 and ord(element) <= 1103) or (ord(element) >= 48 and ord(element) <= 57):
    #        result.append(element)
    
    # и мы просто возвращаем нижний кейс (НО вдруг все-таки придется чистить что-то еще)
    result = content
 
    return result

# функция перевода русского текста в Морзе
def russian_to_morse(content):
    content = re.split(' ', content) # режем тест на слова
    for w in range(len(content)):
        content[w] = ''.join(clean_stroke(content[w]))
        
    #print(content)    
        
    #content = clean_stroke(content)
    result = []
 
    for element in content:
        elm = []
        for ch in element:
            try:
                elm.append(rus_to_morse[ch])
            except:    
                elm.append('..--..') # это "?" - значит пытаемся передать символ, которого нет в алфавите
        
        result.append(elm)
 
    return result


global rus_symb
rus_symb = rus_to_morse.keys()

# ## Коррекция Морзе <a class="anchor" id="morse-vect"></a>
def to_bag(morse):
    tchk = 0
    tire = 0
    
    for m_c in morse:
        if m_c=='.':
            tchk += 1.0
        else:
            tire += 1.5 #1

    return np.array([tchk, tire], dtype=float)


def to_vect(morse):
    cnt = 0
    vect = np.zeros((len(morse)), dtype = float)
    
    for m_c in morse:
        if m_c=='.':
            vect[cnt] = 1.0
        else:
            vect[cnt] = 1.5 #3
        cnt += 1
        #print(m_c, m_c=='.')
        #print(vect)
    
    return vect


global morse_bags
morse_bags = {' ': np.array([0, 0], dtype=float)}
global morse_vect
morse_vect = {' ': np.array([0], dtype=float)}

for r_symb in rus_symb:
    m_code = rus_to_morse[r_symb]
    morse_bags[r_symb] = to_bag(m_code)
    morse_vect[r_symb] = to_vect(m_code)


def to_8(vect):
    dL = 8 - vect.shape[0]
    for _ in range(dL):
        vect = np.append(vect, np.array([0], dtype=float))
    return vect


def morse_correct_bag(err_symb):
    if len(err_symb)<=8:
        err_bag = to_bag(err_symb)
        err_vect = to_vect(err_symb)
        err_vect_8 = to_8(err_vect)

        evkl_rng_bag = []
        for r_symb in rus_symb:
            m_code = rus_to_morse[r_symb]
            evkl_rng_bag.append(np.linalg.norm(err_bag - to_bag(m_code), ord=2))

        evkl_rng_bag = pd.Series(evkl_rng_bag)
        min_evkl_rng_bag = evkl_rng_bag[evkl_rng_bag==evkl_rng_bag.min()].index.to_list()

        min_evkl_rng_bag_symb = []
        list_rus_symb = list(rus_symb)
        for i in min_evkl_rng_bag:
            r_symb = list_rus_symb[i]
            min_evkl_rng_bag_symb.append(r_symb)
    
    else:
        min_evkl_rng_bag_symb = '?..?'
        
    return min_evkl_rng_bag_symb


def morse_correct_bag2vect(min_evkl_rng_bag_symb, prm=2):
    if min_evkl_rng_bag_symb=='?..?':
        result = '?..?'
    else:    
        evkl_rng_vect = []

        for r_symb in min_evkl_rng_bag_symb:
            m_code = rus_to_morse[r_symb]
            m_code_vect = to_vect(m_code)
            m_code_vect_len = m_code_vect.shape[0]
            m_code_vect_8 = to_8(m_code_vect)
            num_var = 8 - m_code_vect_len + 1
            variants = []
            for _ in range(num_var):
                # попробуем пробежаться нулем влево
                nol = np.array([0], dtype=float)
                vvv = m_code_vect_8
                for i in range(7, -1, -1):
                    variants.append(np.linalg.norm(err_vect_8 - vvv, ord=prm))
                    left = m_code_vect_8[:i-1]
                    right = m_code_vect_8[i-1:7]
                    vvv = np.append(left, nol)
                    vvv = np.append(vvv, right)

                # это сдвиг вправо
                last = m_code_vect_8[7]
                m_code_vect_8 = m_code_vect_8[:7]
                m_code_vect_8 = np.append(last, m_code_vect_8)

            evkl_rng_vect.append(min(variants))

        evkl_rng_vect = pd.Series(evkl_rng_vect)
        min_evkl_rng_vect = evkl_rng_vect[evkl_rng_vect==evkl_rng_vect.min()].index.to_list()
        min_evkl_rng_bag_symb = pd.Series(min_evkl_rng_bag_symb)
        result = min_evkl_rng_bag_symb[min_evkl_rng_vect].tolist()

    return result


# Функция перевода Морзе в цифры, с заданным множителем длины
# 1 - точка
# 3 - тире
# 0 - растояние между точками и тире внутри буквы
# 000 - расстояние между буквами
# 0000000 - расстояние между словами

def morse_to_mask(morse, x_len):
    #print(morse)
    #print(len(morse))
    mask = []
    for w in range(len(morse)):
        #print('word', morse[w])
        for sym in morse[w]:
            #print('sym',sym)
            for c in sym:
                #print('c', c)
                if c == '.':
                    for _ in range(x_len):
                        mask.append(1)

                else:
                    for _ in range(x_len):
                        mask.append(1)
                        mask.append(1)
                        mask.append(1)

                for _ in range(x_len):
                    mask.append(0) # добавляем после точки или тире

            for _ in range(x_len * 2): # т.к. одну уже добавили
                mask.append(0) 

        for _ in range(x_len * 4): # т.к. одну после буквы и 2 после слова уже добавили из 7ми
            mask.append(0)                
                
    return mask


# генератор случайных чисел с гауссовким распределением и заданным разбросом
def gauss_rand(n_rand=2, razbros=1, tochn=3):
    res = 0
    for i in range(n_rand):
        res += random.random()
    
    res -= n_rand / 2
    
    return np.round(res * razbros, tochn)


# имитация ввода точек и тире радистом
def human_radist(dot_time, razbros):
    return np.int16(dot_time * (1 + gauss_rand(n_rand=4, razbros=razbros, tochn=3)))

# ## Подгружаем нейронку
# ## Актуальная нейронка <a class="anchor" id="first-bullet"></a>
from nn_models import New_Morse_Signal_Level_k17_8k_Deep_Rolling_BNorm

global net_morse
N_class = 5
N_CH = 16
net_morse = New_Morse_Signal_Level_k17_8k_Deep_Rolling_BNorm(n_input_channels=N_CH, n_class=N_class, act=torch.nn.Mish())
net_morse = torch.load('Morse_Signal_Level_k17_8k_Deep_16_Bnorm.md')
device = torch.device('cpu')
net_morse = net_morse.to(device)
net_morse.eval()


# # Управление потоками выполнения кода
# ### Потоки:
# 0. постоянное генерирование кусков сигнала морзянки с помехами и музыкой, запись их в базу данных (из текстового файла, где строки - сообщения). Берутся случайные строки, случайные помехи, случайная музыка, случайные паузы. Весь сценарий генерируется в самом начале (до потока) и выполняется после запроса параметров (длительность, наличие музыки, наличие помех ВЧ, наличие помех НЧ).
# 1. постоянный поиск начала морзянки в базе данных (если уперлись в последнюю запись, то ждем немного чтобы база наполнилась), инициирование потоков распознавания
# 2. потоки распознавания текста
# 3. если is_radio = True - то данные в базу данных пишет отдельный поток, который принимает их от панели по UDP
def get_scenario():
    answer = True
    
    nn_audio_carrier = 100 # 100 Гц - частота аудионесущей образцов нейронной сети НЕ ИСПОЛЬЗУЕТСЯ
    audio_carrier = nn_audio_carrier * 7 # используемая аудионесущая 700 Гц
    #audio_carrier = 1000
    #audio_carrier = nn_audio_carrier * 1 # используемая аудионесущая 100 Гц
    
    #signal_lenght = 60 * 60 # 60 минут по 60 секунд (для радио-трансляции)
    signal_lenght = 15 * 60 # 15 минут по 60 секунд (для имитации обнаружения)
    #base_signal_pause_lenght = 1 * 60 # 1 минут по 60 секунд
    #base_signal_pause_lenght = .25 * 60 # 15 секунд (для имитации обнаружения)
    base_signal_pause_lenght = 1 # 1 секунда (для радио-трансляции)
    #base_signal_pause_lenght = .1 # 1/10 секунды (для синтеза данных)
    pause_scatter = .5 # разброс 50%
    
    #base_dot_lenght = .075 # 0.075 секунды
    base_dot_lenght = .05 # 50 милисекунд
    radist_scatter = 0.20 # разброс 20%
    
    use_music = False
    #use_music = True
    use_lf_noise = False
    use_hf_noise = True
    #use_hf_noise = False
    #use_damping = True
    use_damping = False
    use_jitter = False
    use_other_frequency = False
    
    is_radio = False # т.е. подключен ли рабиоприемник... похоже этот параметр устарел
    is_play = False
    #is_play = True
    rnd_msg = True # (для имитации обнаружения)
    #rnd_msg = False # (для радио-трансляции)
    
    #source_samplerate = 48000 # что выдает панель
    source_samplerate = 40000
    #source_samplerate = 8000
    nn_samplerate = fs # на чем обучена нейронка (было 6000)
    rnd_snr = False # случайно выбираем SNR?
    snr = -10 # соотношение сигнал-шум в дБ, если выбирается случайно, то не имеет значения

    base_mask_len = np.int64(source_samplerate * base_dot_lenght) # базовый размер элемента маски
    
    resample_ratio = source_samplerate // nn_samplerate
    signal_quant = 16384
    signal_quant_time = signal_quant / source_samplerate
    signal_quant_for_NN = signal_quant // resample_ratio
        
    def show_param():
        print('======= ОБЩИЕ ПАРАМЕТРЫ =======')
        print('Принимаем сигнал с радиостанции:', is_radio)
        print('Проигрываем исходный сигнал:', is_play)
        print('Отправляем сообщения случайным образом:', rnd_msg)
        print('')

        print('======= ПАРАМЕТРЫ СИГНАЛА =======')
        print('Длительность сигнала (в минутах):', signal_lenght / 60)
        print('Базовая длительность пауз между фрагментами сигнала (в минутах):', base_signal_pause_lenght / 60)
        print('Разброс пауз между фрагментами сигнала: {}%'.format(np.round(pause_scatter * 100, 2)))
        print('Используем случайно выбранную музыку в качестве помехи:', use_music)
        print('Используем шум 50 Гц в качестве помехи:', use_lf_noise)
        print('Используем белый гауссовский шум в качестве помехи:', use_hf_noise)
        print('Используем случайно выбранное затухание сигнала Морзе:', use_damping)
        print('Используем случайное дрожжание аудио-несущей в диапазоне плюс-минус 10%:', use_jitter)
        print('Используем случайный множитель частоты аудио-несущей (100 - 1000 Гц):', use_other_frequency)
        print('Заданная частота аудионесущей: {} Гц'.format(audio_carrier))
        print('Частота дискретизации источника сигнала: {} Гц'.format(source_samplerate))
        print('Частота дискретизации образцов для нейронной сети: {} Гц'.format(nn_samplerate))
        print('Коэффициент ресемплинга:', resample_ratio)
        print('Случайный выбор соотношений сигнал/шум:', rnd_snr)
        print('Заданное соотношение сигнал/шум: {} дБ'.format(snr))
        print('')

        print('======= ПАРАМЕТРЫ РАДИСТА =======')
        print('Базовая длительность точки в коде Морзе (в секундах):', base_dot_lenght)
        print('Разброс длительностей точек и тире в коде Морзе: {}%'.format(np.round(radist_scatter * 100, 2)))
        print('')
    
    # тут распишем все дефолтные параметры
    print('======= ПАРАМЕТРЫ ПО УМОЛЧАНИЮ =======') 
    show_param()

    if str(input('Используем параметры по умолчанию? (1 = ДА): '))!='1':
        print('======= ОБЩИЕ ПАРАМЕТРЫ =======')
        is_radio = str(input('Принимаем сигнал с радиостанции? (1 = ДА): '))=='1' # если нет, то пишем в базу локально
        is_play = str(input('Проигрываем исходный сигнал? (1 = ДА): '))=='1' # если нет, то не выводим сигнал и НЕ делаем паузу для его проигрывания!
        rnd_msg = str(input('Отправляем сообщения случайным образом? (1 = ДА, если НЕТ, то сообщения будут отправлены из файла по-строчно): '))=='1'
        print('')
        
        print('======= ПАРАМЕТРЫ СИГНАЛА =======')
        signal_lenght = np.int32(np.float64(input('Длительность сигнала (в минутах): ')) * 60) # мы внутри оперируем секундами
        base_signal_pause_lenght = np.int32(np.float64(input('Базовая длительность пауз между фрагментами сигнала (в минутах): ')) * 60) # от этого будет делать рэндом
        pause_scatter = np.float64(input('Разброс пауз между фрагментами сигнала (в %): ')) / 100
        use_music = str(input('Используем случайно выбранную музыку в качестве помехи? (1 = ДА): '))=='1'    # используем случайную музыку
        use_lf_noise = str(input('Используем шум 50 Гц в качестве помехи? (1 = ДА): '))=='1'    # используем НЧ шум 50 Гц
        use_hf_noise = str(input('Используем белый гауссовский шум в качестве помехи? (1 = ДА): '))=='1'    # используем случайный ВЧ шум
        use_damping = str(input('Используем случайно выбранное затухание сигнала Морзе? (1 = ДА): '))=='1'    # использовать случайное затухание
        use_jitter = str(input('Используем случайное дрожжание аудио-несущей в диапазоне плюс-минус 10%? (1 = ДА): '))=='1'    # использовать случайное дрожжание аудио-несущей 10%
        use_other_frequency = str(input('Используем случайный множитель частоты аудио-несущей (100 - 1000 Гц)? (1 = ДА): '))=='1'    # использовать случайный множитель частоты аудио-несущей (100 - 1000 Гц)
        audio_carrier = np.int32(input('Заданная частота аудионесущей (в Гц): '))
        source_samplerate = np.int32(input('Частота дискретизации источника сигнала (в Гц): '))
        nn_samplerate = np.int32(input('Частота дискретизации образцов для нейронной сети (в Гц): '))
        rnd_snr = str(input('Случайный выбор соотношений сигнал/шум? (1 = ДА): '))=='1'
        snr = np.int8(input('Заданное соотношение сигнал/шум (в дБ): '))
        print('')
        
        print('======= ПАРАМЕТРЫ РАДИСТА =======')
        base_dot_lenght = np.float64(input('Базовая длительность точки в коде Морзе (в секундах): ')) # от этого будем вычислать длительность тире и пауз
        radist_scatter = np.float64(input('Разброс длительностей точек и тире в коде Морзе (в %): ')) / 100
        print('')

        base_mask_len = np.int64(source_samplerate * base_dot_lenght) # базовый размер элемента маски
    
        resample_ratio = source_samplerate // nn_samplerate
        signal_quant_time = signal_quant / source_samplerate
        signal_quant_for_NN = signal_quant // resample_ratio
        
        # просим подтвердить
        print('======= ЗАДАННЫЕ ПАРАМЕТРЫ =======') 
        show_param()
       
    if str(input('Работаем с этими параметрами? (1 = ДА): '))!='1':
        answer = False

    scenario = {
        'is_radio': is_radio,
        'is_play': is_play,
        'rnd_msg': rnd_msg,
        'signal_lenght': signal_lenght,
        'base_signal_pause_lenght': base_signal_pause_lenght,
        'pause_scatter': pause_scatter,
        'use_music': use_music,
        'use_lf_noise': use_lf_noise,
        'use_hf_noise': use_hf_noise,
        'use_damping': use_damping,
        'use_jitter': use_jitter,
        'use_other_frequency': use_other_frequency,
        'audio_carrier': audio_carrier,
        'source_samplerate': source_samplerate,
        'nn_samplerate': nn_samplerate,
        'resample_ratio': resample_ratio,
        'rnd_snr': rnd_snr,
        'snr': snr,
        'base_dot_lenght': base_dot_lenght,
        'base_mask_len': base_mask_len,
        'radist_scatter': radist_scatter,
        'signal_quant': signal_quant,
        'signal_quant_time': signal_quant_time,
        'signal_quant_for_NN': signal_quant_for_NN,
        'nn_audio_carrier': nn_audio_carrier
            }
    
    return answer, scenario

# Возможно надо делать еще наложение 2х разных кусков морзянки с разной несущей и амплитудой...

answer, scenario = get_scenario()

while not answer:
    answer, scenario = get_scenario()

# получаем список сообщений из файла с сообщениями
messages_file_name = os.path.join(TXT_PATH, 'messages.txt')

with open(messages_file_name, "r") as messages_file:
    messages = messages_file.readlines()
    print(messages)

# очищаем сообщения от символов конца строки
for i in range(len(messages)-1):
    messages[i] = messages[i][:-1]
print(messages)    

# ### Код потоков
def to_complex_mult_opor_simple(data, f, length, fs):
    
    t = length / fs
    samples = np.linspace(0, t, int(fs*t), endpoint=False)
    
    conj_opor_sm = []
    smesh = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -1*np.pi/4, -1*np.pi/2, -3*np.pi/4]

    for sm in smesh:
        opor_sm = np.exp(1j * (2 * np.pi * f * samples + sm))
        conj_opor_sm.append(np.conj(opor_sm))
    
    # для каждого сэмпла нам надо получить комплексное представление
    d = []
    c = data[0] + data[1] * 1.0j
    
    for iii in range(len(conj_opor_sm)):
            d.append(c * conj_opor_sm[iii])
    
    d = np.array(d)
    print('Перевели сигнал в комплексное представление:', d.shape)
    
    return d


def low_pass_filter(data, window=12, number_of_stdevs_away_from_mean=3):
    df = pd.DataFrame(data=data, columns=['value'])
    # rolling average
    df['Rolling_Average']=df['value'].rolling(window=window, center=True).mean()
    return df


def create_new_features(data, f, length, fs):
    d = to_complex_mult_opor_simple(data=data, f=f, length=length, fs=fs)
    del data
    
    ra_r = []
    rs_r = []
    ra_i = []
    rs_i = []
    
    rzm = d.shape[0]
    
    # сначала добавим то что без сдвига по фазе а потом - со сдвигом
    for i in range(rzm): # т.к. 6 слоев!
        fltrd = low_pass_filter(data=d[i].real, window=200, number_of_stdevs_away_from_mean=3)
        fltrd_im = low_pass_filter(data=d[i].imag, window=200, number_of_stdevs_away_from_mean=3)

        # дропнем края где нет значений оконно йфункции - это всего по 1му кванту спереди и сзади, может быть потом заполнять будем
        fltrd.dropna(inplace=True)
        fltrd_im.dropna(inplace=True)
        
        # берем абсолютную часть чтобы импульсы всехда были положительными
        fltrd['Rolling_Average'] = fltrd['Rolling_Average'].apply(np.abs)
        fltrd_im['Rolling_Average'] = fltrd_im['Rolling_Average'].apply(np.abs)
        
        ra_r.append(fltrd['Rolling_Average'].tolist())
        ra_i.append(fltrd_im['Rolling_Average'].tolist())

    del d
    
    ra_r = np.array(ra_r)
    ra_i = np.array(ra_i)
    
    # соберем все вместе
    data = np.array(ra_r)
    data = np.append(data, ra_i, axis = 0)
    
    del ra_r, ra_i, rs_r, rs_i #, md
    
    print('Сформировали новые признаки:', data.shape)
    
    return data


# создание аудио-несущей    
def create_audio_carrier(morse_lenght, source_samplerate, nn_samplerate, frequency):
    audio_carrier = np.zeros((2, morse_lenght), dtype='float64', order='C') 

    # действительная часть аудио-несущей, вычисляем длительность по длине маски
    t = morse_lenght / source_samplerate
    samples = np.linspace(0, t, morse_lenght, endpoint=False)
    z = np.exp(1j * 2 * np.pi * frequency * samples)
    audio_carrier[0] = z.real
    audio_carrier[1] = z.imag
        
    return audio_carrier


# модулируем аудио-несущую морзянкой
def modulate(sign, msk):
    result = np.zeros((sign.shape[0], sign.shape[1]), dtype='float64', order='C') 
    msk = np.array(msk)
        
    for ii in range(sign.shape[0]):
        result[ii] = np.multiply(sign[ii], msk)

    return result


# добавляем шум
def add_noise(sign, snr):
    noise = np.zeros((sign.shape[0], sign.shape[1]), dtype='int16', order='C')
    
    for i in range(sign.shape[0]):
        #sign[i] = sign[i] / 32767
        # нормируем сигнал по максимальным значениям
        n = [abs(min(sign[i])), abs(max(sign[i]))]
        print('Max sign:', max(n))
        if max(n) > 0.0:
            tmp = np.float64(sign[i] / max(n))
            
        #noise[i] = awgn(s=sign[i], SNRdB=snr, L=1)
        tmp = awgn(s=tmp, SNRdB=snr)

        # динамический диапазон
        print('Минимум у сигнала {}:'.format(i), min(tmp))
        print('Максимум у сигнала {}:'.format(i), max(tmp))
            
        # Функция добавления шума выбила нас из диапазона -1..+1, вернемся к нему
        n = [abs(min(tmp)), abs(max(tmp))]
        print('Max noise:', max(n))
        if max(n) > 0.0:
            tmp = tmp / max(n)
                        
        tmp = tmp * AMP
        noise[i] = np.int16(tmp)
        
    print('Шум:', noise.shape, type(noise[0, 0]))
        
    return noise #.astype(np.int16)


# добавляем оффсеты спереди и сзади
def add_offset(sign, bspl, ps, sr):
    bspl = bspl * sr # получаем базовую длительность в сэмплах из секунд
    front_off_len = np.int64(bspl * (1 + gauss_rand(n_rand=2, razbros=ps, tochn=4))) # длительность переднего оффсета
    print('Длительность переднего оффсета:', front_off_len)
    back_off_len = np.int64(bspl * (1 + gauss_rand(n_rand=2, razbros=ps, tochn=4))) # длительность заднего оффсета
    print('Длительность заднего оффсета:', back_off_len)
    front_off = np.zeros(front_off_len) # сгенерировали оффсет спереди 
    back_off = np.zeros(back_off_len) # сгенерировали оффсет сзади         

    result = np.zeros((sign.shape[0], sign.shape[1]+front_off_len+back_off_len), dtype='int16', order='C') 
        
    for iii in range(sign.shape[0]):
        tmp = np.concatenate((front_off, sign[iii]))
        result[iii] = np.concatenate((tmp, back_off))
        
    return result


# ### Код потоков <a class="anchor" id="code-thread"></a>
def morse_translation_generator(scenario, music, messages, param):
    print(threading.enumerate())
    
    counter = -1
    detect_thread = []
    detect_thread_em1 = [] # эмуляция другого детектирования
    detect_thread_em2 = []

    # детектирование морзянки - НАДО БУДЕТ вынести это в отдельный блок с чтением из БД
    def morse_detect(counter, signal, message, fs, resample_ratio, audio_carrier):
        if counter!=0:
            detect_thread[counter-1].join()
            
        print('Начинаем детектирование сообщения номер:', counter)
        signal = np.moveaxis(signal, 0, 1)
        print('Размерность сигнала:', signal.shape)
        signal = signal[:, ::resample_ratio]
        print('Размерность сигнала после децимации:', signal.shape)
        
        #Фильтр с обратным преобразованием Фурье
        #filter = False # фильтр нам сейчас только вредит
        filter = True
        #f = 700 # частота аудионесущей
        f = audio_carrier
        df = 100 # 100 - было норм # ставил 200, 150 - ухудшилось, 50 вроде тоже не очень, надо еще и 150 попробовать т.к. полоса вроде бы 300
        if filter:
            yf_music = rfft(signal[0])
            xf_music = rfftfreq(len(signal[0]), 1/fs)
            yf_music_im = rfft(signal[1])
            xf_music_im = rfftfreq(len(signal[1]), 1/fs)

            # Отфильтруем НЧ до 600 Гц и ВЧ начиная с 800 Гц
            # Максимальная частота составляет половину частоты дискретизации
            points_per_freq = len(xf_music) / (fs / 2)

            # Наша целевая частота - 700 Гц - вырезаем кусок 2 х df вокруг нее
            min_target_idx = int(points_per_freq * (f-df))
            max_target_idx = int(points_per_freq * (f+df))

            # Удалим вообще все что вне диапазона
            yf_music[:min_target_idx] = 0
            yf_music[max_target_idx:] = 0        
            yf_music_im[:min_target_idx] = 0
            yf_music_im[max_target_idx:] = 0         

            tmp = irfft(yf_music)
            tmp_im = irfft(yf_music_im)

            # Надо выровнять длины
            del signal
            #d = [signal.shape[1], tmp.shape[0], tmp_im.shape[0]]
            d = [tmp.shape[0], tmp_im.shape[0]]
            d_min = min(d) - 1

            tmp = tmp[:d_min]
            tmp_im = tmp_im[:d_min]

            signal = np.zeros((2, d_min), dtype='float64', order='C')

            signal[0] = tmp
            signal[1] = tmp_im

            del tmp, tmp_im

            #signal = np.float64(signal / 32767)
            # После фильтрации мы вышли из диапазона -1..+1, вернемся к нему
            n = [abs(min(signal[0])), abs(max(signal[0])), abs(min(signal[1])), abs(max(signal[1]))]
            if max(n) > 0.0:
                signal = np.float64(signal / max(n))
        # конец фильтра
        
        # создадим новые признаки и перезапишем их в сигнал (первичные при этом теряем)
        signal = create_new_features(data=signal, f=f, length=signal.shape[1], fs=fs)
        
        
        # начинаем распознавать
        sps = []
        veroyatn = []
        # вариант с циклом
        cycle_dec = False
        if cycle_dec:
            signal = np.expand_dims(signal, axis = 0)
            print('Размерность сигнала после добавления нового измерения:', signal.shape)
            
            signal = torch.from_numpy(signal)
            signal = signal.float()
            signal = torch.FloatTensor(signal)
            signal = signal.to(device)
            
            for iii in range(signal.shape[2] // ln - 0):
                sss = signal[:, :, 0+ln*iii+0:ln+ln*iii+0]
                try:
                    pred_morse = net_morse.inference(sss).detach().cpu()
                except:
                    pred_morse = net_morse.forward(sss).detach().cpu()

                p_morse = pred_morse.argmax(dim=1).numpy().tolist()[0][0]
                #p_morse = pred_morse.numpy().tolist()[0][0]
                sps.append(p_morse)

                veroyatn_morse = pred_morse.numpy().tolist()[0]
                veroyatn.append({'low': veroyatn_morse[0][0],
                                 'up': veroyatn_morse[1][0],
                                 'top': veroyatn_morse[2][0],
                                 'down': veroyatn_morse[3][0],
                                 '-': veroyatn_morse[4][0]})
        
        else: # вариант без цикла (с батчем)
            print('Размерность сигнала до решейпинга:', signal.shape)
            signal = np.moveaxis(signal, 0, 1) # чтобы потом корректно зарешейпить
            print('Передвинули оси:', signal.shape)
            cut_size = int((signal.shape[0]//ln)*ln)
            print('Режем по:', cut_size)
            signal = signal[:cut_size]
            batch_size = int(signal.shape[0]//ln)
            print('Размер батча:', batch_size)
            signal = signal.reshape(((batch_size, ln, 16)))
            signal = np.moveaxis(signal, 1, 2) # должно получиться: кол-во сэмплов, 2, ln

            signal = torch.from_numpy(signal)
            signal = signal.float()
            signal = torch.FloatTensor(signal)
            signal = signal.to(device)

            print('Размерность сигнала после решейпинга:', signal.shape)

            try:
                pred_morse = net_morse.inference(signal).detach().cpu()
            except:
                pred_morse = net_morse.forward(signal).detach().cpu()

            sps = pred_morse.argmax(dim=1).numpy().reshape((pred_morse.shape[0],)).tolist()        
        # конец варианта без цикла (с батчем)
        
        print('Уровни в Сообщении {} распознаны'.format(counter))    
        #fs = 8000 # ЭТО НАДО ПОТОМ ВЫНЕСТИ В ПАРАМЕТРЫ
        quant_time = ln / fs

        # сначала на основе знаний о фронтах и спадах восстановим верхний и низкий уровни, а потом (c фронтами и спадами) скормим его в следующий алгоритм подсчета длительностей
        # низкий уровень сигнала
        # верхний уровень сигнала
        # 1 подъем наверх
        # 3 - спад вниз
        # между 1 и 3 все должно быть заполнено 2
        # между 3 и 1 - заполнено нулями
        # если вдруг после 1 мы снова поймали 1, то ищем последнюю 2 до второй 1 и до нее все заполняем 2, а в конце ставим 3
        # если вдруг после 3 мы снова поймали 3, то ищем последний 0 до второй 3 и до нее все заполняем 0, а в конце ставим 1
        new_fall = 0 # новый фронт/спад
        new_fall_n = 0 # номер нового фронта/спада
        last_fall = 0 # последний до этого фронт/спад
        last_fall_n = 0 # номер последнего до этого фронта/спада
        len_sps = len(sps)
        tmp = last_fall
        tmp_n = last_fall_n
        
        # для начала нам надо вырезать кусок с морзянкой
        print('Вырезаем кусок с морзянкой...')
        for i in range(len_sps):
            if sps[i] > 0:
                begin_morse = i
                break
                
        for i in range(len_sps-1, -1, -1):
            if sps[i] > 0:
                end_morse = i + 30 # чтобы захватить в конце пробел после слова и не терять последнюю букву
                break
        
        sps = sps[begin_morse:end_morse]
        len_sps = len(sps)
        
        # смотрим % фронтов от общего количества распознаваний
        print('Считаем долю распознаваний фронтов/спадов:')
        df = pd.DataFrame(data=sps, columns=['znach'])
        n_z0 = df.query('znach==0')['znach'].count()
        n_z1 = df.query('znach==1')['znach'].count()
        n_z2 = df.query('znach==2')['znach'].count()
        n_z3 = df.query('znach==3')['znach'].count()
        z13_rate = (n_z1 + n_z3) / (n_z0 + n_z1 + n_z2 + n_z3)
        print(np.round(z13_rate*100, 2),'%')
        #идем восстанавливать фронты/спады только если их доля больше 5%
        
        do_it = False  # пока отключил чтобы не мешался
        if (z13_rate>.05) and (z13_rate<.4) and (do_it==True): # в противном случае 1 и 3 только портят картину
            print('Восстанавливаем фронты/спады...')
            # ищем самый первый фронт, если до него ВДРУГ был спад, то пытаемся восстановить импульс
            for i in range(len_sps):
                if sps[i]==3:
                    sps[i-1] = 1
                    last_fall_n = i - 1
                    break
                elif sps[i]==1:
                    last_fall = 1
                    last_fall_n = i
                    break
                elif sps[i]==2:
                    sps[i] = 1
                    last_fall_n = i
                    break
                    
            # теперь уже, зная место первого фронта, идем сразу от следующего за ним
            n = last_fall_n + 1
            tmp = last_fall
            tmp_n = last_fall_n

            # сначала обработаем разорванные импульсы, а потом уже нормальные
            for nn in range(n, len_sps, 1):
                if (last_fall==1 and sps[nn]==1) or (last_fall==3 and sps[nn]==3): #поймали НЕОЖИДАННОЕ начало/конец нового импульса
                    if last_fall==1 and sps[nn]==1:
                        for iii in range(nn-1, last_fall_n, -1):
                            if sps[iii]==2 or sps[iii]==1:
                                sps[iii+1] = 3
                                last_fall = 3
                                last_fall_n = iii+1
                                break
                    elif last_fall==3 and sps[nn]==3:
                        for iii in range(nn-1, last_fall_n, -1):
                            if sps[iii]==0 or sps[iii]==3:
                                sps[iii+1] = 1
                                last_fall = 1
                                last_fall_n = iii+1                            
                                break  
                    continue
                elif sps[nn]==1 or sps[nn]==3:
                    last_fall = sps[nn]
                    last_fall_n = nn
                
        
            # теперь все заново но уже - заполняем
            print('Восстанавливаем импульсы...')
            last_fall = tmp
            last_fall_n = tmp_n

            for nn in range(n, len_sps, 1):
                if last_fall==1 and sps[nn]==3 and nn-last_fall_n>1: #поймали конец импульса
                    new_fall = sps[nn]
                    new_fall_n = nn
                    for nnn in range(new_fall_n-1, last_fall_n, -1):
                        sps[nnn] = 2 # все заполняем двойками
                    last_fall = new_fall
                    last_fall_n = new_fall_n
                elif (last_fall==3 and sps[nn]==1) and nn-last_fall_n>1: #поймали конец паузы
                    new_fall = sps[nn]
                    new_fall_n = nn
                    for nnn in range(new_fall_n-1, last_fall_n, -1):
                        sps[nnn] = 0 # все заполняем нулями
                    last_fall = new_fall
                    last_fall_n = new_fall_n        
        
        # считаем грязные длительности
        print('Считаем длительности...')
        dlitelnosti = []
        znach = 0
        dlit = 0.0
        for i in range(len_sps):
            if sps[i] == 1:
                sps[i] = 2 # тк это фронт импульса (1.5 в корне неверно ставить!)
            elif sps[i] == 3:
                sps[i] = 2 # тк это спад импульса    
            elif sps[i] == 4:
                sps[i] = 0 # а это вообще мусор!
            if sps[i] == znach:
                dlit = dlit + quant_time * (sps[i] - 1)
            else:
                dlitelnosti.append(dlit)
                dlit = 0.0
                znach = sps[i]
                dlit = dlit + quant_time * (sps[i] - 1)

        dlitelnosti.append(dlit)          

        print('Длительности в Сообщении {} посчитаны'.format(counter))        
        
        # очищаем от нулей и длинных пауз больших чем 1 сек - НЕТ ! так мы паузы после сообщений убиваем а они нам нужны
        # но сначала приведем длинные паузы к 0.9
        for ii in range(len(dlitelnosti)):
            if dlitelnosti[ii] < -0.4:
                dlitelnosti[ii] = -0.4 # это значение требует подбора !!!
            elif dlitelnosti[ii] > 0.2:
                dlitelnosti[ii] = 0.2 # это значение требует подбора !!!    
        
        cleaned_dlit = []
        for dlit in dlitelnosti:
            if dlit > -1.0 and dlit < 0.5 and dlit != 0.0: # поставил верхнюю отсечку на 0.5 - это значение требует подбора !!!
                cleaned_dlit.append(dlit)

        # скармливаем длительности в алгоритм К-срдених
        n_dlit_clust = 5
        model = KMeans(n_clusters=n_dlit_clust, max_iter=300, tol=0.0001, verbose=0, random_state=42, algorithm='elkan')
        cleaned_dlit = np.array(cleaned_dlit)
        cleaned_dlit = np.reshape(cleaned_dlit, (cleaned_dlit.shape[0], 1))
        
        try:
            model.fit(cleaned_dlit)
            print('Метод К-срдених в Сообщении {} отработал'.format(counter)) 
        except:
            print('ОШИБКА!!! Работа метода К-средних в Сообщении {} невозможна!'.format(counter))
            result.append(None)
        
        # поймем что из этого точки, тире и разные паузы
        cleaned_dlit = pd.DataFrame(data=cleaned_dlit, columns=['dlit'])
        cleaned_dlit['class'] = model.labels_
        
        print('Cообщение {}. Диапазоны длительностей уровней сигнала морзянки в зашумленном эфире'.format(counter))
        cent_dlit = []
        for i in range(n_dlit_clust):
            tmp = cleaned_dlit[cleaned_dlit['class']==i]
            print(i, ':', np.round(tmp['dlit'].min(), 2), '-', np.round(tmp['dlit'].max(), 2))
            cent_dlit.append(np.round(tmp['dlit'].mean(), 2))
        print(cent_dlit)   
        
        cent_dlit = pd.DataFrame(data=cent_dlit, columns=['cent_dlit'])
        cent_dlit = cent_dlit.sort_values(by='cent_dlit', ascending=True)
        cent_dlit.reset_index(drop=False, inplace=True)
        print(cent_dlit)
        
        morzyanka = []
        symb = []
        
        code_dlit = cent_dlit['index'].tolist() # получаем упорядоченный по возрастанию список кодов классов длительноcтей
        
        for dlit_class in cleaned_dlit['class']:
            if dlit_class == code_dlit[1]:
                txt_symb = ''.join(symb)
                morzyanka.append(txt_symb)
                symb = []
            elif dlit_class == code_dlit[4]:
                symb.append('-')
            elif dlit_class == code_dlit[3]:
                symb.append('.')
            elif dlit_class == code_dlit[0]:
                txt_symb = ''.join(symb)
                morzyanka.append(txt_symb)
                morzyanka.append(' ')
                symb = []
            else:
                pass
        
        
        print('Декодировано:', ' '.join(morzyanka))
        
        dec_txt = []
        for chr in morzyanka:
            try:
                dec_txt.append(morse_to_rus[chr])
            except:
                # старый метод
                #dec_txt.append('?')
                #if len(chr)>6:
                #    dec_txt.append('..?')
                # новый метод
                pred_morse_vect = morse_correct_bag2vect(morse_correct_bag(err_symb))
                if pred_morse_vect=='?..?':
                    dec_txt.append('?..?')
                else:
                    if len(pred_morse_vect)>1:
                        pred_morse_vect = pred_morse_vect[0] # пока для простоты берем 1й вариант
                    dec_txt.append(pred_morse_vect)    

        morse_text = ''.join(dec_txt)

        # Надо бы ПРАВИЛЬНО посчитать ошибки
        print('==============================================================')
        print('Распознано сообщение:', morse_text)
        print('Исходное сообщение:', message)
        print('==============================================================')
        result.append(morse_text)
        
        return 'OK'

    # тут были записи в файлы и БД, проигрывание которое нам в этой версии не актуально
    
    # Основная часть    
    time_count = 0 # счетчик времени
    num_msg = -1 # счетчик сообщений
    message_id = 0
    
        
    while time_count < scenario['signal_lenght']:
        message_id += 1
        # создаем кусок эфира
        
        if scenario['rnd_msg']:
            # случайно выбираем текстовое сообщение
            num_msg = np.int16(random.random() * len(messages))
        else:
            num_msg += 1 #0 # пока беру 10:1
        
        try:
            message = messages[num_msg]
        except:
            print('Сообщения закончились, Милорд!')
            break # это значит что сообщения закончились и надо сваливать из цикла
            
        print('Прошло времени от начала эфира: {} сек.'.format(time_count))
        print('Идёт передача сообщения:', message)
        morse_message = russian_to_morse(message)
        print('Код Морзе:', morse_message)
        base_target_mask = morse_to_mask(morse=morse_message, x_len=1)[:-4]
        print('Базовая маска ключа:', base_target_mask)
        print('')
        
        # по базовой маске, длительности и разбросу готовим итоговую маску (которая в т.ч. и определит длительность)
        mask = []
        for m in base_target_mask:
        #    for mask_dlit in range(human_radist(dot_time=scenario['base_mask_len'], razbros=scenario['radist_scatter'])):
        #        mask.append(m)
            mask_dlit = [m] * human_radist(dot_time=scenario['base_mask_len'], razbros=scenario['radist_scatter'])
            mask = mask + mask_dlit
        
        morse_lenght = len(mask)
        
        # создаем аудио-несущую
        signal = create_audio_carrier(morse_lenght=morse_lenght, 
                                      source_samplerate=scenario['source_samplerate'], 
                                      nn_samplerate=scenario['nn_samplerate'],
                                      frequency=scenario['audio_carrier'])
        
        print('Аудио-несущая:', signal.shape)
        
        # модулируем аудио-несущую маской морзянки
        mod_signal = modulate(sign=signal, msk=mask)
        del signal
        
        # приводим к формату int16
        mod_signal *= AMP
        mod_signal = np.int16(mod_signal)
        
        print('После модуляции:', mod_signal.shape)
        
        # добавляем к чистой морзянке оффсеты спереди и сзади
        mod_signal = add_offset(sign=mod_signal, 
                                bspl=scenario['base_signal_pause_lenght'], 
                                ps=scenario['pause_scatter'],
                                sr=scenario['source_samplerate'])
        
        print('С оффсетами:', mod_signal.shape)

        # добавим белый шум (если надо) - ПЕРЕНЕСЛИ ЭТО ДО МУЗЫКИ !
        if scenario['use_hf_noise']:
            if scenario['rnd_snr']: # случайный ли SNR ?
                snr = np.int16(40 * random.random() - 15)
            else:
                snr = scenario['snr']
            print('SNR:', snr)
            mod_signal = add_noise(sign=mod_signal, 
                                   snr=snr)        
        
        
        print('Тип:', type(mod_signal[0, 0]))    
 
        #cut_sign = False
        #cut_sign = True
        cut_sign = param['cut_sign']
        if cut_sign:
            frag_size = scenario['signal_quant'] * 4
            n_quant = mod_signal.shape[1] // frag_size
            frag_sign = mod_signal[:, :(frag_size * n_quant)] # обрезали сигнал кратно квантам
            del mod_signal

            print('После обрезки:', frag_sign.shape)
        else:
            frag_sign = mod_signal
        
        # переставляем оси
        n_chnl = frag_sign.shape[0]
        frag_sign = np.moveaxis(frag_sign, 0, 1)
        print('После перестановки осей:', frag_sign.shape)

        frag_len = np.int64(frag_sign.shape[0] // scenario['source_samplerate'])
        counter += 1
        
        # распознаем сообщение - это надо вытащить в отдельный поток
        #detect = False
        #detect = True
        detect = param['detect']
        if detect:
            detect_thread.append(threading.Thread(target=morse_detect, name='DETECT_{}'.format(counter), args=(counter, frag_sign, message, scenario['nn_samplerate'], scenario['resample_ratio'], scenario['audio_carrier'])))  
            detect_thread[counter].start() 
            
            # эмуляция детектирования
            emulate = param['emulate']
            if emulate:
                detect_thread_em1.append(threading.Thread(target=morse_detect, name='DETECT_{}'.format(counter), args=(counter, frag_sign, message, scenario['nn_samplerate'], scenario['resample_ratio'], scenario['audio_carrier'])))  
                detect_thread_em1[counter].start() 

                detect_thread_em2.append(threading.Thread(target=morse_detect, name='DETECT_{}'.format(counter), args=(counter, frag_sign, message, scenario['nn_samplerate'], scenario['resample_ratio'], scenario['audio_carrier'])))  
                detect_thread_em2[counter].start()
        

        #time_count += scenario['signal_lenght'] // 4
        time_count += frag_len
        
        # делаем задержку
        delay = param['delay']
        if delay:
            signal_lenght = max([frag_sign.shape[0], frag_sign.shape[1]])
            dt = signal_lenght//scenario['source_samplerate'] # вычисляем длительность файла в секундах
            print('Длительность сигнала: {} сек.'.format(dt))
            time.sleep(dt)
            #for _ in tqdm(range(dt)):
            #    time.sleep(1) # красиво показываем процесс
        
    return 'OK'


# параметры для 1го (MAIN) процесса демо
param = {'detect': True,
         'write_db': False,
         'cut_sign': True,
         'save_wav': False,
         'show_graph': True,
         'delay': False,
         'emulate': False}


# morse_translation_generator(scenario=scenario, music=music, messages=messages, param=param)


# параметры для 4х процессов
param = {'detect': True,
         'write_db': False,
         'cut_sign': True,
         'save_wav': False,
         'show_graph': False,
          'delay': True,
         'emulate': True}


# параметры для 4х процессов без эмуляции
param = {'detect': True,
         'write_db': False,
         'cut_sign': True,
         'save_wav': False,
         'show_graph': False,
          'delay': True,
         'emulate': False}


prc=[]
for i in range(2): # 4 процесса по 1 потоку дали плохой результата
    prc.append(Process(target=morse_translation_generator, name='morse_trans_gen_{}'.format(i), args=(scenario, music, messages, param)))
    prc[i].start()

print(result)