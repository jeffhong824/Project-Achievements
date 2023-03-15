import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont

import os

from os import listdir
from os.path import isfile, isdir, join
from os import walk

import tkinter as tk  # 使用Tkinter前需要先匯入
import tkinter.messagebox
import pickle
from PIL import Image,ImageTk
from tkinter.filedialog import askdirectory

#task 1
from xlrd import open_workbook 
import os
from os import listdir , mkdir
from os.path import isfile, isdir, join,splitext
from os import walk
from os.path import isdir
from shutil import copyfile


# task2中 資料擷取 resample  並記錄id  
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, pick_types, events_from_annotations
from scipy import signal
import numpy as np

import time

import mne
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, pick_types, events_from_annotations
from scipy import signal
import numpy as np
from mne.decoding import CSP


import math
import scipy.io as sio 
from scipy.stats import entropy,skew,kurtosis
import pandas as pd
from math import e

from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(1)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from scipy import signal

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA

import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

def tidy_up(channel_position,all_data):

    Cannel=['EEG Fp1-Ref','EEG Fp2-Ref','EEG F7-Ref','EEG F3-Ref','EEG Fz-Ref','EEG F4-Ref','EEG F8-Ref','EEG T3-Ref','EEG C3-Ref','EEG Cz-Ref','EEG C4-Ref','EEG T4-Ref',
            'EEG T5-Ref','EEG P3-Ref','EEG Pz-Ref','EEG P4-Ref','EEG T6-Ref','EEG O1-Ref','EEG O2-Ref','EEG A1-Ref','EEG A2-Ref','Photic Ph'
           ]
    count_already_find_channel = 0
    sort_channel_list = []
    for find_channel in Cannel:
        #each channel - data
        for channel_position_each_channel in range(len(channel_position)):
            #find channel position
            if (channel_position[channel_position_each_channel].lower() == find_channel.lower()):
                count_already_find_channel+=1
                sort_channel_list.append(channel_position_each_channel)
    if(count_already_find_channel==22):      
        #save want channel data
        all_data_save = all_data[sort_channel_list]
        return all_data_save,sort_channel_list,1
    else:
        Cannel[21]='Photic-REF'
        count_already_find_channel = 0
        sort_channel_list = []
        for find_channel in Cannel:
            #each channel - data
            for channel_position_each_channel in range(len(channel_position)):
                #find channel position
                if (channel_position[channel_position_each_channel].lower() == find_channel.lower()):
                    count_already_find_channel+=1
                    sort_channel_list.append(channel_position_each_channel)
        if(count_already_find_channel==22):      
            #save want channel data
            all_data_save = all_data[sort_channel_list]
            return all_data_save,sort_channel_list,1        
        else:
            sort_channel_list=[]
            all_data_save =[]
            return all_data_save,sort_channel_list,0

def save_data(data,want_append_data):
    collect_data_num=data.shape[0]
    collect_data=np.zeros((collect_data_num+1,22,2000))
    collect_data[0:collect_data_num]=data
    collect_data[collect_data_num]=want_append_data
    return collect_data

def photo_stimulate_position(photo_times_list,photo_size_diff_list,data_math,collect_data,collect_data_after,collect_data_befor,data_name,sampling_rate,want_sampling_rate,select_data_save):
#def photo_stimulate_position(collect_data,collect_data_after,collect_data_befor,list_name[data_num][:5],raw.info['sfreq'],want_sampling_rate,select_data_save):
    photo_channel = select_data_save[21,:]
    photo_point = np.where(photo_channel!=0)[0]
    photo_point2 = np.where(photo_channel>200)[0]
    if(len(photo_point2)==0):
        photo_size_diff_list.append(data_name)
    except_calibrate_photo_point = np.where(photo_point>10000)[0]
    real_photo_point = photo_point[except_calibrate_photo_point]
    find_real_photo_point_diff=np.diff(real_photo_point)
    photo_frequency_change_position=np.where(find_real_photo_point_diff>2000)[0]
    howmanyphoto=len(photo_frequency_change_position)
    photo_times_list.append(howmanyphoto)
    success_time = 0
    if(howmanyphoto!=0):
        for photo_times in range(howmanyphoto):
            if (photo_times !=0):
                switch=1
            else:
                switch=0
            #the same hz point to photo change position
            event_data = select_data_save[:,real_photo_point[0+switch*(1+photo_frequency_change_position[photo_times-1])]:1+real_photo_point[photo_frequency_change_position[photo_times]]]
            event_data_long = len(event_data[0,:])
            sampling_long=int(event_data_long*(want_sampling_rate/sampling_rate))
            event_data_after = select_data_save[:,real_photo_point[0+switch*(1+photo_frequency_change_position[photo_times-1])]+event_data_long:1+real_photo_point[photo_frequency_change_position[photo_times]]+event_data_long]
            event_data_befor = select_data_save[:,real_photo_point[0+switch*(1+photo_frequency_change_position[photo_times-1])]-event_data_long:1+real_photo_point[photo_frequency_change_position[photo_times]]-event_data_long]
            #resample
            if(sampling_rate!=250):
                resampling_data=signal.resample(event_data,sampling_long,axis=1)
                resampling_data_after=signal.resample(event_data_after,sampling_long,axis=1)
                resampling_data_befor=signal.resample(event_data_befor,sampling_long,axis=1)
            else:
                resampling_data=event_data
                resampling_data_after=event_data_after
                resampling_data_befor=event_data_befor
            #cut down
            if(len(resampling_data[0,:])>2000):
                process_event_data = resampling_data[:,0:2000]
                process_event_data_after = resampling_data_after[:,0:2000]
                process_event_data_befor = resampling_data_befor[:,0:2000]
                collect_data=save_data(collect_data,process_event_data)
                collect_data_after=save_data(collect_data_after,process_event_data_after)
                collect_data_befor=save_data(collect_data_befor,process_event_data_befor)
                data_math+=1
                success_time+=1
            else:
                process_event_data = resampling_data
                process_event_data_after = resampling_data_after
                process_event_data_befor = resampling_data_befor

    return success_time,photo_times_list,photo_size_diff_list,data_math,collect_data,collect_data_after,collect_data_befor

def MorletWavelet(fc):

    F_RATIO = 7
    Zalpha2 = 3.3

    sigma_f = fc/F_RATIO
    sigma_t = 1/(2*math.pi*sigma_f)
    A = 1/((sigma_t*(math.pi**0.5))**0.5)
    #print(A)
    max_t = math.ceil(Zalpha2 * sigma_t)

    t = []
    for t_index in range(-max_t,max_t+1):
        t.append(t_index)
    MW = []
    for t_multi in range(len(t)):
        v1 = 1/(-2*sigma_t**2)
        v2 = (2j)*math.pi*fc
        want = t[t_multi]*(t[t_multi]*v1+v2)
        MW.append(A * e**(want))
    return MW

def tfa_morlet(td, fs, fmin, fmax, fstep):
    TFmap = []

    for fc in range(fmin,fmax+1):#,fs
        MW = MorletWavelet(fc/fs)  
        #np.convolve(td, MW, 'same')

        npad = len(MW) - 1
        u_padded = np.pad(td, (npad//2, npad - npad//2), mode='constant')
        cr = np.convolve(u_padded, MW, 'valid')

        TFmap.append(abs(cr))
    return TFmap

all_path = "D:/Record/Other/履歷/myself/4碩士論文finish/畢業論文/程式/ver2/"
path_model_intro = all_path+"model_intro/"

model_name=[]
path_model = all_path+"model/"    #獲取當前路徑 !!!!!!!!!!!!!!

num_dirs = 0 #路徑下資料夾數量
num_files = 0 #路徑下檔案數量(包括資料夾)
num_files_rec = 0 #路徑下檔案數量,包括子資料夾裡的檔案數量，不包括空資料夾


for root,dirs,files in os.walk(path_model):    #遍歷統計''
    for name in files:
        num_dirs += 1
        #print (os.path.join(root,name),"資料夾名稱:",name)
        model_name.append(name)
        
        
data_name=[]
path_data = all_path+"data/"    #獲取當前路徑 !!!!!!!!!!!!!!

num_dirs = 0 #路徑下資料夾數量
num_files = 0 #路徑下檔案數量(包括資料夾)
num_files_rec = 0 #路徑下檔案數量,包括子資料夾裡的檔案數量，不包括空資料夾

for root,dirs,files in os.walk(path_data):    #遍歷統計''
    for name in files:
        num_dirs += 1
        #print (os.path.join(root,name),"資料夾名稱:",name)
        data_name.append(name)

path_save =  all_path+'step/'
if not os.path.isdir(path_save+'create_data/'):
    os.mkdir(path_save+'create_data/')
if not os.path.isdir(path_save+'bandpass/'):
    os.mkdir(path_save+'bandpass/')
if not os.path.isdir(path_save+'erd_ers/'):
    os.mkdir(path_save+'erd_ers/')
if not os.path.isdir(path_save+'wavelet/'):
    os.mkdir(path_save+'wavelet/')
if not os.path.isdir(path_save+'ans/'):
    os.mkdir(path_save+'ans/')
path_create = path_save+'create_data/'
path_bandpass = path_save+'bandpass/'
path_erd_ers = path_save+'erd_ers/'
path_wavelet = path_save+'wavelet/'
path_ans = path_save+'ans/'

channel_name = ['EEG Fp1-Ref','EEG Fp2-Ref','EEG F7-Ref','EEG F3-Ref','EEG Fz-Ref','EEG F4-Ref','EEG F8-Ref','EEG T3-Ref','EEG C3-Ref','EEG Cz-Ref','EEG C4-Ref','EEG T4-Ref',
            'EEG T5-Ref','EEG P3-Ref','EEG Pz-Ref','EEG P4-Ref','EEG T6-Ref','EEG O1-Ref','EEG O2-Ref','EEG A1-Ref','EEG A2-Ref'  ]

window = tk.Tk() 
window.geometry('1100x680') #470x300
window.iconbitmap(all_path+'icon/eeg_icon.ico')
window.resizable(False, False)
#window.minsize(470,300)
#window.maxsize(940, 600)
window.title(' Taipei Veterans General Hospital')
window.configure(bg='white')#bg='white' 

add_pic = 5
add_num = 100

canvas_back_left = tk.Canvas(window, width=363, height=230, bg="white")
canvas_back_left.place(x=5, y=10)
canvas_back_right_1 = tk.Canvas(window, width=670, height=105, bg="white")
canvas_back_right_1.place(x=400, y=10)
canvas_back_right_2 = tk.Canvas(window, width=670, height=105, bg="white")
canvas_back_right_2.place(x=400, y=135)

photo_eeg = tk.PhotoImage(file=all_path+"icon/eeg.png")
photo_eeg = photo_eeg.zoom(4)
photo_eeg = photo_eeg.subsample(8)
imgLabel_eeg = tk.Label(window,image=photo_eeg,bg='white')#把圖片整合到標簽類中
imgLabel_eeg.place(x=30, y=7+add_pic)

photo_ai = tk.PhotoImage(file=all_path+"icon/ai.png")
photo_ai = photo_ai.zoom(2)
photo_ai = photo_ai.subsample(10)
imgLabel_ai = tk.Label(window,image=photo_ai,bg='white')#把圖片整合到標簽類中
imgLabel_ai.place(x=7, y=110+add_pic)

photo_eeg_run_1 = tk.PhotoImage(file=all_path+"icon/eeg_run.png")
photo_eeg_run_1 = photo_eeg_run_1.zoom(4)
photo_eeg_run_1 = photo_eeg_run_1.subsample(8)
photo_eeg_run_2 = tk.PhotoImage(file=all_path+"icon/eeg_run.png")
photo_eeg_run_2 = photo_eeg_run_2.zoom(4)
photo_eeg_run_2 = photo_eeg_run_2.subsample(8)

#photo_eeg_run_space_1 = tk.PhotoImage(file=all_path+"icon/eeg_run_space.png")
#photo_eeg_run_space_1 = photo_eeg_run_space_1.zoom(4)
#photo_eeg_run_space_1 = photo_eeg_run_space_1.subsample(8)
#photo_eeg_run_space_2 = tk.PhotoImage(file=all_path+"icon/eeg_run_space.png")
#photo_eeg_run_space_2 = photo_eeg_run_space_2.zoom(4)
#photo_eeg_run_space_2 = photo_eeg_run_space_2.subsample(8)


photo_logo_1 = tk.PhotoImage(file=all_path+"icon/北榮.png") # 3 : 10
photo_logo_1 = photo_logo_1.zoom(3)
photo_logo_1 = photo_logo_1.subsample(10)

photo_logo_2 = tk.PhotoImage(file=all_path+"icon/中央.png") # 4 : 10
photo_logo_2 = photo_logo_2.zoom(4)
photo_logo_2 = photo_logo_2.subsample(10)

'''
photo_logo_3 = tk.PhotoImage(file=all_path+"icon/NLP.png") # 1 : 14
photo_logo_3 = photo_logo_3.zoom(1)
photo_logo_3 = photo_logo_3.subsample(14)
'''

open_logo = False

def step4(text_data,stroke_epilepsy_filter_erders):
    if not os.path.isdir(path_wavelet+text_data[:-4]):
        os.mkdir(path_wavelet+text_data[:-4])
        
        ans_stroke_epilepsy = np.zeros((len(stroke_epilepsy_filter_erders),21,23,12))

        for data_len in range(len(stroke_epilepsy_filter_erders)):
            
            print(data_len+1,len(stroke_epilepsy_filter_erders))
            for channel in range(21):
                data=stroke_epilepsy_filter_erders[data_len][channel]
                samplerate=250
                N=2000

                f1 = 8 
                f2 = 30 
                fstep=1

                ts=[]
                taxis = []
                for i in range(1,int(N/4)+1):
                    ts.append(i/samplerate)
                for i in range(1,int(N)+1):
                    taxis.append(i/samplerate)

                spec = tfa_morlet(data, samplerate, f1, f2, fstep)
                Mag=abs(np.array(spec))
                for hz in range(23):
                    for sec in range(4):
                        e_value = entropy(spec[hz][0+500*sec:500+500*sec])
                        s_value = skew(spec[hz][0+500*sec:500+500*sec])
                        k_value = kurtosis(spec[hz][0+500*sec:500+500*sec])        
                        ans_stroke_epilepsy[data_len][channel][hz][sec] = e_value
                        ans_stroke_epilepsy[data_len][channel][hz][sec+4] = s_value
                        ans_stroke_epilepsy[data_len][channel][hz][sec+8] = k_value    

        np.save(path_wavelet+text_data[:-4]+'/stroke_epilepsy_filter_erders_wavelet_static' , np.array(ans_stroke_epilepsy))
        
def step3(text_data):
    if not os.path.isdir(path_erd_ers+text_data[:-4]):
        os.mkdir(path_erd_ers+text_data[:-4])

        stroke_epilepsy_filter = np.load(path_bandpass+text_data[:-4]+'/stroke_epilepsy_filter.npy')
        stroke_epilepsy_filter_erders = []
        for data_8s in range(len(stroke_epilepsy_filter)):
            print('stroke_epilepsy erd_ers......',data_8s+1,'/',len(stroke_epilepsy_filter))
            stroke_epilepsy_filter_erders.append([])
            for data_8s_channel in range(21):
                data_8s_channel_mean = np.mean(stroke_epilepsy_filter[data_8s][data_8s_channel])
                want_8s_erders = (stroke_epilepsy_filter[data_8s][data_8s_channel] - data_8s_channel_mean) / np.abs(data_8s_channel_mean)
                stroke_epilepsy_filter_erders[data_8s].append(want_8s_erders.tolist())

        np.save( path_erd_ers+text_data[:-4]+'/stroke_epilepsy_filter_erders' , np.array(stroke_epilepsy_filter_erders))
        
    else:
        stroke_epilepsy_filter = np.load(path_bandpass+text_data[:-4]+'/stroke_epilepsy_filter.npy')
        stroke_epilepsy_filter_erders = np.load( path_erd_ers+text_data[:-4]+'/stroke_epilepsy_filter_erders.npy')
                                                
    return stroke_epilepsy_filter_erders , len(stroke_epilepsy_filter)

def step2(text_data):
    if not os.path.isdir(path_bandpass+text_data[:-4]):
        os.mkdir(path_bandpass+text_data[:-4])
        
        stroke_epilepsy = np.load(path_create+text_data[:-4]+'/after.npy')
        stroke_epilepsy_id = np.load(path_create+text_data[:-4]+'/success_id.npy')

        sampling_freq = 250 
        duration = 8
        t = np.arange(0.0, duration, 1/sampling_freq) 
        lowcut = 1
        highcut = 50
        fs = 250

        filter = signal.firwin(numtaps=400, cutoff=[lowcut/(fs/2), highcut/(fs/2)], pass_zero=False) 
        stroke_epilepsy_filter = np.zeros([stroke_epilepsy.shape[0],stroke_epilepsy.shape[1]-1,stroke_epilepsy.shape[2]])

        for run_epilepsy in range(stroke_epilepsy.shape[0]): #len(stroke_epilepsy.shape[0])
            for run_21_channel in range(0,21):
                y = stroke_epilepsy[run_epilepsy][run_21_channel]
                y2 = signal.convolve(y, filter, mode='same')
                stroke_epilepsy_filter[run_epilepsy][run_21_channel]=y2

        data=np.delete(stroke_epilepsy_filter,0,axis=0)

        np.save( path_bandpass+text_data[:-4]+'/stroke_epilepsy_filter' , np.array(data))

def step1(text_data):

    if not os.path.isdir(path_create+text_data[:-4]):
        os.mkdir(path_create+text_data[:-4])
        data = text_data

        count_success_math=0
        photo_times_list=[]
        photo_size_diff_list = []
        data_math=0
        success_data=[]
        fail_data=[]
        fail_tidy_data=[]
        success_id=[]
        success_id_time=[]
        success_eeg=[]
        success_eeg_time=[]
        want_sampling_rate=250
        collect_data=np.zeros((1,22,2000))
        collect_data_after=np.zeros((1,22,2000))
        collect_data_befor=np.zeros((1,22,2000))

        try:
            raw = read_raw_edf(path_data+data, preload=True)
            all_data = raw.get_data()
            frequency=raw.info['sfreq']
            channel_position=raw.info["ch_names"]
            select_data_save,sort_channel_list,success_or_not = tidy_up(channel_position,all_data)
            if(success_or_not==1):

                success_time,photo_times_list,photo_size_diff_list,data_math,collect_data,collect_data_after,collect_data_befor=photo_stimulate_position(photo_times_list,photo_size_diff_list,data_math,collect_data,collect_data_after,collect_data_befor,data[:-4],raw.info['sfreq'],want_sampling_rate,select_data_save)
                success_data.append(data)
                if(success_time!=0):
                    success_eeg.append(data[:-6])
                    success_eeg_time.append(success_time)
                    if(data[:-6] not in success_id):
                        success_id.append(data[:-6])
                        success_id_time.append(success_time)
                    else:
                        success_id_time[-1]=int(success_id_time[-1])+success_time
            else:
                fail_tidy_data.append(data)
            print("read success")
            count_success_math+=success_or_not
        except:
            print("read not success")
            fail_data.append(data)

        print("tidy all",len(success_data),"success",count_success_math)

        np.save( path_create+text_data[:-4]+'/'+'now' , np.array(collect_data))
        np.save( path_create+text_data[:-4]+'/'+'after' , np.array(collect_data_after))
        np.save( path_create+text_data[:-4]+'/'+'befor' , np.array(collect_data_befor))
        np.save( path_create+text_data[:-4]+'/'+'success_id' , np.array(success_id))
        np.save( path_create+text_data[:-4]+'/'+'success_id_time' , np.array(success_id_time))
        np.save( path_create+text_data[:-4]+'/'+'success_eeg' , np.array(success_eeg))
        np.save( path_create+text_data[:-4]+'/'+'success_eeg_time' , np.array(success_eeg_time))
        with open(path_create+text_data[:-4]+'/'+'success_id.txt','w',) as file_success :
            for line in range(len(success_id)):
                file_success.write(success_id[line]+' '+str(success_id_time[line])+'\n')
        with open(path_create+text_data[:-4]+'/'+'success_eeg.txt','w',) as file_success :
            for line in range(len(success_eeg)):
                file_success.write(success_eeg[line]+' '+str(success_eeg_time[line])+'\n')
    
def show_ans_pic(text_data,text_model,data_8s_times,channel_num,ans_text_show,ans_text_show_2,color,eeg_ans,Notice_sec):
    def check_21channel():
        newWindow = tk.Toplevel(window)
        newWindow.geometry('600x750') #470x300
        newWindow.iconbitmap(all_path+'icon/eeg_icon.ico')
        #newWindow.resizable(False, False)
        newWindow.title(' Taipei Veterans General Hospital - Notice')
        newWindow.configure(bg='white')
        num = data_8s_times

        stroke_epilepsy_filter = np.load(path_create+text_data[:-4]+'/befor.npy')

        sampling_freq = 250 # 採樣頻率
        duration = 8 # 持續秒數
        t = np.arange(0.0, duration, 1/sampling_freq) #從0秒開始 週期
        if(len(num2)<=2):
            fig = Figure(figsize=(10,20), dpi=60)
        elif(len(num2)<=4):
            fig = Figure(figsize=(10,40), dpi=60)
        elif(len(num2)<=6):
            fig = Figure(figsize=(10,60), dpi=60)
        elif(len(num2)<=8):
            fig = Figure(figsize=(10,80), dpi=60)
        elif(len(num2)<=10):
            fig = Figure(figsize=(10,100), dpi=60)
        
        num_sec = 0
        for i in num2:
            for channel_num in range(21):
                y = stroke_epilepsy_filter[i+1][channel_num]
                fig.add_subplot(len(num2)*21,1,num_sec+1).plot(t,y)
                num_sec+=1



        canvas = FigureCanvasTkAgg(fig, master=newWindow)  # A tk.DrawingArea. root
        canvas.draw()
        #canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, newWindow)
        toolbar.update()
        #canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        
        if(len(num2)<=2):
            canvas.get_tk_widget().place(x=10, y=-70)
        elif(len(num2)<=4):
            canvas.get_tk_widget().place(x=10, y=-210)
        elif(len(num2)<=6):
            canvas.get_tk_widget().place(x=10, y=-350)
        elif(len(num2)<=8):
            canvas.get_tk_widget().place(x=10, y=-490)
        elif(len(num2)<=10):
            canvas.get_tk_widget().place(x=10, y=-630)
        toolbar.place(x=173, y=10)
        def on_key_press(event):
            print("you pressed {}".format(event.key))
            key_press_handler(event, canvas, toolbar)


        canvas.mpl_connect("key_press_event", on_key_press)
        print('has')
    #eeg(text_data,data_8s_times)
    
    num2 = Notice_sec
    
    if(eeg_ans==1):
        button_check_21_channel = tk.Button(window, text = "統整", command = check_21channel,font = 30)
        button_check_21_channel.place(x=320, y=282)

    num = data_8s_times
    
    stroke_epilepsy_filter = np.load(path_create+text_data[:-4]+'/befor.npy')

    sampling_freq = 250 # 採樣頻率
    duration = 8 # 持續秒數
    t = np.arange(0.0, duration, 1/sampling_freq) #從0秒開始 週期
    fig = Figure(figsize=(12, 10), dpi=60)
    
    for i in range(num):
        y = stroke_epilepsy_filter[i+1][channel_num]
        fig.add_subplot(num,1,i+1).plot(t,y)
    


    canvas = FigureCanvasTkAgg(fig, master=window)  # A tk.DrawingArea. root
    canvas.draw()
    #canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()
    #canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
    canvas.get_tk_widget().place(x=380, y=-55)
    toolbar.place(x=622, y=575)
    def on_key_press(event):
        print("you pressed {}".format(event.key))
        key_press_handler(event, canvas, toolbar)


    canvas.mpl_connect("key_press_event", on_key_press)

    label_data = tk.Label(window,text = text_data,bg='white',font = 30)
    label_data.place(x=158, y=55)
    label_model = tk.Label(window,text = text_model,bg='white',font = 30)
    label_model.place(x=160, y=180)
    label_data_show = tk.Label(window,text = ans_text_show ,bg='white',font = 30)#,font = 30  tkFont.Font(family="Lucida Grande", size=20)
    label_data_show.place(x=20, y=260,width=300, height=400)  
    label_data_show_2 = tk.Label(window,text = ans_text_show_2 ,bg='white',fg=color,font =("Helvetica",16))#,font = 30  tkFont.Font(family="Lucida Grande", size=20)
    label_data_show_2.place(x=265, y=282)
    def _quit():
        canvas.get_tk_widget().destroy()
        toolbar.destroy()
        label_data.destroy()
        label_model.destroy()
        label_data_show.destroy()
        label_data_show_2.destroy()
        button_return.destroy()
        combo_channel.destroy()
        try :
            button_check_21_channel.destroy()
        except:
            True        
        
        data_model_choose()
        
    def other_channel(channel_position):

        fig.clf()
        for i in range(num):
            y = stroke_epilepsy_filter[i+1][channel_position]
            fig.add_subplot(num,1,i+1).plot(t,y)
        canvas = FigureCanvasTkAgg(fig, master=window)  # A tk.DrawingArea. root
        canvas.draw()
        #canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        #canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        canvas.get_tk_widget().place(x=380, y=-55)

        def _quit():
            canvas.get_tk_widget().destroy()
            toolbar.destroy()
            label_data.destroy()
            label_model.destroy()
            label_data_show.destroy()
            label_data_show_2.destroy()
            button_return.destroy()
            combo_channel.destroy()
            try :
                button_check_21_channel.destroy()
            except:
                True
            data_model_choose()

        button_return = tkinter.Button(master=window, text="Return", command=_quit,font = 30)
        button_return.place(x=1040, y=575)#pack(side=tkinter.BOTTOM)          
    
    button_return = tkinter.Button(master=window, text="Return", command=_quit,font = 30)
    button_return.place(x=1040, y=575)#pack(side=tkinter.BOTTOM)    



        
    def callbackFunc_channel(event):
        print("New channel Selected",combo_channel.get())   
        channel_position = channel_name.index(combo_channel.get())
        canvas.get_tk_widget().destroy()
        toolbar.destroy()
        label_data.destroy()
        label_model.destroy()
        label_data_show.destroy()
        label_data_show_2.destroy()
        button_return.destroy()
        combo_channel.destroy()
        try :
            button_check_21_channel.destroy()
        except:
            True
        show_ans_pic(text_data,text_model,data_8s_times,channel_position,ans_text_show,ans_text_show_2,color,eeg_ans,Notice_sec)
        
    combo_channel = ttk.Combobox(window,values=channel_name ,width=12, height=8,font = 30)#,state="disabled"
    combo_channel.place(x=465, y=580)
    combo_channel.current(0)
    combo_channel.bind("<<ComboboxSelected>>", callbackFunc_channel)

def data_model_choose(): 
    def callbackFunc_data(event):
        print("New Data Selected",combo_data.get())
        try:
            print(path_data+combo_data.get())
            raw = read_raw_edf(path_data+combo_data.get(), preload=True)
            frequency=raw.info['sfreq']
            
            
            timeStamp = str(raw.info['meas_date'])[:-6]+ ' GMT'
#             timeArray = time.gmtime(timeStamp)
#             otherStyleTime = time.strftime("%Y - %m - %d  %H : %M : %S GMT", timeArray)
            
            lowpass_data = raw.info['lowpass']
            '''
            print(int((raw.n_times/frequency)/60),' min ',int((raw.n_times/frequency)%60),' sec')
            print(frequency)
            print(otherStyleTime)
            print(lowpass)
            print(raw.info)
            print(raw)
            
            '''
            string_data_1 = "experiment date  :  " + timeStamp 
            #string_data_1 = "date        :  " + otherStyleTime 
            string_data_2 = "lowpass filter      :  "+ str(int(lowpass_data)) +" Hz."
            string_data_3 = "sampling frequency  :  " + str(int(frequency)) +" Hz."
            string_data_4 = "times duration           :  " + str(int((raw.n_times/frequency)/60)) + " Min. " + str(int((raw.n_times/frequency)%60)) + " Sec."

            data_intro_1.set(string_data_1)
            data_intro_2.set(string_data_2)
            data_intro_3.set(string_data_3)
            data_intro_4.set(string_data_4)
            
        except:
            #print("reading False")
            string_data_1 = "experiment date  :  " + "reading False"
            string_data_2 = "lowpass filter      :  "+ "reading False"
            string_data_3 = "sampling frequency  :  " + "reading False"
            string_data_4 = "times duration           :  " + "reading False"
            data_intro_1.set(string_data_1)
            data_intro_2.set(string_data_2)
            data_intro_3.set(string_data_3)
            data_intro_4.set(string_data_4)
        
        
    def callbackFunc_model(event):
        print("New Model Selected",combo_model.get())
        sentence_list = []
        with open(path_model_intro+combo_model.get()[:-7]+".txt",'r',encoding='utf-8') as file:
            for line in file:
                sentence_list.append(line)
        model_intro_1.set(sentence_list[0][:-1])
        model_intro_2.set(sentence_list[1][:-1])
        model_intro_3.set(sentence_list[2][:-1])
        model_intro_4.set(sentence_list[3][:-1])
        model_intro_5.set(sentence_list[4][:-1])

    #imgLabel_eeg_run_space_1 = tk.Label(window,image=photo_eeg_run_space_1,bg='white')
    #imgLabel_eeg_run_space_1.place(x=30, y=450+add_pic)  
    #imgLabel_eeg_run_space_2 = tk.Label(window,image=photo_eeg_run_space_2,bg='white')
    #imgLabel_eeg_run_space_2.place(x=370, y=450+add_pic)

    
    data_intro_1 = tk.StringVar()
    data_intro_1.set('')
    data_intro_2 = tk.StringVar()
    data_intro_2.set('')
    data_intro_3 = tk.StringVar()
    data_intro_3.set('')
    data_intro_4 = tk.StringVar()
    data_intro_4.set('')
    

    model_intro_1 = tk.StringVar()
    model_intro_1.set('')
    model_intro_2 = tk.StringVar()
    model_intro_2.set('')
    model_intro_3 = tk.StringVar()
    model_intro_3.set('')
    model_intro_4 = tk.StringVar()
    model_intro_4.set('')
    model_intro_5 = tk.StringVar()
    model_intro_5.set('')
    
    #data_intro = ""
    label_data_intro_1 = tk.Label(window,textvariable = data_intro_1 ,bg='white',font = 30)#,font = 30  tkFont.Font(family="Lucida Grande", size=20)
    label_data_intro_1.place(x=420, y=30+add_pic)    
    label_data_intro_2 = tk.Label(window,textvariable = data_intro_2 ,bg='white',font = 30)#,font = 30  tkFont.Font(family="Lucida Grande", size=20)
    label_data_intro_2.place(x=420, y=70+add_pic) 
    label_data_intro_3 = tk.Label(window,textvariable = data_intro_3 ,bg='white',font = 30)#,font = 30  tkFont.Font(family="Lucida Grande", size=20)
    label_data_intro_3.place(x=790, y=30+add_pic) 
    label_data_intro_4 = tk.Label(window,textvariable = data_intro_4 ,bg='white',font = 30)#,font = 30  tkFont.Font(family="Lucida Grande", size=20)
    label_data_intro_4.place(x=790, y=70+add_pic) 
    
    model_data_intro_1 = tk.Label(window,textvariable = model_intro_1 ,bg='white',font = 30)#,font = 30  tkFont.Font(family="Lucida Grande", size=20)
    model_data_intro_1.place(x=420, y=150+add_pic)    
    model_data_intro_2 = tk.Label(window,textvariable = model_intro_2 ,bg='white',font = 30)#,font = 30  tkFont.Font(family="Lucida Grande", size=20)
    model_data_intro_2.place(x=420, y=190+add_pic) 
    model_data_intro_3 = tk.Label(window,textvariable = model_intro_3 ,bg='white',font = 30)#,font = 30  tkFont.Font(family="Lucida Grande", size=20)
    model_data_intro_3.place(x=790, y=150+add_pic) 
    model_data_intro_4 = tk.Label(window,textvariable = model_intro_4 ,bg='white',font = tkFont.Font(family="Lucida Grande", size=9))#,font = 30  tkFont.Font(family="Lucida Grande", size=20)
    model_data_intro_4.place(x=974, y=154+add_pic) 
    model_data_intro_5 = tk.Label(window,textvariable = model_intro_5 ,bg='white',font = 30)#,font = 30  tkFont.Font(family="Lucida Grande", size=20)
    model_data_intro_5.place(x=790, y=190+add_pic) 

    if(open_logo == True):
    
        imgLabel_logo_1 = tk.Label(window,image=photo_logo_1,bg='white')#把圖片整合到標簽類中 # 30 300 150
        imgLabel_logo_1.place(x=420+600, y=560+add_pic)    

        copyright = "Ver. : 0.1.0 Updated : March 4, 2020 || © 2020- Copyright : NCU || Co-development : National Central University                      Taipei Veterans General Hospital"
        label_copyright = tk.Label(window,text = copyright,bg='white',font = 30)
        label_copyright.place(x=5, y=620)

        imgLabel_logo_2 = tk.Label(window,image=photo_logo_2,bg='white')#把圖片整合到標簽類中
        imgLabel_logo_2.place(x=120+600, y=560+add_pic)      
    else:
        
        copyright = "Ver. : 0.1.0     ||     Updated : March 4, 2020     ||     © 2020- Copyright : NCU     ||     Co-development : National Central University   &   Taipei Veterans General Hospital"
        label_copyright = tk.Label(window,text = copyright,bg='white',font = 30)
        label_copyright.place(x=15, y=620)



    
    '''
    imgLabel_logo_3 = tk.Label(window,image=photo_logo_3,bg='black')#把圖片整合到標簽類中
    imgLabel_logo_3.place(x=300+600, y=567+add_pic)
    '''

    
    label_data = tk.Label(window,text = "Choose your data",bg='white',font = 30)#,font = 30  tkFont.Font(family="Lucida Grande", size=20)
    label_data.place(x=160, y=30+add_pic)
    combo_data = ttk.Combobox(window,values=data_name ,width=19, height=3,font = 30)
    combo_data.place(x=160, y=65+add_pic)
    combo_data.current(0)
    combo_data.bind("<<ComboboxSelected>>", callbackFunc_data)

    label_model = tk.Label(window,text = "Choose your model",bg='white',font = 30)
    label_model.place(x=160, y=150+add_pic)
    combo_model = ttk.Combobox(window,values=model_name,width=19, height=3,font = 30)
    combo_model.place(x=160, y=185+add_pic)
    combo_model.current(0)
    combo_model.bind("<<ComboboxSelected>>", callbackFunc_model)

    canvas_size = 180
    canvas_1 = tk.Canvas(window, width=canvas_size, height=22, bg="white")
    canvas_1.place(x=20, y=300+add_num)
    canvas_2 = tk.Canvas(window, width=canvas_size, height=22, bg="white")
    canvas_2.place(x=20+canvas_size*1, y=300+add_num)
    canvas_3 = tk.Canvas(window, width=canvas_size, height=22, bg="white")
    canvas_3.place(x=20+canvas_size*2, y=300+add_num)
    canvas_4 = tk.Canvas(window, width=canvas_size, height=22, bg="white")
    canvas_4.place(x=20+canvas_size*3, y=300+add_num)

    fill_line_1 = canvas_1.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="green")
    fill_line_2 = canvas_2.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="green")
    fill_line_3 = canvas_3.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="green")
    fill_line_4 = canvas_4.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="green")

    label_run_step1 = tk.Label(window, text='preprocessing', bg="white" ,font = 30) #width=15, height=2
    label_run_step1.place(x=61, y=302+add_num)
    label_run_step2 = tk.Label(window, text='bandpass filtering', bg="white" ,font = 30)
    label_run_step2.place(x=230, y=302+add_num)
    label_run_step3 = tk.Label(window, text='CSP Filtering', bg="white" ,font = 30)
    label_run_step3.place(x=425, y=302+add_num)
    label_run_step4 = tk.Label(window, text='wavelet transform', bg="white" ,font = 30)
    label_run_step4.place(x=590, y=302+add_num)

    steps_position = 75
    label_step1 = tk.Label(window, text='steps 1', bg="white" , font=("Helvetica",15))
    label_step1.place(x=steps_position, y=265+add_num)
    label_step2 = tk.Label(window, text='steps 2', bg="white" , font=("Helvetica",15))
    label_step2.place(x=steps_position+canvas_size*1, y=265+add_num)
    label_step3 = tk.Label(window, text='steps 3', bg="white" , font=("Helvetica",15))
    label_step3.place(x=steps_position+canvas_size*2, y=265+add_num)
    label_step4 = tk.Label(window, text='steps 4', bg="white" , font=("Helvetica",15))
    label_step4.place(x=steps_position+canvas_size*3, y=265+add_num)

    path_step = all_path+'step/'
    path_def = all_path+'data'

        
    def progress():
        print(window.winfo_screenwidth(),window.winfo_screenheight() ) #window.winfo_reqwidth(),window.winfo_reqheight() 
        
        n = 0
        text_data = combo_data.get()
        text_model = combo_model.get()
        print(text_data)
        print(text_model)

        with open(path_model+text_model, 'rb') as f:
            clf = pickle.load(f)


            
        step1(text_data)
        imgLabel_eeg_run_1 = tk.Label(window,image=photo_eeg_run_1,bg='white',width=150)
        
        n = n + canvas_size / 1
        canvas_1.coords(fill_line_1, (0, 0, n, 60))
        label_run_step1_2 = tk.Label(window, text='preprocessing', fg="white",bg="green" ,font = 30)
        label_run_step1_2.place(x=61, y=302+add_num)
        window.update()

        step2(text_data)
        imgLabel_eeg_run_1.destroy()
        imgLabel_eeg_run_1 = tk.Label(window,image=photo_eeg_run_1,bg='white',width=340)
        imgLabel_eeg_run_1.place(x=30, y=450+add_pic)
        
        n = n + canvas_size / 1
        canvas_2.coords(fill_line_2, (0, 0, n, 60))
        label_run_step2_2 = tk.Label(window, text='bandpass filtering', bg="green" , fg="white",font = 30)
        label_run_step2_2.place(x=230, y=302+add_num)
        window.update()
        
        step3_data , data_8s_times = step3(text_data)
        imgLabel_eeg_run_2 = tk.Label(window,image=photo_eeg_run_2,bg='white',width=170)
        imgLabel_eeg_run_2.place(x=370, y=450+add_pic)
        n = n + canvas_size / 1
        canvas_3.coords(fill_line_3, (0, 0, n, 60))
        label_run_step3_2 = tk.Label(window, text='CSP Filtering', bg="green" , fg="white",font = 30)
        label_run_step3_2.place(x=425, y=302+add_num)
        window.update()

        step4(text_data,step3_data)
        imgLabel_eeg_run_2.destroy()
        imgLabel_eeg_run_2 = tk.Label(window,image=photo_eeg_run_2,bg='white',width=360)
        imgLabel_eeg_run_2.place(x=370, y=450+add_pic)
        n = n + canvas_size / 1
        canvas_4.coords(fill_line_4, (0, 0, n, 60))
        label_run_step4_2 = tk.Label(window, text='wavelet transform', bg="green", fg="white" ,font = 30)
        label_run_step4_2.place(x=590, y=302+add_num)
        window.update()

        
        label_data_intro_1.destroy()
        label_data_intro_2.destroy()
        label_data_intro_3.destroy()
        label_data_intro_4.destroy()
        
        model_data_intro_1.destroy()
        model_data_intro_2.destroy()
        model_data_intro_3.destroy()
        model_data_intro_4.destroy()
        model_data_intro_5.destroy()

        label_data.destroy()
        combo_data.destroy()
        label_model.destroy()
        combo_model.destroy()
        canvas_1.destroy()
        canvas_2.destroy()
        canvas_3.destroy()
        canvas_4.destroy()
        label_run_step1.destroy()
        label_run_step2.destroy()
        label_run_step3.destroy()
        label_run_step4.destroy()
        label_run_step1_2.destroy()
        label_run_step2_2.destroy()
        label_run_step3_2.destroy()
        label_run_step4_2.destroy()
        label_step1.destroy()
        label_step2.destroy()
        label_step3.destroy()
        label_step4.destroy()
        button.destroy()        



 


        with open (path_create+text_data[:-4]+'/'+'success_eeg.txt','r') as eeg_file :
            stroke_epilepsy_eeg = []
            #label_eeg = []
            run_time = 0
            start_position = 0
            for line in eeg_file :
                a,b = line.split()
                stroke_epilepsy_eeg.append([])
                stroke_epilepsy_eeg[run_time].append(run_time+1)
                stroke_epilepsy_eeg[run_time].append(a)
                stroke_epilepsy_eeg[run_time].append(start_position)
                stroke_epilepsy_eeg[run_time].append(start_position+int(b))
                stroke_epilepsy_eeg[run_time].append(1)
                start_position+=int(b)
                #label_eeg.append(1)
                run_time+=1


        with open (path_create+text_data[:-4]+'/'+'success_id.txt','r') as eeg_file :
            stroke_epilepsy_id = []
            #label_id = []
            run_time = 0
            start_position = 0
            for line in eeg_file :
                a,b = line.split()
                stroke_epilepsy_id.append([])
                stroke_epilepsy_id[run_time].append(run_time+1)
                stroke_epilepsy_id[run_time].append(a)
                stroke_epilepsy_id[run_time].append(start_position)
                stroke_epilepsy_id[run_time].append(start_position+int(b))
                stroke_epilepsy_id[run_time].append(1)
                start_position+=int(b)
                #label_id.append(1)
                run_time+=1

        num_tree = 100

        stroke_epilepsy_8s = np.load(path_wavelet+text_data[:-4]+'/'+'stroke_epilepsy_filter_erders_wavelet_static.npy')
        stroke_8s_old = stroke_epilepsy_8s

        stroke_8s = np.zeros([stroke_8s_old.shape[0],21*23*12])
        for data_combine_channel_hz in range(stroke_8s_old.shape[0]):
            hz_channel = 0
            for channel in range(21):
                for hz in range(23):
                    stroke_8s[data_combine_channel_hz][ 0+12*hz_channel : 12+12*hz_channel ] = stroke_8s_old[data_combine_channel_hz][channel][hz]
                    hz_channel+=1

        test_feature = []
        #test_label = []
        test_feature_judge_record = []

        count_train_eeg = 0
        count_train_8s = 0
        for copy_feature in range(len(stroke_epilepsy_eeg)):
            test_feature_judge_record.append([])
            test_feature_judge_record[count_train_eeg].append(stroke_epilepsy_eeg[copy_feature][1])
            test_feature_judge_record[count_train_eeg].append(count_train_8s)
            for need_space in range(int(stroke_epilepsy_eeg[copy_feature][3])-int(stroke_epilepsy_eeg[copy_feature][2])):
                test_feature.append([])
                test_feature[count_train_8s] = stroke_8s[stroke_epilepsy_eeg[copy_feature][2]+need_space]
                #test_label.append(stroke_epilepsy_eeg[copy_feature][4])
                count_train_8s+=1
            test_feature_judge_record[count_train_eeg].append(count_train_8s)
            test_feature_judge_record[count_train_eeg].append(stroke_epilepsy_eeg[copy_feature][4])
            count_train_eeg+=1

        ans_inside = clf.predict(test_feature)
        ans_clf = ans_inside
        pro_clf = clf.predict_proba(test_feature)
        #print(classification_report(test_label, ans_inside))
        print(ans_inside)
        threshold = 0.01
        eeg_ans = []
        now_8s_ans_position = 0
        for test_eeg in range(len(test_feature_judge_record)):
            this_ans = 0
            this_eeg_times = int(test_feature_judge_record[test_eeg][2]) - int(test_feature_judge_record[test_eeg][1]) 
            for each_ans in range(this_eeg_times):
                this_ans+=int(ans_inside[now_8s_ans_position])
                now_8s_ans_position+=1
            if(this_ans/this_eeg_times >= threshold * this_eeg_times):
                eeg_ans.append(1)
            else:
                eeg_ans.append(0)
        #print(classification_report(label_eeg, eeg_ans))
        with open(path_ans+text_data[:-4]+'record_eeg_8s.txt','w') as file_8s:
            for test_ans_8s in ans_inside :
                file_8s.write(str(test_ans_8s))
            #file_8s.write(classification_report(label_eeg, eeg_ans))
        print(eeg_ans[0])
        ans_text_show = '認為中風後癲癇可能 : '
        ans_text_show_2 = ''
        color = 'black'
        if(eeg_ans[0]==1):
            ans_text_show += '是\n\n'
            ans_text_show_2 = '⚠'
            color = 'red'

        else:
            ans_text_show += '否\n\n'
            ans_text_show_2 = ''
            color = 'black'
            
        Notice_sec = []    
        for times in range(len(ans_inside)):
            add_str = '第 '+str(times+1)+' 段 8 秒 '
            if(ans_inside[times]==1):
                add_str += ('▲ %.2f\n\n'%pro_clf[times][1])
                Notice_sec.append(times)
            else:
                add_str += ('△ %.2f\n\n'%pro_clf[times][1])
            
            ans_text_show += add_str
            
            
        if(open_logo == True):
            imgLabel_logo_1.destroy()
            imgLabel_logo_2.destroy()
            label_copyright.destroy()
        else :
            label_copyright.destroy()

        imgLabel_eeg_run_1.destroy()
        imgLabel_eeg_run_2.destroy()
        #imgLabel_logo_3.destroy()
        show_ans_pic(text_data,text_model,data_8s_times,0,ans_text_show,ans_text_show_2,color,eeg_ans[0],Notice_sec)
    
    
    #button =ttk.Button(window, text = "點擊之後\n執行預測", command = progress,width=15 )
    #button.place(x=800, y=293+add_num)
    button = tk.Button(window, text = "執行", command = progress,font = 30)
    button.place(x=800, y=298+add_num)

data_model_choose()

window.mainloop()
