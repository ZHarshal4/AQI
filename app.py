# -*- coding: utf-8 -*-

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from pandas import DataFrame
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import requests
import json
from pytz import timezone
from datetime import datetime
from datetime import timedelta
import time
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
import july
from july.utils import date_range
from  matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")

def add_bg_from_url():
   	st.markdown(
		f"""
		<style>
		.stApp {{
		    background-image: url("https://cdn.cssauthor.com/wp-content/uploads/2015/01/Free-Halftone-Watercolor-Textures-Backgrounds.jpg?strip=all&lossy=1&ssl=1");
		    background-attachment: fixed;
		    background-position: 25% 75%;
		    background-size: cover
		}}
		</style>
		""",
		unsafe_allow_html=True
		)

add_bg_from_url()
ind_time = datetime.now(timezone("Asia/Kolkata"))
to_time = int(time.mktime(ind_time.timetuple()))

lat = '18.5204'
lon = '73.8567'
start = '1609456500'
end = str(to_time)
appid = 'aa5ba39ad21a567b7e57153705a52caf'

response_API = requests.get('http://api.openweathermap.org/data/2.5/air_pollution/history?lat='+lat+'&lon='+lon+'&start='+start+'&end='+end+'&appid='+appid)
parse_data = response_API.text

data = json.loads(parse_data)
df = pd.json_normalize(data['list'])
df['dt'] = pd.to_datetime(df['dt'],unit='s')
df.rename(columns = {'dt':'Date',
                     'main.aqi':'AQIclass',
                     'components.co':'CO',
                     'components.no':'NO',
                     'components.no2':'NO2',
                     'components.o3':'O3',
                     'components.so2':'SO2',
                     'components.pm2_5':'PM25',
                     'components.pm10':'PM10',
                     'components.nh3':'NH3'}, inplace = True)

df['NOx'] = df['NO'] + df['NO2']
df = df.drop(['NO','NO2'], axis=1)
df['CO'] = df['CO']/1000

df["PM10_24hr_avg"] = df["PM10"].rolling(window = 24, min_periods = 16).mean().values
df["PM25_24hr_avg"] = df["PM25"].rolling(window = 24, min_periods = 16).mean().values
df["SO2_24hr_avg"] = df["SO2"].rolling(window = 24, min_periods = 16).mean().values
df["NOx_24hr_avg"] = df["NOx"].rolling(window = 24, min_periods = 16).mean().values
df["NH3_24hr_avg"] = df["NH3"].rolling(window = 24, min_periods = 16).mean().values
df["CO_8hr_max"] = df["CO"].rolling(window = 8, min_periods = 1).max().values
df["O3_8hr_max"] = df["O3"].rolling(window = 8, min_periods = 1).max().values

## PM2.5 Sub-Index calculation
def get_PM25_subindex(x):
	if x <= 30:
		return x * 50 / 30
	elif x <= 60:
		return 50 + (x - 30) * 50 / 30
	elif x <= 90:
		return 100 + (x - 60) * 100 / 30
	elif x <= 120:
		return 200 + (x - 90) * 100 / 30
	elif x <= 250:
		return 300 + (x - 120) * 100 / 130
	elif x > 250:
		return 400 + (x - 250) * 100 / 130
	else:
		return 0

df["PM25_SubIndex"] = df["PM25_24hr_avg"].apply(lambda x: get_PM25_subindex(x))

## PM10 Sub-Index calculation
def get_PM10_subindex(x):
	if x <= 50:
		return x
	elif x <= 100:
		return x
	elif x <= 250:
		return 100 + (x - 100) * 100 / 150
	elif x <= 350:
		return 200 + (x - 250)
	elif x <= 430:
		return 300 + (x - 350) * 100 / 80
	elif x > 430:
		return 400 + (x - 430) * 100 / 80
	else:
		return 0

df["PM10_SubIndex"] = df["PM10_24hr_avg"].apply(lambda x: get_PM10_subindex(x))

## SO2 Sub-Index calculation
def get_SO2_subindex(x):
	if x <= 40:
		return x * 50 / 40
	elif x <= 80:
		return 50 + (x - 40) * 50 / 40
	elif x <= 380:
		return 100 + (x - 80) * 100 / 300
	elif x <= 800:
		return 200 + (x - 380) * 100 / 420
	elif x <= 1600:
		return 300 + (x - 800) * 100 / 800
	elif x > 1600:
		return 400 + (x - 1600) * 100 / 800
	else:
		return 0

df["SO2_SubIndex"] = df["SO2_24hr_avg"].apply(lambda x: get_SO2_subindex(x))

## NOx Sub-Index calculation
def get_NOx_subindex(x):
	if x <= 40:
		return x * 50 / 40
	elif x <= 80:
		return 50 + (x - 40) * 50 / 40
	elif x <= 180:
		return 100 + (x - 80) * 100 / 100
	elif x <= 280:
		return 200 + (x - 180) * 100 / 100
	elif x <= 400:
		return 300 + (x - 280) * 100 / 120
	elif x > 400:
		return 400 + (x - 400) * 100 / 120
	else:
		return 0

df["NOx_SubIndex"] = df["NOx_24hr_avg"].apply(lambda x: get_NOx_subindex(x))

## NH3 Sub-Index calculation
def get_NH3_subindex(x):
	if x <= 200:
		return x * 50 / 200
	elif x <= 400:
		return 50 + (x - 200) * 50 / 200
	elif x <= 800:
		return 100 + (x - 400) * 100 / 400
	elif x <= 1200:
		return 200 + (x - 800) * 100 / 400
	elif x <= 1800:
		return 300 + (x - 1200) * 100 / 600
	elif x > 1800:
		return 400 + (x - 1800) * 100 / 600
	else:
		return 0

df["NH3_SubIndex"] = df["NH3_24hr_avg"].apply(lambda x: get_NH3_subindex(x))

## CO Sub-Index calculation
def get_CO_subindex(x):
	if x <= 1:
		return x * 50 / 1
	elif x <= 2:
		return 50 + (x - 1) * 50 / 1
	elif x <= 10:
		return 100 + (x - 2) * 100 / 8
	elif x <= 17:
		return 200 + (x - 10) * 100 / 7
	elif x <= 34:
		return 300 + (x - 17) * 100 / 17
	elif x > 34:
		return 400 + (x - 34) * 100 / 17
	else:
		return 0

df["CO_SubIndex"] = df["CO_8hr_max"].apply(lambda x: get_CO_subindex(x))

## O3 Sub-Index calculation
def get_O3_subindex(x):
	if x <= 50:
		return x * 50 / 50
	elif x <= 100:
		return 50 + (x - 50) * 50 / 50
	elif x <= 168:
		return 100 + (x - 100) * 100 / 68
	elif x <= 208:
		return 200 + (x - 168) * 100 / 40
	elif x <= 748:
		return 300 + (x - 208) * 100 / 539
	elif x > 748:
		return 400 + (x - 400) * 100 / 539
	else:
		return 0

df["O3_SubIndex"] = df["O3_8hr_max"].apply(lambda x: get_O3_subindex(x))

## AQI bucketing
def get_AQI_bucket(x):
	if x <= 50:
		return "Good"
	elif x <= 100:
		return "Satisfactory"
	elif x <= 200:
		return "Moderate"
	elif x <= 300:
		return "Poor"
	elif x <= 400:
		return "Very Poor"
	elif x > 400:
		return "Severe"
	else:
		return np.NaN

df["Checks"] = (df["PM25_SubIndex"] > 0).astype(int) + \
                (df["PM10_SubIndex"] > 0).astype(int) + \
                (df["SO2_SubIndex"] > 0).astype(int) + \
                (df["NOx_SubIndex"] > 0).astype(int) + \
                (df["NH3_SubIndex"] > 0).astype(int) + \
                (df["CO_SubIndex"] > 0).astype(int) + \
                (df["O3_SubIndex"] > 0).astype(int)

df["AQI_calculated"] = round(df[["PM25_SubIndex", "PM10_SubIndex", "SO2_SubIndex", "NOx_SubIndex",
                                 "NH3_SubIndex", "CO_SubIndex", "O3_SubIndex"]].max(axis = 1))

df.loc[df["PM25_SubIndex"] + df["PM10_SubIndex"] <= 0, "AQI_calculated"] = np.NaN
df.loc[df.Checks < 3, "AQI_calculated"] = np.NaN

df["AQI_bucket_calculated"] = df["AQI_calculated"].apply(lambda x: get_AQI_bucket(x))

# plotting
plotdata = df.copy()
cmap=LinearSegmentedColormap.from_list('rg',["g","y","r"], N=128)
till = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d")
def local_css(file_name):
	with open(file_name) as f:
		st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

def calendarplot(caldata,tilldate,cmap):
	fig,ax = plt.subplots()
	july.heatmap(date_range("2020-01-01", "2020-12-31"), caldata, cmap=cmap,month_grid=True,ax=ax,colorbar=True,fontsize=8)
	st.pyplot(fig)
	fig,ax = plt.subplots()
	july.heatmap(date_range("2021-01-01", "2021-12-31"), caldata, cmap=cmap,month_grid=True,ax=ax,colorbar=True,fontsize=8)
	st.pyplot(fig)
	fig,ax = plt.subplots()
	july.heatmap(date_range("2022-01-01", "2022-12-31"), caldata, cmap=cmap,month_grid=True,ax=ax,colorbar=True,fontsize=8)
	st.pyplot(fig)
	fig,ax = plt.subplots()
	july.heatmap(date_range("2023-01-01", tilldate), caldata, cmap=cmap,month_grid=True,ax=ax,colorbar=True,fontsize=8)
	st.pyplot(fig)

selected = option_menu(menu_title = None,
                       options=["AQI Info","Historical Readings","Forecast"],
                       icons=["journal-bookmark","hourglass-split","rocket-takeoff"],
                       default_index=0,
                       orientation="horizontal")

if selected == "AQI Info":	
	st.title("Air Quality Index")
	with st.container():
		col1, col2 = st.columns(2)
		with col1:
			htp1 = "https://raw.githubusercontent.com/ZHarshal4/AQI/main/aqi.jpg"
			st.image(htp1, caption="AQI logo",width=600)
		
		with col2:
			st.header("AQI")
			st.subheader("An air quality index (AQI) is used by government agencies to communicate to the public how polluted the air currently is or how polluted it is forecast to become. AQI information is obtained by averaging readings from an air quality sensor, which can increase due to vehicle traffic, forest fires, or anything that can increase air pollution. Pollutants tested include particulates, ozone, nitrogen dioxide, carbon monoxide, sulphur dioxide, among others.")
	with st.expander("Indian AQI Standards"):
		st.subheader("The National Air Quality Index (AQI) was launched in New Delhi on September 17, 2014, under the Swachh Bharat Abhiyan.  The continuous monitoring systems that provide data on near real-time basis are installed in New Delhi, Mumbai, Pune, Kolkata and Ahmedabad.")
		st.subheader("The Central Pollution Control Board along with State Pollution Control Boards has been operating National Air Monitoring Program (NAMP) covering 240 cities of the country having more than 342 monitoring stations. An Expert Group comprising medical professionals, air quality experts, academia, advocacy groups, and SPCBs was constituted and a technical study was awarded to IIT Kanpur. IIT Kanpur and the Expert Group recommended an AQI scheme in 2014. While the earlier measuring index was limited to three indicators, the new index measures eight parameters.")
		st.subheader("There are six AQI categories, namely Good, Satisfactory, Moderate, Poor, Severe, and Hazardous. The proposed AQI will consider eight pollutants (PM10, PM2.5, NO2, SO2, CO, O3, NH3, and Pb) for which short-term (up to 24-hourly averaging period) National Ambient Air Quality Standards are prescribed.")
	with st.expander("AQI Levels and Severity"):
		htp="https://raw.githubusercontent.com/ZHarshal4/AQI/main/ScaleInfo.JPG"
		st.image(htp, caption= 'AQI Levels')
	with st.expander("Precautions"):
		st.subheader(" Stay indoors if you can, with the windows and doors closed.Avoid frying food, which can increase indoor smoke.")
		st.subheader("If you have air conditioning, run it continuously, not on the auto cycle. It’s also helpful to close the fresh air intake so that smoke doesn’t get inside the house. If your system allows for it, install a high efficiency air filter, classified as MERV 13 or higher.")
		st.subheader("Portable air cleaners can also reduce indoor particulate matter in smaller spaces.")
		st.subheader("Avoid strenuous outdoor activities like exercising or mowing the lawn.")
		st.subheader("Don’t smoke cigarettes.")
		st.subheader("And though exercising outdoors can be a great way to stay healthy, the 101-150 range on the Air Quality Index is probably the highest level at which it remains safe to do so.")
		
if selected == "Historical Readings":
	st.subheader("Historical Readings of Individual Pollutants")
	option = st.selectbox("Historical Component-wise Readings",("AQI - Air Quality Index","CO - Carbon Monoxide","O3 - Ozone or Trioxygen","SO2 - Sulfur Dioxide","PM25 - Fine particles, or Particulate Matter 2.5","PM10 - Particulate Matter 10","NH3 - Ammonia","NOx - Nitric Oxide (NO) and Nitrogen Dioxide (NO2)"),label_visibility="hidden")
	if option == "AQI - Air Quality Index":
		calendarplot(plotdata.AQI_calculated,till,cmap)
	
	if option == "CO - Carbon Monoxide":
		calendarplot(plotdata.CO,till,cmap)

	if option == "O3 - Ozone or Trioxygen":
		calendarplot(plotdata.O3,till,cmap)

	if option == "SO2 - Sulfur Dioxide":
		calendarplot(plotdata.SO2,till,cmap)

	if option == "PM25 - Fine particles, or Particulate Matter 2.5":
		calendarplot(plotdata.PM25,till,cmap)

	if option == "PM10 - Particulate Matter 10":
		calendarplot(plotdata.PM25,till,cmap)

	if option == "NH3 - Ammonia":
		calendarplot(plotdata.NH3,till,cmap)

	if option == "NOx - Nitric Oxide (NO) and Nitrogen Dioxide (NO2)":
		calendarplot(plotdata.NOx,till,cmap)
		
if selected == "Forecast":
	data = df.copy()
	data = data.set_index('Date')
	data = data.dropna()
	timeseriessq = data['AQI_calculated']
	series = timeseriessq.resample('D').mean()
	series = series.interpolate()
	
	# fit model
	model = ARIMA(series, order=(5,1,0))
	model_fit = model.fit()
	predict = model_fit.forecast(steps = 2)
	tomo = (datetime.now(timezone("Asia/Kolkata")) + timedelta(days=1)).strftime("%d-%m-%Y")
	datomo = (datetime.now(timezone("Asia/Kolkata")) + timedelta(days=2)).strftime("%d-%m-%Y")
	st.title("AQI Prediction for next 2 days in Pune")
	st.header("AQI forecast for "+ tomo)
	st.subheader(str(round(predict[0],3)))
	st.header("AQI forecast for "+ datomo)
	st.subheader(str(round(predict[1],3)))


