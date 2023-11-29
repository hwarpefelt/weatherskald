"""This is WeatherSkald: a generator for voice readings of skaldic poems about the current weather. 

This program uses the WeatherFlow, OpenAI, and CoquiTTS libraries to gather weather data, transform it into a skaldic poem, and finally read that poem out loud using a synthesized voice. 

To run this program you will need to install the following modules: 
* OpenAI
* PyTorch (optional)
* CoquiTTS (optional)

This file can be run as an independent script and imported as a module. 

See the accompanying LICENSE file for license specifications. 
"""

import requests
from openai import OpenAI


class WeatherSkald:
	"""This is the main skald class. To use, please create an instance of this class and call the skald_weather() (for OpenAI TTS) or skald_weather_local() (for local CoquiTTS) member functions. Please configure the various keys in weatherskald.cfg before running. 

	If you want to use local CoquiTTS you will need to provide a training voice file in wav format (default filename weatherskald_default.wav). 

	Attributes
	----------
	config : dict()
		a key:value pairing of the configurable attributes (weatherflow_token, weatherflow_station_id, and openai_token)
	outputfile : str
		the name of the default output file

	Methods
	-------
	fetch_forecast()
		Gets the forecast for the next 10 days (including today) from the weather station
	weather_poem()
		Calls fetch_forecast() to get weather forecast. Creates a GPT-4 prompt from the forecast and sends it to OpenAI's servers to generate the skaldic poem
	skald_weather(speakerfile=weatherskald_default.wav)
		Class weather_poem() to get the poem to be read. Uses CoquiTTS to create sound file. Requires a speaker file to function (default weatherskald_default.wav)
	"""
	def __init__(self, configfile = "weatherskald.cfg", outputfile="skaldic_weather"):
		"""Creates a new WeatherSkald instance. 
		
		Parameters
		----------
		configfile : str
			the name of the config file (default weatherskald.cfg)
		outputfile : str
			the name of the output file for the generator (default skaldic_weather.wav/mp3)
		"""
		self.config = dict()
		self.outputfile = outputfile
		f = open(configfile, 'r')
		for line in f.readlines():
			larr = line.split(" ")
			self.config[larr[0]] = larr[1].strip('\n') # Grab config data
		self.client = OpenAI(api_key=self.config['openai_key'])

	def fetch_forecast(self):
		"""Gets the forecast from the weather station configured in the config file. 

		Returns
		-------
		str
			the forecast for the next 10 days (including today)
		"""
		retstr = ""
		
		token = self.config['weatherflow_token']
		station_id = self.config['weatherflow_station_id']
		
		fetch_url = f"https://swd.weatherflow.com/swd/rest/better_forecast?station_id=67295&units_temp=f&units_wind=mph&units_pressure=mmhg&units_precip=in&units_distance=mi&token={token}"
		headers = {"accept": "application/json"}
		response = requests.get(fetch_url, headers=headers)
		cc = response.json()['current_conditions']
		dfcs = response.json()['forecast']['daily']
		retstr = f"Air-Temp {cc['air_temperature']}F (feels like {cc['feels_like']}F). {cc['relative_humidity']}% humidity"

		# Get conditions for the next 10 days
		for i in range(9):
			dfc = dfcs[i]
			retstr += f"{dfc['month_num']}/{dfc['day_num']}: {dfc['conditions']}, {dfc['air_temp_high']}/{dfc['air_temp_low']}F"

		return retstr

	def weather_poem(self):
		"""Grabs the weather forecast from fetch_forecast() and runs it through GPT-4 with a custom prompt. 

		Returns
		-------
		str
			the skaldic poem describing the weather forecast
		"""
		data = self.fetch_forecast()
		content = f"Write a paragraph describing the following weather in the style of a Viking skald: {data}"
		gpt_response = self.client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": content}])
		return gpt_response.choices[0].message.content

	def skald_weather(self):
		"""Gets the poem from weather_poem() and runs it through OpenAI's TTS service to read the poem. Will output a .mp3 file.
		"""
		output_file_path = self.outputfile + ".mp3"
		content = self.weather_poem()
		response = self.client.audio.speech.create(model="tts-1", voice="onyx", input=content)
		response.stream_to_file(output_file_path)

	def skald_weather_local(self, speakerfile = "weatherskald_default.wav"):
		"""Gets a poem from weather_poem() and runs it through CoquiTTS to read the poem. Will output a .wav file. 

		Parameters
		----------
		speakerfile : str
			the file name to use for the speaker voice (default is weatherskald_default.wav)
		"""
		import torch
		from TTS.api import TTS

		content = self.weather_poem()
		
		device = "cuda" if torch.cuda.is_available() else "cpu"
		model = "tts_models/multilingual/multi-dataset/xtts_v2"
		tts = TTS(model).to(device)
		tts.tts_to_file(text = content, speaker_wav=speakerfile, language="en", file_path=self.outputfile + ".mp3")
		

def main():
	ws = WeatherSkald()
	ws.skald_weather()
	print("WeatherSkald output saved to" + str(ws.outputfile))
	

if __name__ == '__main__':
	main()
