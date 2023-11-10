# About
Weather Skald is a whimsical but (somewhat) useful AI utility that reads the weather in a Viking style. 

For more information see https://www.warpefelt.se/weather-skald

# Usage
This program can be run either as a stand-alone script (``python weatherskald.py``) or imported as a module. 

You will need to configure keys for OpenAI and WeatherFlow as well as a WeatherFlow station number in ``weatherskald.cfg`` before running the program. 

You will also need to provide a voice sample (the longer the better) for the TTS. The default file name for this is ``weatherflow_default.wav``. 

# Dependencies
* openai>=1.1.1
* pytorch>=2.1.0
* coquitts>=0.20.0

A GPU with CUDA support (plus the associated pytorch configuration) is strongly recommended. 
