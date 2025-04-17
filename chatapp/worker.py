from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import requests

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

def speech_to_text(audio_binary):
    base_url = "https://sn-watson-stt.labs.skills.network"
    api_url = base_url+'/speech-to-text/api/v1/recognize'

    params = {
        'model': 'en-US_Multimedia',
    }
    
    body = audio_binary
    
    response = requests.post(api_url, params=params, data=audio_binary).json()
    text = 'null'
    while bool(response.get('results')):
        print('speech to text response:', response)
        text = response.get('results').pop().get('alternatives').pop().get('transcript')
        print('recognised text: ', text)
        return text


def text_to_speech(text, voice=""):
    base_url = "https://sn-watson-tts.labs.skills.network"
    api_url = base_url + '/text-to-speech/api/v1/synthesize?output=output_text.wav'

    if voice != "" and voice != "default":
        api_url += "&voice=" + voice

    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json',
    }

    json_data = {
        'text': text,
    }

    response = requests.post(api_url, headers=headers, json=json_data)
    print('text to speech response:', response)
    return response.content


def openai_process_message(user_message):
    prompt = "Act like a personal assistant. You can respond to questions, translate sentences, summarize news, and give recommendations.\n\nUser: " + user_message + "\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=512, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text.split("Assistant:")[-1].strip()
