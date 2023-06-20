import json
import random
import datetime
import requests
import os
import logging
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
from functools import wraps

if os.getenv("DEBUG", False):
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARNING)

GPT_MODEL = "gpt-3.5-turbo-0613"

def load_csv_as_df(csv_path):
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)

def log_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Function {func.__name__} called with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"Function {func.__name__} returned: {result}")
        return result
    return wrapper

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

if OPENAI_API_KEY is None:
    OPENAI_API_KEY = input("Enter your OpenAI API key: ")

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, function_call=None, model=GPT_MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    json_data = {"model": model, "messages": messages, "temperature": 0, "top_p": 0} 
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        if (response.status_code != 200):
            print(f"Error: {response.status_code}")
            print(response.json())
            raise Exception("OpenAI API Error")
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "grey",
    }
    formatted_messages = []
    for message in messages:
        if message["role"] == "system":
            formatted_messages.append(f"system: {message['content']}\n")
        elif message["role"] == "user":
            formatted_messages.append(f"user: {message['content']}\n")
        elif message["role"] == "assistant" and message.get("function_call"):
            formatted_messages.append(f"\n   >>> function call ({message['function_call']['name']}): {json.dumps(message['function_call']['arguments'])}")
        elif message["role"] == "assistant" and not message.get("function_call"):
            formatted_messages.append(f"assistant: {message['content']}\n")
        elif message["role"] == "function":
            formatted_messages.append(f"   <<< function result ({message['name']}): {message['content']}\n")
    for formatted_message in formatted_messages:
        
        if (formatted_message.startswith("\n") or formatted_message.startswith(" ")):
            print(
                colored(
                    formatted_message,
                    role_to_color[messages[formatted_messages.index(formatted_message)]["role"]],
                )
            )
        else:
            print(
                colored(
                    formatted_message,
                    role_to_color[messages[formatted_messages.index(formatted_message)]["role"]],
                    attrs=["bold"],
                )
            )
            
        

functions = [
    {
        "name": "get_today",
        "description": "Gets the current date",
        "parameters": {
            "type": "object",
            "properties": { },
        }
    },
    {
        "name": "get_temperature",
        "description": "Get the temperature in a location at a date",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name. Always acquire this information, never assume.",
                },
                "start_date": {
                    "type": "string",
                    "format": "date",
                    "description": "The start date to get the weather for, format YYYY-MM-DD. Defaults to none",
                },
                "end_date": {
                    "type": "string",
                    "format": "date",
                    "description": "The end date of period to get the weather for, format YYYY-MM-DD, defaults to none.",
                }
            },
            "required": ["location", "start_date", "end_date"],
        },
    },
    {
        "name": "get_current_weather",
        "description": "Get the current weather.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name. Always acquire this information, never assume location.",
                },
                "date": {
                    "type": "string",
                    "format": "date",
                    "description": "The date to get the weather for. Always acquire this information, never assume the current date.",
                }
            },
            "required": ["location", "date"],
        },
    },
    {
        "name": "get_user_location",
        "description": "Gets the location city of the user",
        "parameters": {
            "type": "object",
            "properties": { },
        }
    },
    {
        "name": "where_am_i",
        "description": "Gets the location city of the user",
        "parameters": {
            "type": "object",
            "properties": { },
        }
    },
    {
        "name": "get_user_information",
        "description": "Gets the user information object, with name, email, and location fields",
        "parameters": {
            "type": "object",
            "properties": { },
        },
    },
    {
        "name": "get_dress_for_temperature",
        "description": "Gets clothing information for a given weather temperature",
        "parameters": {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    "description": "The temperature in celsius",
                },
            },
            "required": ["temperature"]
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to a given email address. Only use when explicited called by the user saying email.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to send",
                },
                "email": {
                    "type": "string",
                    "description": "The email address to send the message to",
                },
            },
            "required": ["message", "email"]
        },
    },
]

@log_function
def where_am_i(args):
    return "London"

@log_function
def get_user_information(args):
    result = json.dumps({ "name": "Peter", "location": "London", "email": "pz@me.com" })
    return  result
    
@log_function
def get_current_weather(args):
    parsed_args = json.loads(args)
    location = parsed_args.get("location", None)
    if location == "current":
        return "location info required"
    result = f"It's {random.randint(-20,30)} degrees celsius and {random.choice(['sunny', 'cloudy', 'rainy'])}"
    return result

@log_function
def get_today(args):
    result = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return result
 
@log_function
def get_temperature(args):
    parsed_args = json.loads(args)
    start_date_str = parsed_args.get("start_date", None)
    end_date_str = parsed_args.get("end_date", None)
    location = parsed_args.get("location", None)
    
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
    values = []
    
    if location:
        df = load_csv_as_df(f"data/{location.lower()}.csv")
    
        
    while start_date <= end_date:
        
        
        #find a matching date in the dataframe
        if df is not None:
            date_str = start_date.strftime("%Y-%m-%d")
            row = df.loc[df['Date'] == date_str]
            if len(row) > 0:
                values.append(int(row['Temperature'].values[0]))
            else:
                values.append(random.randint(-20,30))
        else:
            values.append(random.randint(-20,30))
        start_date += datetime.timedelta(days=1)
    
    values.sort()
    return json.dumps(values)

@log_function
def get_dress_for_temperature(args):
    parsed_args = json.loads(args)
    temperatureArray = parsed_args.get("temperature", None)
    results = []
    for temperature in temperatureArray:
        if temperature < 15:
            result = "coat"
        elif temperature < 18:
            result = "jacket"
        elif temperature < 20:
            result = "thin jacket"
        elif temperature < 23:
            result = "shirt"
        else:
            result = "t-shirt"
        results.append({ "temperature": temperature, "clothing": result })
    return json.dumps(results)

@log_function
def send_email(args):
    parsed_args = json.loads(args)
    message = parsed_args.get("message", None)
    email = parsed_args.get("email", None)    
    result = f"Sent message '{message}' to {email}"
    return result

def python(args):
    print(args)
    return ""

fn_map = {
    "where_am_i": where_am_i,
    "get_user_location": where_am_i,
    "get_user_information": get_user_information,
    "get_current_weather": get_current_weather,
    "get_today": get_today,
    "get_temperature": get_temperature,
    "get_dress_for_temperature": get_dress_for_temperature,
    "send_email": send_email,
    "python": python,
}

messages = []
messages.append({"role": "system", "content": r"""Don't make assumptions about what values to plug into functions, before invoking a function, ask the user to confirm the values for parameters
                Do call python directly. Always start conversation with invoking functions.get_user_information to get the user's name and location.
                 """ 
                 + f"\nToday is {datetime.datetime.now().strftime('%Y-%m-%d')}"
                 })
print("\n")
while True:
    user_message = input(colored("user:", "green", attrs=["bold"]))
    if user_message == "quit":
        break
    
    messages.append({"role": "user", "content": user_message})
    chat_response = chat_completion_request(messages, functions=functions, function_call="auto")
    if isinstance(chat_response, Exception):
        break
     
    logging.info(chat_response.json())
    choices = chat_response.json()["choices"]
    response_messages = [choice["message"] for choice in choices]
    
    while len(response_messages) > 0:
        response_message = response_messages.pop(0)
        pretty_print_conversation([response_message])
        messages.append(response_message)
        function_call = response_message.get("function_call")
        if function_call is not None:
            function_name, function_arguments = function_call["name"], function_call["arguments"]
            if function_name == "python":
                print(response_message)
                # exit("python function is not supported")
            function_to_call = fn_map[function_name]
            fn_result = function_to_call(function_arguments)
            fn_response_message = { "role": "function", "name": function_name, "content": fn_result,  }
            pretty_print_conversation([fn_response_message])
            messages.append(fn_response_message)
            fn_chat_response = chat_completion_request(messages, functions=functions, function_call="auto")
            logging.info(fn_chat_response.json())
            fn_chat_choices = fn_chat_response.json()["choices"]
            fn_message = [choice["message"] for choice in fn_chat_choices][0]
            response_messages.insert(0, fn_message)
            
            
            
            
        
        
