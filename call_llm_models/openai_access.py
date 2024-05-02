from openai import OpenAI
import os
import time
import openai

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def get_oai_completion(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_tokens=2048,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        return response.choices[0].message['content']
    except openai.RateLimitError as e:
        print("Rate limit exceeded, waiting before retrying...")
        time.sleep(70)
        return get_oai_completion(prompt)
    except openai.APIError as e:
        print(f"API returned an error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def call_chatgpt(ins):
    re_try_count = 3  # 重试次数
    while re_try_count > 0:
        re_try_count -= 1
        try:
            response = get_oai_completion(ins)
            if response:
                return response
        except Exception as e:
            print(f"Retrying due to an error: {e}")
            time.sleep(1)  # 等待 1 秒后重试

    print("Failed to get a response after retries.")
    return None
