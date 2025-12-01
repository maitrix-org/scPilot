# LLM.py
from openai import OpenAI
from config.settings import OPENAI_API_KEY
from config.settings import GOOGLE_API_KEY


def query_llm(content, system_role, model_provider="openai", model_name="gpt-4o"):
    """
    Sends a request to the specified LLM provider and returns the response.

    :param content: The input prompt to send to the model.
    :param model_provider: The LLM provider to use ("openai", "anthropic", "local", etc.).
    :param model_name: The specific model to use.
    :return: The model's response as a string.
    """
    client = OpenAI(
        api_key=OPENAI_API_KEY,
    )
    if model_provider == "openai":
        if "o1-mini" in model_name:
            # For o1-mini models, combine system and user messages
            combined_content = f"{system_role}\n\n{content}"
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": combined_content}
                ]
            )
            
            response_text = response.choices[0].message.content.strip()
            sanitized_str = response_text.replace("```python", "").replace("```", "").strip()
            return sanitized_str
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": content}
                ]
            )
            response_text = response.choices[0].message.content.strip()
            if "```python" in response_text:
                sanitized_str = response_text.replace("```python", "").replace("```", "").strip()
            elif "```text" in response_text:
                sanitized_str = response_text.replace("```text", "").replace("```", "").strip()
            else:
                sanitized_str = response_text
            return sanitized_str

    elif model_provider == "anthropic":
        pass

    elif model_provider == "google":
        client = OpenAI(
            api_key=GOOGLE_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        response = client.chat.completions.create(
            model=model_name,
            n=1,
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": content}
            ]
        )
        response_text = response.choices[0].message.content.strip()
        if "```python" in response_text:
            sanitized_str = response_text.replace("```python", "").replace("```", "").strip()
        elif "```text" in response_text:
            sanitized_str = response_text.replace("```text", "").replace("```", "").strip()
        else:
            sanitized_str = response_text
        return sanitized_str

    elif model_provider == "local":
        pass

    elif model_provider == "openrouter":
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=GOOGLE_API_KEY,
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_role},
                        {"role": "user", "content": content}]
        )
        #print(response)
        response_text = response.choices[0].message.content.strip()
        if "```python" in response_text:
            sanitized_str = response_text.replace("```python", "").replace("```", "").strip()
        elif "```text" in response_text:
            sanitized_str = response_text.replace("```text", "").replace("```", "").strip()
        else:
            sanitized_str = response_text
        return sanitized_str
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")
