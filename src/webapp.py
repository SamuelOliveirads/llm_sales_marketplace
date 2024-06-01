import chainlit as cl
import httpx

API_URL = "http://localhost:8000/query"


@cl.on_chat_start
def start():
    cl.user_session.set("stage", "main")


@cl.on_message
async def main(message: cl.Message) -> cl.Message:
    question = message.content

    payload = {"question": question}

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            response = await client.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            response_message = result["message"]
        except httpx.HTTPStatusError as e:
            response_message = f"Ocorreu um erro ao consultar a API: {str(e)}"
        except httpx.RequestError as e:
            response_message = f"Erro na requisição: {str(e)}"

    await cl.Message(content=response_message).send()
