import chainlit as cl
import httpx

API_URL = "http://localhost:8000"


async def end_session_api_call():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{API_URL}/end-session")
            response.raise_for_status()
            response_text = "Session ended on the server."
        except Exception as e:
            response_text = f"Failed to end session: {str(e)}"
    return response_text


@cl.on_message
async def main(message: cl.Message) -> cl.Message:
    question = message.content
    payload = {"question": question}

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            response = await client.post(f"{API_URL}/query", json=payload)
            response.raise_for_status()
            result = response.json()
            response_message = result["message"]
            if result.get("end_session", False):
                await cl.Message(
                    content="Obrigado por usar nos serviços, até mais!"
                ).send()
                response_message = await end_session_api_call()
                return
        except httpx.HTTPStatusError as e:
            response_message = f"Ocorreu um erro ao consultar a API: {str(e)}"
        except httpx.RequestError as e:
            response_message = f"Erro na requisição: {str(e)}"

    await cl.Message(content=response_message).send()
