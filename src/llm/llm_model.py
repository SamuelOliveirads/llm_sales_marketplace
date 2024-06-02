import logging
import os
import uuid
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain.memory import ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from src.llm.dinamic_state import DocumentManager, MarketplaceAgent, MarketplaceChatbot

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MarketplaceJourney:
    def __init__(self, retriever: Chroma, llm_type="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=llm_type, temperature=0)
        self.session_id = str(uuid.uuid4())
        self.history = ChatMessageHistory()
        self.document_manager = DocumentManager(retriever)
        self.chatbot = MarketplaceChatbot(self.document_manager)
        self.state_agent = MarketplaceAgent(self.chatbot)

        self.main_prompt_template = PromptTemplate(
            template="""Você é um assistente de marketplace. Seu objetivo é ajudar os usuários a
            encontrar os produtos que eles estão interessados. Aqui está a questão do usuário:
            {question}""",
            input_variables=["question", "document"],
        )

        self.main_chain = LLMChain(
            prompt=self.main_prompt_template,
            llm=self.llm,
            output_parser=StrOutputParser(),
        )
        self.chain_with_history = RunnableWithMessageHistory(
            self.main_chain,
            get_session_history=lambda session_id: self.history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

    def add_to_history(self, sender, message):
        if sender == "user":
            self.history.add_user_message(message)
        else:
            self.history.add_ai_message(message)

    def save_history_to_file(self):
        df = pd.DataFrame(
            [
                {
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "sender": sender,
                    "message": message,
                }
                for sender, message in self.history.messages
            ]
        )
        output_dir = "../07_model_output/"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}{self.session_id}.csv"
        df.to_csv(filename, index=False)
        logging.info(f"Memory saved to {filename}")

    def clear_history(self):
        self.history.clear()

    def run_interaction(self, question: str, document: str) -> Optional[str]:
        try:
            response = self.chain_with_history.invoke(
                {"question": question, "document": document},
                {"configurable": {"session_id": self.session_id}},
            )
        except AttributeError as e:
            logging.error(f"Erro ao acessar .messages: {str(e)}")
            response = None
            raise
        return response

    def end_session(self):
        self.save_history_to_file()
        self.clear_history()

    def update_prompt(self, next_prompt):
        self.main_prompt_template = next_prompt
        self.main_chain.prompt = self.main_prompt_template

    def get_answer(self, question: str) -> Tuple[str, Optional[str]]:
        """
        Get an answer from the LLM based on the stage of interaction.

        This function retrieves relevant documents and formats them, then uses
        different chains to get an answer based on the stage (intro, main, end).

        Parameters
        ----------
        question : str
            The question asked by the user.

        Returns
        -------
        Tuple[str, Optional[str]]
            A tuple containing the response message and the formatted documents
            (if applicable).
        """
        self.add_to_history("user", question)
        formatted_docs = self.document_manager.get_product_details(question)
        next_prompt = self.state_agent.handle_input(self.history)
        self.update_prompt(next_prompt)
        response = self.run_interaction(question, formatted_docs)
        response_text = response.get("text", "Sem resposta disponível.")
        self.add_to_history("ai", response_text)
        rag_content = formatted_docs

        return response, rag_content
