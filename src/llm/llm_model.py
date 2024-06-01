import logging
import os
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document as LangchainDocument
from langchain.memory import ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MarketplaceJourney:
    def __init__(self, llm_type="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=llm_type, temperature=0)
        self.session_id = str(uuid.uuid4())
        self.history = ChatMessageHistory()

        self.main_prompt_template = PromptTemplate(
            template="""
        Você é um assistente de marketplace, seu objetivo é ajudar usuários a encontrar os produtos
        e tirar quaisquer dúvidas sobre dúvidas que o usuário tiver.
        Quando o usuário não tiver mais dúvidas você pode iniciar o processo de compra onde vai
        primeiro pedir os dados de Nome completo, número de telefone e e-mail, só terminando a coleta
        desses dados é que vai finalizar a compra gerando o seguinte link: <http://www.test-markeplace.com.br>.
        Com isso finalize o atendimento e agradeço o usuário pedindo um feedback positivo ou negativo.

        Durante a conversa o fluxo será:
        1. Perguntar quais produtos o usuário está interessado, irá tratar apenas uma categoria de produto por vez.
        2. Se o usuário tiver dúvidas sobre o produto irá dar detalhes do mesmo.
        3. Com as dúvidas satisfeitas vai pedir os dados do usuário.
        4. Vai finalizar a compra.
        5. Vai agradeçer e pedir feedback.

        Se o usuário perguntar sobre os produtos querendo detalhes você pode usar a seguinte
        informação abaixo e deverá responder apenas se o produto conter aqui: \n\n {document} \n\n.

        Aqui está a questão do usuário: {question}
        """,
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
            self.add_to_history("user", question)
            response = self.chain_with_history.invoke(
                {"question": question, "document": document},
                {"configurable": {"session_id": self.session_id}},
            )
            response_text = response.get("text", "Sem resposta disponível.")
            self.add_to_history("ai", response_text)
        except AttributeError as e:
            logging.error(f"Erro ao acessar .messages: {str(e)}")
            response = None
            raise
        return response

    def end_session(self):
        self.save_history_to_file()
        self.clear_history()

    def format_docs(self, docs: List[LangchainDocument]) -> str:
        """
        Format documents into a structured string.

        This function formats a list of documents into a structured string,
        including the source and type of each document (e.g., Vídeo, PDF,
        Texto, Exercício, Imagem).

        Parameters
        ----------
        docs : List[LangchainDocument]
            The list of documents to be formatted.

        Returns
        -------
        str
            A formatted string representing the documents.
        """
        formatted_docs = ""
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Desconhecido")
            if source.endswith(".txt"):
                format_type = "Texto"
            else:
                format_type = "Desconhecido"

            formatted_docs += (
                f"Documento {i+1} ({format_type}):" f"\n{doc.page_content}\n\n"
            )
        return formatted_docs

    def get_answer(self, question: str, retriever: Chroma) -> Tuple[str, Optional[str]]:
        """
        Get an answer from the LLM based on the stage of interaction.

        This function retrieves relevant documents and formats them, then uses
        different chains to get an answer based on the stage (intro, main, end).

        Parameters
        ----------
        question : str
            The question asked by the user.
        retriever : Chroma
            The retriever used to get relevant documents.

        Returns
        -------
        Tuple[str, Optional[str]]
            A tuple containing the response message and the formatted documents
            (if applicable).
        """
        retrieved_docs = retriever.get_relevant_documents(question)
        formatted_docs = self.format_docs(retrieved_docs)
        response = self.run_interaction(question, formatted_docs)
        rag_content = formatted_docs

        return response, rag_content
