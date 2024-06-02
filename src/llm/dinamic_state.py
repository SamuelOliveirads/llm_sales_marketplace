from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import UserProxyAgent
from langchain.memory import ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma


class ProductRetrievalManager:
    def __init__(self, retriever: Chroma):
        self.retriever = retriever

    def get_product_details(self, query: str) -> str:
        """
        Retrieve product details from RAG based on the query.

        Parameters
        ----------
        query : str
            The user's query or product name to search for in the RAG.

        Returns
        -------
        str
            String with product details.
        """
        retrieved_docs = self.retriever.get_relevant_documents(query)
        return retrieved_docs


class ConversationCoordinator:
    """
    A chatbot class designed for managing interactions within a marketplace environment.
    This chatbot assists users by providing information on products based on their queries
    and guiding them through various stages of the buying process.

    Attributes:
        state (str): Represents the current state of the chatbot in the conversation flow.
        document_manager (ProductRetrievalManager): Manages retrieval and formatting of product
                                           details from a document database.
        prompts (dict): A dictionary mapping conversation states to their respective
                        prompt templates, which are used to generate responses based on
                        user inputs and document data.

    Methods:
        __init__: Initializes the chatbot with a document retriever.
    """

    def __init__(self, retriever: Chroma):
        self.state = "Welcome"
        self.document_manager = ProductRetrievalManager(retriever)
        self.prompts = {
            "Welcome": PromptTemplate(
                template="""
                    Você é um assistente de marketplace. Seu objetivo é ajudar os usuários a encontrar os produtos que eles estão interessados.
                    Se apresente como tal e responda qualquer dúvida do usuário.
                    Aqui está a questão do usuário: {question}
                """,
                input_variables=["question"],
            ),
            "ProductSearch": PromptTemplate(
                template="""
                    Você é um assistente de marketplace, seu objetivo é ajudar usuários a encontrar os produtos.
                    Quando o usuário perguntar sobre produtos querendo detalhes, você pode usar a seguinte
                    informação abaixo e deverá responder apenas se o produto conter aqui: \n\n {document} \n\n.
                    Aqui está a questão do usuário: {question}
                """,
                input_variables=["question", "document"],
            ),
            "ProductQA": PromptTemplate(
                template="""
                    Se o usuário tiver dúvidas sobre o produto você irá dar detalhes do mesmo, utilizando a informação disponível: \n\n {document} \n\n.
                    Importante notar que deve utilizar apenas as informações disponibilizadas, senão tiver diga que não possui maiores detalhes sobre
                    o produto.
                    Aqui está a questão do usuário: {question}
                """,
                input_variables=["question", "document"],
            ),
            "CollectInfo": PromptTemplate(
                template="""
                    Com as dúvidas satisfeitas, agora você vai pedir os dados do usuário.
                    Primeiro peça o Nome completo, e-mail e telefone.
                    É obrigatório que o usuário passe essas três informações para continuar a próxima etapa.
                    Aqui está a questão do usuário: {question}
                """,
                input_variables=["question"],
            ),
            "ConfirmPurchase": PromptTemplate(
                template="""
                    Agora, você vai finalizar a compra. Por favor, confirme o pedido e gere o link de finalização: <http://www.test-markeplace.com.br>.
                    Aqui está a questão do usuário: {question}
                """,
                input_variables=["question"],
            ),
            "ThankYou": PromptTemplate(
                template="""
                    Finalize o atendimento e agradeça o usuário, pedindo um feedback positivo ou negativo.
                    Aqui está a questão do usuário: {question}
                """,
                input_variables=["question"],
            ),
        }


class StateController:
    """
    Manages conversation state transitions within a marketplace environment, utilizing
    a state machine approach with an LLM to determine and update states based on user interaction.

    Attributes:
        chatbot (ConversationCoordinator): The chatbot instance managing the conversation.
        llm (RetrieveAssistantAgent): LLM agent used to determine conversation state.
        visited_states (List[str]): A list of states the conversation has already visited.
        user_proxy (UserProxyAgent): Proxy agent that manages communication with the LLM.
    """

    def __init__(self, chatbot: ConversationCoordinator):
        self.chatbot = chatbot
        self.llm = RetrieveAssistantAgent(
            name="MarketplaceStateAgent",
            system_message="Determine the current state of the conversation based on the history provided.",
            llm_config={
                "timeout": 600,
                "cache_seed": 42,
                "config_list": [{"model": "gpt-3.5-turbo", "temperature": 0}],
            },
        )
        self.visited_states = []
        self.user_proxy = UserProxyAgent(
            name="state_agent",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            is_termination_msg=lambda x: x.get("content", "")
            .rstrip()
            .endswith("TERMINATE")
            or x.get("content", "").rstrip().endswith("TERMINATE."),
            code_execution_config={
                "use_docker": False,
            },
        )

    def determine_state(self, history: ChatMessageHistory) -> str:
        """
        Determines the current conversation state based on the provided history.

        Parameters:
            history (ChatMessageHistory): The historical record of the conversation.

        Returns:
            str: The predicted current state of the conversation.
        """
        prompt = self.generate_prompt(history)
        state_prediction = self.user_proxy.initiate_chat(self.llm, message=prompt)
        return state_prediction

    def generate_prompt(self, history: ChatMessageHistory) -> str:
        visited_states = ", ".join(self.visited_states)
        prompt = f"""
        Given the conversation history:
        '{history}'

        Determine the current stage of the marketplace process. The stages are in order:
        Welcome > ProductSearch > ProductQA > CollectInfo > ConfirmPurchase > ThankYou.

        The details of each stage are:
        Welcome: welcome message if the user gives a greeting.
        ProductSearch: This step is mandatory, if the user asks something they will access the RAG containing product information and search.
        ProductQA: If the user has any questions about the product, they will access the RAG and search for more details.
        CollectInfo: This stage is compulsory, when the user selects a product we must collect initial data to start the purchase.
        ConfirmPurchase: This step is sequential to CollectInfo, it will complete the purchase and generate a link to track the order.
        ThankYou: This last step is mandatory and will thank you and ask for feedback.

        The user has already visited these stages: '{visited_states}'

        If the user dont visited any stage, the next stage will be Welcome or ProductSearch.
        If the user pass the name, email and phone number, the next stage will be ConfirmPurchase.

        What is the current stage? Reply only the specified stage name.
        """
        return prompt

    def update_chatbot_state(self, history: ChatMessageHistory) -> str:
        """
        Updates the chatbot's state based on the conversation history.

        Parameters:
            history (ChatMessageHistory): The historical record of the conversation.

        Returns:
            str: The chatbot's prompt for the newly updated state.
        """
        predicted_state = self.determine_state(history)
        llm_response = predicted_state.summary
        if llm_response not in self.chatbot.state:
            self.visited_states.append(llm_response)
            self.chatbot.state = llm_response
        return self.chatbot.prompts[llm_response]

    def handle_input(self, history: ChatMessageHistory) -> str:
        return self.update_chatbot_state(history)
