# Planejamento

- **Output**:
    - Sistema de jornada de compra para marketplace via LLM
        - API para o llm.
        - Site em chainlit para protótipo.
- **Input**:
    - Textos de produtos e suas características
- **Processo**:
    - **Coleta e Indexação de Dados**:
        - Utilizar Langchain para leitura e processamento dos dados:
            - **Textos**: Indexar para permitir busca por palavras-chave e frases relevantes.
        - Indexação dos dados em RAG como ChromaDB.

    - **Desenvolvimento do Prompt de Aprendizagem Adaptativa**:
        - Desenvolver as camadas de prompt para cada etapa da jornada:
            - Jornada de busca do produto
                - QA sobre os produtos caso necessário.
            - Coleta de dados mínimos para a compra.
            - Finalização da compra com link de acompanhamento.
    
    - **Armazenar histórico**:
        - Controle de histórico para inferir o progresso das etapas da jornada.
    
    - **Agente para controle de estado**:
        - Agente que vai acompanhar o processo da jornada de compra garantindo que tudo seja executado.
        - Agente que vai acompanhar se cada resposta está de acordo ou se alucinou.

    - **Implementação da Interface do Usuário**:
        - Utilizar Chainlit como protótipo para demonstração.

    - **Desenvolvimento e Deploy da API**:
        - Criar uma API para integrar o sistema.
        - Utilizar Docker para facilitar o deploy e escalabilidade.

    - **Verificação de Alucinações do Modelo**:
        - Implementar uma segunda etapa que verifica se o modelo alucinou ou se respondeu

    - **Adição de coleta de feedback**:
        - Coleta de histórico e feedback para teste A/B
