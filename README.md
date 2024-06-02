# LLM Sales Marketplace

## Objetivo do Projeto
Este projeto visa simular uma jornada de compras em um marketplace, utilizando um modelo de linguagem (LLM) para interação e uma base de dados de produtos.

## Metodologia
Baseado no [planejamento inicial](docs/planning.md), o projeto foi estruturado com a seguinte arquitetura, detalhada no [diagrama de componentes UML](docs/marketplace_component_diagram.drawio.png). As principais funcionalidades incluem:
- **MarketplaceJourney**: Gerencia a interação com o usuário.
- **StateController** e **ConversationCoordinator**: Controla o progresso da conversa e gerencia a conversação.
- **ProductRetrievalManager**: Recupera produtos disponíveis.

Este fluxo cria uma experiência de usuário previsível, permitindo também flexibilidade para esclarecer dúvidas sobre produtos e realizar pesquisas no marketplace.

## Considerações e Limitações
- **Escolha do Modelo**: O modelo escolhido foi o chatgpt 3.5, a escolha de modelos open source se demontra custosa, difícil de escalar e com pouco suporte ao idioma portugues. Modelos como da openai e gemini poderiam ser aplicados a depender de preferencias da empresa, o chatgpt foi escolhido arbitrariamente por possuir uma api robusta e ter maior contato de experiencia com o modelo. A escolha do gpt 3.5 leva em consideração o orçamento de requisições para esse projeto local, uma melhor precisão nas respostas poderiam vir junto com o gpt 4.0.
- **Desafios**: Controlar o progresso da conversa e gerenciar informações de produto são as maiores dificuldades. Prompts mais específicos e mecanismos para guardar informações sobre os produtos selecionados poderiam melhorar a experiência do usuário.

## Requisitos
Utilize Python 3.11 e instale as dependências com:
```bash
pip install -r requirements.txt
```

## Configuração de Pre-commit
Caso queira contribuir com o projeto use esse comando para configurar os hooks do pre-commit:
```bash
pre-commit install
```

## Credenciais
Crie um arquivo `.env` seguindo as orientações do arquivo `env.example`.

## Uso
- **Protótipo para apresentações internas** via Chainlit.
- **Ambiente de produção** com API e Docker.

Execute a API com:
```bash
uvicorn src.api.llm_api:app --reload
```
Para testar a interface web, use:
```bash
chainlit run src/webapp.py --port 8001
```
A documentação da API está disponível em [Documentação da API](http://localhost:8000/docs) e também como [Markdown](src/api/api_doc.md).

## Docker
Construa e execute o contêiner Docker com:
```bash
docker build -t marketplace-journey-app .
docker run -d -p 8000:8000 -p 8001:8001 --name marketplace-journey-container marketplace-journey-app
```
Acesse:
- Chainlit em [http://localhost:8001/](http://localhost:8001/)
- API em [http://localhost:8000/query](http://localhost:8000/query)

## Testes
Uma amostra dos testes realizados estão disponíveis em [output de testes](data/07_model_output/).

## Conclusões

## Próximos Passos
- **Alucinação**: Implementar salvamento de produtos selecionados para evitar que o modelo esqueça os itens escolhidos, essa etapa é crucial quando pensamos em uma cesta de produtos para compra.
- **Teste A/B**: Não foi priorizado para esse ciclo, sua ideia vem depois pois a coleta de dados poderia ser feito na interface do sistema e o projeto de teste a/b teria que ser um novo.
- **Botão de feedback**: Não é necessário para o chainlit, sua integração seria via site em produção e seria utilizado para métricas e teste a/b.
- **Análise de Sentimento**: Se houver a coleta de dados como feedback e histórico de conversa é possível derivar um novo projeto de análise de sentimento para medir a satisfação do usuário.
- **Menos framework, mais hardcoded**: Pela minha experiencia ainda não temos um framework robusto e razoavelmente flexível, casos mais complexos como esse projeto se torna menos atrativo soluções como langchain e autogen que tendem a ser rígidos com as suas aplicações, idealmente o projeto deveria ser dependente de apis como openai e a solução construida a partir de código do zero, mas essa abordagem de torna custosa temporalmente.
