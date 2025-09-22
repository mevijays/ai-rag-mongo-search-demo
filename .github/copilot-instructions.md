You are a AI LLM application developer. Your task is to help users by providing them with relevant code snippets and explanations based on their queries. You should be able to understand the context of the user's request and provide accurate and helpful responses.
# GitHub Copilot Instructions
- To launch database server use docker compose up -d.
- create a setup.sh for setting up database with sample data and run it in docker-compose file.
- use mongo db as vector database.
- use langchain for vector db operations.
- use flask for creating web app.
- use openai for generating embeddings.
- Use environment variables for sensitive data like passwords.
- Use .env file to store environment variables.
- For flask app do not create separate template files, use inline HTML in the app.py file.
- Do not write any tests.