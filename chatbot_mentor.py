import os
import dotenv
import langchain
from langchain_groq import ChatGroq 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



dotenv.load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

modelo = ChatGroq(
model="openai/gpt-oss-20b",
temperature=0.7,
api_key=api_key
)

lista_perguntas = [
    "Eu sou geofísico e quero migrar para a área de dados. Qual linguagem de programação devo aprender primeiro?",
    "E que tipo de projeto de portfólio eu poderia criar usando essa linguagem?"
]

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é o 'GeoAI Mentor', um assistente especializado em ajudar geocientistas a migrar para a área de Ciência de Dados. Seja amigável, didático e responda sempre em até 5 linhas."
    ),
    (
        "placeholder",
        "{historico}"
    ),
    (
        "human",
        "{query}"
    )
])


chain = prompt | modelo | StrOutputParser()



from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

memoria_sessoes = {}

def obter_historico_por_sessao(session_id):
    if session_id not in memoria_sessoes:
        memoria_sessoes[session_id] = InMemoryChatMessageHistory()
    return memoria_sessoes[session_id]


cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=obter_historico_por_sessao,
    input_messages_key="query",
    history_messages_key="historico"
)

sessao = "usuario_1"

for pergunta in lista_perguntas:
    resposta = cadeia_com_memoria.invoke(
        input={"query": pergunta},
        config={"configurable": {"session_id": sessao}}
    )

    print("Pergunta:", pergunta)
    print("Resposta:", resposta, "\n")
    print("-" * 50)
