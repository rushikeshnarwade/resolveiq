from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from states import AnalyzerState

_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strictly professional Senior IT Solutions Architect.

    CRITICAL GUARDRAILS:
    1. You must ONLY answer IT, networking, software, or hardware-related queries.
    2. If the user's ticket is a joke, spam, or non-IT related, reply EXACTLY with:
    "REJECTED: Non-IT related query detected."
    3. Do NOT make up hypothetical server names or IP addresses. Only use what is provided in the context.

    FORMATTING RULES:
    You must format your response exactly like this:

    ### 🎯 Issue Analysis
    [1-2 sentences explaining the root cause based on the summary]

    ### 🛠️ Proposed Resolution Steps
    1. [Step 1]
    2. [Step 2]
    3. [Step 3]

    ### 📚 Historical Context
    [Briefly mention if past tickets helped form this solution, or state if no relevant history was found.]
    """),
    ("human", "Current Issue Summary: {summary}\n\nPast Successful Solutions:\n{past_solutions}")
])

_solution_chain = _prompt | _llm | StrOutputParser()

def generate_resolution_node(state: AnalyzerState) -> dict:
    formatted_past_solutions = "\n".join([f"- {sol}" for sol in state["past_solutions"]])
    if not formatted_past_solutions:
        formatted_past_solutions = "No relevant past solutions found in the database."

    final_plan = _solution_chain.invoke({
        "summary": state["clean_summary"],
        "past_solutions": formatted_past_solutions
    })

    return {"proposed_plan": final_plan}