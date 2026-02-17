from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

system_prompt = """You are Sora — a witty, clever, and helpful assistant.
Here’s how you operate:
    - FIRST and FOREMOST, figure out from the query asked whether it requires a look via the webcam to be answered, if yes call the analyze_image_with_query tool for it and proceed.
    - Don't ask for permission to look through the webcam; call it straight away.
    - Always present results (if they come from a tool) in a natural, witty, and human-sounding way.
"""

system_prompt_without_camera = """You are Sora — a witty, clever, and helpful assistant.
Here’s how you operate:
    - You will only answer based on text input and will never attempt to access the webcam or any external tools.
    - If a question requires a camera to answer, politely inform the user that you cannot access it.
    - Always present results naturally and human-sounding.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
)

def ask_agent(user_query: str) -> str:
    # Step 1: Decide if camera is needed
    pre_check_prompt = f"""
You are an assistant that decides whether a query absolutely requires camera access.
Answer only 'Yes' or 'No'. If unsure, answer 'No'.

Query: "{user_query}"
"""
    pre_check_response = llm.invoke([{"role": "user", "content": pre_check_prompt}])
    answer = pre_check_response.content.strip().lower()
    print("Camera required?", answer)

    # Step 2: Setup tools if needed
    if answer == "yes":
        from tools import get_analyze_image_with_query
        analyze_image_with_query = get_analyze_image_with_query()
        tools_list = [analyze_image_with_query]
        prompt_to_use = system_prompt
    else:
        tools_list = []
        prompt_to_use = system_prompt_without_camera

    # Step 3: Create agent and ask query
    agent = create_react_agent(
        model=llm,
        tools=tools_list,
        prompt=prompt_to_use
    )

    input_messages = {"messages": [{"role": "user", "content": user_query}]}
    response = agent.invoke(input_messages)

    return response['messages'][-1].content


# Example usage:
# print(ask_agent(user_query="Where is Portugal located?"))
# print(ask_agent(user_query="Can you describe my surroundings?"))
