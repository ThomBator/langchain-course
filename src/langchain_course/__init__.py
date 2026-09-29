from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

def main() -> None:
    print("Hello from langchain-course!")
    llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-3.1-flash-lite")
    # llm = ChatOllama(temperature=0, model="dna5rm/granite4.2:3b-8k")

    information = """
        Jeffrey Preston Bezos (/ˈbeɪzoʊs/ BAY-zohss;[2] né Jorgensen; born January 12, 1964) is an American businessman, and the founder, executive chairman, and former president and CEO of Amazon, the world's largest e-commerce and cloud computing company. According to the Bloomberg Billionaires Index and Forbes, he was the world's wealthiest person from 2017 to 2021,[3] and in 2026, his net worth was approximately US$284 billion.[4]

        Bezos was born in Albuquerque, and raised in Houston and Miami. He graduated from Princeton University in 1986 with a degree in engineering. He worked on Wall Street in a variety of related fields from 1986 to early 1994. He founded Amazon in mid-1994 on a road trip from New York City to Seattle. The company began as an online bookstore and expanded to a variety of other e-commerce products and services, including video and audio streaming, cloud computing, and artificial intelligence. It is the world's largest online sales company, the largest Internet company by revenue, and the largest provider of virtual assistants and cloud infrastructure services through its Amazon Web Services branch.

        Bezos founded the aerospace manufacturer and sub-orbital spaceflight services company Blue Origin in 2000, and he flew into space on Blue Origin NS-16 in 2021. He purchased the major American newspaper The Washington Post in 2013 and manages many other investments through his venture capital firm, Bezos Expeditions.

        The first centibillionaire on the Forbes Real Time Billionaires Index, Bezos was named the "richest man in modern history" after his net worth increased to $150 billion in July 2018 (equivalent to $190,000,000,000 in 2025).[5] On July 5, 2021, Bezos took over the role of executive chairman and stepped down as the CEO and president of Amazon, succeeded by Amazon Web Services CEO Andy Jassy.

    """

    summary_template = """
    Given the information provided about a given person I want you to create: 
    1. A short summary of the person in 2-3 sentences.
    2. 2 interesting facts about the person.

    Information:
    {information}
    """
    # teamplates are useful to limit risk of prompt injection attacks. They also make it easier to reuse prompts and keep them organized.
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    # Langchain expression language (LCEL) allows you to use a template to create a prompt and then call the llm with the prompt and the input variables. This is useful for creating prompts that are more complex than just a single string.
    # The chain requires an llm, a chain of elements and a response 

    chain = summary_prompt_template | llm 
    response = chain.invoke(input={"information": information})

    print(response.content)

    
# Not technically needed for uv but allows me to use the VS Code Run button to run the file directly
if __name__ == "__main__":
    main()