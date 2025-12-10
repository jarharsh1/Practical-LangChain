from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# ------------- Setup -------------
load_dotenv()  # needs HUGGINGFACEHUB_API_TOKEN in your .env

# Model 1: Llama 3.1 – used for notes
llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",  # IMPORTANT: no :cerebras
    task="conversational",                       # provider only supports conversational
    max_new_tokens=256,
    do_sample=False,
)

model1 = ChatHuggingFace(llm=llm1)

# Model 2: Mistral 7B – used for questions
llm2 = HuggingFaceEndpoint(
    repo_id="EssentialAI/rnj-1-instruct",
    task="conversational",                       # also used via chat API
    max_new_tokens=256,
    do_sample=False,
)

model2 = ChatHuggingFace(llm=llm2)

# ------------- Prompts -------------
prompt1 = PromptTemplate(
    template="Generate short and simple notes for the following text:\n\n{text}",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template="Generate 5 short question–answer pairs from the following text:\n\n{text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template=(
        "Merge the following notes and questions into a single, well-formatted study document.\n\n"
        "Notes:\n{notes}\n\n"
        "Questions:\n{questions}\n"
    ),
    input_variables=["notes", "questions"],
)

parser = StrOutputParser()

# ------------- Chains (parallel + merge) -------------
parallel_chain = RunnableParallel(
    notes=prompt1 | model1 | parser,      # Llama 3.1 branch
    questions=prompt2 | model1 | parser,  # Mistral 7B branch
)

merge_chain = prompt3 | model1 | parser   # merge with model1 (can be model2 if you prefer)

chain = parallel_chain | merge_chain

# ------------- Input text -------------
text = """
Once upon a time, a lonely monkey lived on a big fruit tree that grew on the bank of a wide river.  
The tree gave sweet fruits all year, and the monkey spent his days eating, playing among the branches, and watching the river flow by.

One hot afternoon, a tired crocodile swam to the bank and rested under the tree.  
Seeing him, the monkey plucked a bunch of ripe fruits and tossed them down as a friendly gift.

The crocodile loved the taste and returned the next day, and the next.  
Slowly, the monkey and the crocodile became close friends: they talked every day, shared fruits, and told each other stories of life on land and in water.

One day, the crocodile carried some fruits home for his wife.  
She enjoyed them so much that a greedy thought came to her mind: “If these fruits are so tasty, how delicious must be the heart of the monkey who eats them every day!”

She demanded that her husband bring her the monkey’s heart to eat.  
The crocodile was shocked and refused at first, but his wife kept insisting, crying and pretending to be ill until he finally gave in.

With a heavy heart, the crocodile went to the tree the next day, hiding his true intention.  
He told the monkey, “My wife loved your fruits and wants to thank you in person. Please come home with me for a feast.”

The monkey felt honored but said, “I cannot swim across the river.”  
The crocodile replied, “Just sit on my back. I’ll carry you safely.”

Trusting his friend, the monkey climbed down and sat on the crocodile’s back.  
They started across the deep river, the water swirling around them.

In the middle of the river, the crocodile began to sink a little lower and swim more slowly.  
The monkey grew scared and asked, “Why are you going so deep? I will drown if I fall!”

Feeling that escape was impossible for the monkey now, the crocodile confessed everything.  
“My friend, forgive me. My wife says she will die unless she eats a monkey’s heart. I promised to bring her yours.”

The monkey was shocked and terrified, but he stayed calm and thought quickly.  
With a steady voice, he said, “Oh, why didn’t you tell me earlier? I would gladly save your wife. But you see, I keep my heart safely on the tree. I never carry it with me when I travel.”

The foolish crocodile believed this strange explanation.  
He turned around at once and swam back, thinking the monkey would fetch his heart and come again.

As soon as they reached the bank, the monkey leaped from the crocodile’s back and scampered up to the highest branch of the tree.  
Safe among the leaves, he shouted, “You foolish creature! No one can leave their heart hanging on a tree. You tried to betray a friend. Go away and never return!”

The crocodile realized his mistake, felt ashamed, and slowly swam back to his side of the river.  
The friendship was lost forever, and the monkey never trusted him again.
"""

if __name__ == "__main__":
    result = chain.invoke({"text": text})
    print(result)

    print("\n\nChain Graph:\n")
    chain.get_graph().print_ascii()
