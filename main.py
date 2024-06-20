#to run the following code use command : streamlit name_generator.py

import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from secret_key import openapi_key

#apikey.py is my file for storing my openai api secret key
# create your own key and store it in that file

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = openapi_key

# Initialize the OpenAI model
llm = OpenAI(temperature=0.7)

def generate_restaurant_name_and_items(cuisine):
    # Chain 1: Restaurant Name
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

    # Chain 2: Menu Items
    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest 10 menu items for {restaurant_name}. Return it as a comma separated string"
    )

    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

    chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', "menu_items"]
    )

    response = chain({'cuisine': cuisine})

    return response

# Streamlit app function
def main():
    st.title("Restaurant Name Generator")

    cuisine = st.sidebar.selectbox("Pick a cuisine", ("Indian", "Italian", "Mexican", "Chinese", "American", "French", "German", "Thai", "Japanese", "Korean", "Russian", "Spanish", "Turkish"))

    if cuisine:
        response = generate_restaurant_name_and_items(cuisine)
        st.header(response['restaurant_name'].strip())
        menu_items = response['menu_items'].strip().split(",")
        st.write("**Menu Items**")
        st.write("\n".join(item.strip() for item in menu_items))

# Run the Streamlit app
if __name__ == "__main__":
    main()




