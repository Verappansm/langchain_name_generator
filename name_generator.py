#to run the following code use command : streamlit name_generator.py
#this is a code to generate names starting with a letter and ending with a letter you specify

"""import streamlit as st
import os
from langchain_openai.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from apikey import openai_key
#apikey.py is my file for storing my openai api secret key
# create your own key and store it in that file
st.title("Name Generator")
start = st.text_input("Enter start letter")
end = st.text_input("Enter last letter")

os.environ['OPENAI_API_KEY'] = openai_key #openai key here
llm = OpenAI(temperature=0.6)

def generate_name(start, end):
    prompt_template_name = PromptTemplate(
        input_variables=['x', 'y'],
        template="Suggest a single name starting with letter {x} and ending with letter {y}"
    )
    prompt_template_fnames = PromptTemplate(
        input_variables=['name'],
        template="for the name {name} add suitable last names and provide the list of full names 10 is enough  "
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="name")
    item_chain = LLMChain(llm=llm, prompt=prompt_template_fnames, output_key="fname")

    chain = SequentialChain(
        chains=[name_chain, item_chain],
        input_variables=['x', 'y'],
        output_variables=["name", "fname"]
    )

    return chain({'x': start, 'y': end})

if start and end and len(start)==1 and len(end)==1:
        response = generate_name(start, end)
        st.header(response['name'])
        fnames = response['fname'].split(",")
        
        for item in fnames:
            st.write(item)
else:
     st.write("*Enter valid inputs*")"""