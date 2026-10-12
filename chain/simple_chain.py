from langchain_ollama import ChatOllama

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(
    model="gemma2:2b",
    temperature=0.7,
)

prompt = PromptTemplate(
    template="Generate 5 Interesting fact about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser 

res = chain.invoke({'topic':"Sun"})

print(res)


chain.get_graph().print_ascii()    # it visuals the chain 


# output  

'''

(venv) PS E:\project 2026\gen-ai> python .\chain\simple_chain.py
Here are 5 interesting facts about the Sun:

1. **The Sun is a "star nursery"**:  The sun is actually the source of energy for life on Earth. It's also a massive and complex system that is constantly changing. 
2. **The Sun's magnetic field is incredibly powerful**:  The Sun's magnetic field is so strong that it can launch solar flares and coronal mass ejections.  These events can have a significant impact on Earth's technology and even our atmosphere.
3. **The Sun's core is incredibly dense**:  The sun's core is so dense that it's believed to be made up of solid iron. This incredible pressure and heat are what allow the sun to generate energy through nuclear fusion.
4. **The Sun is a giant ball of plasma**:  The Sun is made up of mostly hydrogen and helium gas that is constantly fusing together in its core, creating energy and heat. 
5. **The Sun has a 27-year cycle**:  There's a 27-year cycle in the Sun's activity called the solar cycle. This cycle sees the Sun's magnetic field change, leading to periods of increased activity and solar flares. 

Let me know if you'd like to hear more about the Sun! ☀️ 

     +-------------+       
     | PromptInput |       
     +-------------+       
            *              
            *              
            *              
    +----------------+     
    | PromptTemplate |     
    +----------------+     
            *              
            *              
            *              
      +------------+       
      | ChatOllama |       
      +------------+       
            *              
            *              
            *              
   +-----------------+     
   | StrOutputParser |     
   +-----------------+     
            *              
            *              
            *              
+-----------------------+  
| StrOutputParserOutput |  
+-----------------------+  
(venv) PS E:\project 2026\gen-ai> 

'''
