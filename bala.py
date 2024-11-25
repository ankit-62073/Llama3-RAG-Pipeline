from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from src.model import create_llm
from langchain_core.messages import HumanMessage

app = Flask(__name__)
api = Api(app)

llm = create_llm()
# Define your chatbot logic here (synchronous version)
def chatbot_response(query):
    try:
        # Create LLM (Large Language Model)
        
        messages = [HumanMessage(content=query)]
        
        # Generate response synchronously (without 'await')
        response = llm.generate([messages])

        # Ensure the response is available
        if response:
            return {"status": "success", "response": response.generations[0][0].text}
        else:
            return {"status": "failure", "response": "No valid response from the model."}
    
    except Exception as e:
        return {"status": "failure", "response": str(e)}

# Define the Chatbot Resource
class Chatbot(Resource):
    def post(self):
        # Parse the JSON request
        data = request.get_json()

        # Ensure the 'query' key is present in the request body
        query = data.get('query')
        if not query:
            return {"error": "Query is required"}, 400

        # Get the chatbot response
        response = chatbot_response(query)

        # Return the response as JSON
        return {"query": query, "response": response}, 200

# Add the Chatbot resource to the API
api.add_resource(Chatbot, '/chatbot')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
