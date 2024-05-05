# app.py
import os
from flask import Flask, render_template, request, jsonify
import openai
from dotenv import load_dotenv
import logging
from anthropic import Client
import time

app = Flask(__name__, template_folder='.')
load_dotenv('_API_Keys.env')

logging.basicConfig(filename='app.log', level=logging.INFO)

class WerkzeugFilter(logging.Filter):
    def filter(self, record):
        return not (record.levelname == 'INFO' and 'werkzeug' in record.name and '127.0.0.1' in record.getMessage())

logging.getLogger('werkzeug').addFilter(WerkzeugFilter())

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

HUMAN_PROMPT = "Human: "
AI_PROMPT = "\nAssistant: "

class CollabAI:
    def __init__(self):
        self.manager = ClaudeOpus()
        self.worker = OpenAIWorker()

    def process_query(self, query):
        conversation_history = []
        num_iterations = 4

        context = "This is a CHAINED conversation between two LLMs, Claude (MANAGER) and ChatGPT (WORKER). Each RESPONSE from one LLM is used as the PROMPT for the next LLM, with a focus on maintaining responses within 1000 characters."

        manager_response = self.manager.generate_response(query, conversation_history, context)
        logging.info(f"Initial manager response: {manager_response}")
        if not manager_response:
            logging.warning("Empty initial manager response. Providing default response.")
            manager_response = f"Please provide a detailed itinerary for a summer trip to Iceland, focusing on the following interests mentioned in the original query: {query}"
        conversation_history.append({"role": "Claude", "content": manager_response})

        for i in range(num_iterations):
            worker_response = self.worker.generate_response(manager_response, max_retries=3, retry_delay=1)
            logging.info(f"Worker response: {worker_response}")
            conversation_history.append({"role": "OpenAI", "content": worker_response})

            if self.is_irrelevant(worker_response, query):
                logging.warning("Irrelevant worker response detected. Providing feedback to improve relevance.")
                feedback = self.manager.generate_irrelevance_feedback(worker_response, conversation_history, query)
            else:
                feedback = self.manager.generate_feedback(worker_response, conversation_history, query)
            logging.info(f"Manager feedback: {feedback}")
            conversation_history.append({"role": "Claude", "content": feedback})

            if self.manager.is_satisfactory(worker_response, query):
                conversation_history.append({"role": "System", "content": "Conversation terminated early due to satisfactory response."})
                break

            manager_response = self.manager.generate_response(feedback, conversation_history, query)
            logging.info(f"Manager response: {manager_response}")
            if not manager_response:
                logging.warning("Empty manager response. Providing default response.")
                manager_response = f"Please provide more specific and relevant suggestions for the Iceland trip itinerary, focusing on the user's interests mentioned in the original query: {query}"
            conversation_history.append({"role": "Claude", "content": manager_response})

        if i == num_iterations - 1:
            conversation_history.append({"role": "System", "content": "Conversation reached maximum number of iterations."})

        final_response_prompt = f"{HUMAN_PROMPT}Based on the conversation history and the original query: {query}, please provide a concise, informative, and engaging final response that directly addresses the user's interests in hiking, exploring natural wonders, and learning about the local culture in Iceland. Focus on synthesizing the key insights and recommendations from the dialogue into a cohesive and polished response, highlighting the most relevant and insightful takeaways. Please ensure that your response is a standalone answer to the original question, without any meta-commentary or references to the underlying process.\n\n{AI_PROMPT}"

        final_response = self.manager.generate_refined_response(final_response_prompt, conversation_history, query)
        logging.info(f"Final response: {final_response}")

        if not final_response:
            logging.warning("Empty final response. Providing default response.")
            final_response = "I apologize, but I don't have enough information to provide a detailed response. Could you please provide more context about your specific interests and preferences for your Iceland trip? For example, what type of hiking trails do you prefer (easy, moderate, challenging)? Are there any particular natural wonders you'd like to prioritize seeing? How much time do you have for your trip? Knowing more details will help me create a tailored itinerary that best suits your needs."

        return conversation_history, final_response, i + 1, num_iterations

    def is_irrelevant(self, response, query):
        prompt = f"{HUMAN_PROMPT}Given the original query: {query}, and the response: {response}, please determine if the response is irrelevant or off-topic. Respond with 'YES' if the response is irrelevant, or 'NO' if it is relevant.\n\n{AI_PROMPT}"
        relevance_check = self.manager.client.completions.create(
            prompt=prompt,
            max_tokens_to_sample=20,
            model="claude-v1.3"
        )
        return "yes" in relevance_check.completion.lower()

class ClaudeOpus:
    def __init__(self):
        self.client = Client(api_key=ANTHROPIC_API_KEY)

    def generate_response(self, query, conversation_history, context):
        prompt = self._build_prompt(query, conversation_history, context)

        prompt += "\n\nAs the MANAGER, your task is to guide the conversation towards providing the most helpful and insightful recommendations for the user's Iceland trip. Break down the task into subtasks focusing on the user's key interests in hiking, natural wonders, and local culture. Encourage ChatGPT to provide specific examples and details to justify its suggestions. Ask follow-up questions to gain more context about the user's preferences and constraints. Offer constructive feedback to improve the relevance and depth of ChatGPT's responses. Ensure the conversation stays on track and aligned with the original request."

        try:
            response = self.client.completions.create(
                prompt=prompt,
                max_tokens_to_sample=600,
                model="claude-v1.3"
            )
            generated_response = response.completion.strip()
            logging.info(f"Generated response from Claude: {generated_response}")

            if not generated_response:
                logging.warning("Empty response generated from Claude.")
                generated_response = "I apologize, but I need more information about your specific interests and preferences to provide helpful recommendations. Could you share more details about the type of hiking trails, natural wonders, and cultural experiences you're most interested in? Also, how much time do you have for your Iceland trip? I'm happy to offer tailored suggestions once I have a better understanding of your needs."

            return generated_response
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            if "invalid_request_error" in str(e) and "credit balance is too low" in str(e):
                return "Error: Insufficient Anthropic API credits. Please check your account balance and update the API key."
            else:
                return "Error generating response. Please try again."

    def generate_feedback(self, response, conversation_history, original_query):
        prompt = f"{HUMAN_PROMPT}You are an expert evaluator. The original query is: {original_query}. Here's a response from ChatGPT: {response}. Please provide a score from 1 to 5 for relevance and insightfulness, along with concrete examples of what strong relevance and insightfulness look like for this topic. The primary goal is to achieve consistent relevance and insightfulness scores of 4 or higher in every response cycle. Provide specific suggestions and examples to help ChatGPT improve its responses.\n\n{AI_PROMPT}"

        feedback = self.client.completions.create(
            prompt=prompt,
            max_tokens_to_sample=300,
            model="claude-v1.3"
        )
        return feedback.completion.strip()

    def generate_irrelevance_feedback(self, response, conversation_history, original_query):
        prompt = f"{HUMAN_PROMPT}You are an expert evaluator. The original query is: {original_query}. Here's a response from ChatGPT: {response}. The response seems to be irrelevant or off-topic. Please provide constructive feedback on how ChatGPT can improve the relevance of its response. Give specific examples and suggestions for more relevant content that directly addresses the user's interests in hiking, natural wonders, and local culture in Iceland. Explain how the improved response would better meet the user's needs.\n\n{AI_PROMPT}"

        feedback = self.client.completions.create(
            prompt=prompt,
            max_tokens_to_sample=300,
            model="claude-v1.3"
        )
        return feedback.completion.strip()

    def is_satisfactory(self, response, original_query):
        prompt = f"{HUMAN_PROMPT}You are an expert evaluator. Based on the original query: {original_query}, please assess the following response from ChatGPT: {response}. Is this response satisfactory in terms of directly addressing the query, providing valuable insights, and offering relevant recommendations for hiking, natural wonders, and local culture in Iceland? Respond with 'YES' if the response is satisfactory, or 'NO' if it needs improvement.\n\n{AI_PROMPT}"

        evaluation = self.client.completions.create(
            prompt=prompt,
            max_tokens_to_sample=20,
            model="claude-v1.3"
        )
        return "yes" in evaluation.completion.lower()

    def generate_refined_response(self, query, conversation_history, context):
        prompt = self._build_prompt(query, conversation_history, context)

        prompt += "\n\nBased on the conversation history and the original query, please provide a refined, polished, and engaging final response that synthesizes the key insights and recommendations for the user's Iceland trip. The response should directly address the user's interests in hiking, exploring natural wonders, and learning about the local culture. Highlight the most relevant and insightful takeaways from the dialogue, and present the information in a clear, concise, and well-structured manner. Ensure that your response is a standalone answer to the original question, without any meta-commentary or references to the underlying process."

        try:
            response = self.client.completions.create(
                prompt=prompt,
                max_tokens_to_sample=800,
                model="claude-v1.3"
            )
            generated_response = response.completion.strip()
            return generated_response
        except Exception as e:
            logging.error(f"Error generating refined response: {str(e)}")
            return "Error generating refined response. Please try again."

    def _build_prompt(self, query, conversation_history, context):
        prompt = f"{context}\n\n{HUMAN_PROMPT}{query}\n"
        for entry in conversation_history:
            role = entry["role"]
            response = entry["content"]
            prompt += f"{role}: {response}\n"
        prompt += f"{AI_PROMPT}"
        return prompt

class OpenAIWorker:
    def generate_response(self, query, max_retries=3, retry_delay=1):
        openai.api_key = OPENAI_API_KEY
        retries = 0
        while retries < max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": query}],
                    max_tokens=600,
                    temperature=0.7
                )
                generated_text = response.choices[0].message.content.strip()
                logging.info(f"Generated response from OpenAI: {generated_text}")
                return generated_text
            except Exception as e:
                logging.error(f"Error generating response: {str(e)}")
                retries += 1
                if retries < max_retries:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        return "Error generating response. Please try again."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.json['query']
        logging.info(f"Received query: {query}")
        collab_ai = CollabAI()
        try:
            conversation_history, final_response, current_iteration, total_iterations = collab_ai.process_query(query)
            return jsonify({
                'conversation': conversation_history,
                'final_response': final_response,
                'current_iteration': current_iteration,
                'total_iterations': total_iterations
            })
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return jsonify({'error': 'An error occurred while processing the query.'}), 500
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)