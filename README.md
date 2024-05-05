# COLLABAI - Collaborative AI System

COLLABAI is a collaborative AI system that enables multiple AI models to work together seamlessly to process user queries and generate accurate and validated responses. The system leverages the strengths of different AI models, including ChatGPT4, GeminiPro, and MetaAI, to provide a comprehensive and efficient solution.

## Features

- Task allocation by ChatGPT4 orchestrator
- Code generation by GeminiPro
- Creative writing and conversational tasks by ChatGPT4
- Research and advanced tasks by MetaAI
- Response validation by ChatGPT4

## Installation

1. Clone the repository:
git clone https://github.com/ademczuk/collabai.git

2. Install the required dependencies:
pip install -r requirements.txt

3. Set up the necessary API keys in the `.env` file:
CHATGPT_API_KEY=your_chatgpt_api_key
GEMINI_API_KEY=your_gemini_api_key
META_AI_API_KEY=your_meta_ai_api_key

4. Run the Flask application:
python app.py

5. Open your web browser and navigate to `http://localhost:5000` to access the COLLABAI interface.

## Usage

1. Enter your query in the provided text area on the COLLABAI web interface.
2. Click the "Submit" button to process your query.
3. The system will allocate the task to the appropriate AI model based on the query type.
4. The generated response will be validated and displayed on the web page.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).