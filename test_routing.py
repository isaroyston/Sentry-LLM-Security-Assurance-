from src.chatbot.withdrawal_chatbot import WithdrawalChatbot
from src.vector_store.vector_store import VectorStore

def main():
    vs = VectorStore(persist_directory="vectordb")
    bot = WithdrawalChatbot(vector_store=vs)

    tests = [
        # emergency
        "I have a medical emergency, can I withdraw cash immediately?",
        # identity
        "What ID documents do I need and will I need OTP?",
        # fraud/monitoring
        "My withdrawal was put on hold due to AML checks. What happens next?",
        # general withdrawal
        "How much notice do I need for a large cash withdrawal?",
    ]

    for q in tests:
        print("=" * 80)
        print("Q:", q)
        print(bot.chat(q, debug=True))

if __name__ == "__main__":
    main()