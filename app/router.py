from semantic_router import Route
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder

encoder = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")

faq = Route(
    name='faq',
    utterances=[
        "What is the return policy of the products?",
        "Do I get discount with the HDFC credit card?",
        "How can I track my order?",
        "What payment method are accepted?",
        "How long does it take to process a refund?",
        "What happens if I receive a damaged product?",
        "Exchange policy for defective items"
    ],
    score_threshold=0.3
)

sql = Route(
    name='sql',
    utterances=[
        "I want to buy nike shoes that have 50% discount.",
        "Are there any shoes under Rs. 3000?",
        "Do you have formal shoes in size 9?",
        "Are there any Puma shoes on sale?",
        "What is the price of puma running shoes?"
    ],
    score_threshold=0.3
)

router = SemanticRouter(encoder=encoder)

router.add(routes=[faq, sql])

if __name__ == "__main__":
    print("\n--- Testing Router ---")
    
    # Test 1
    query1 = "What is your policy on defective product?"
    classification1 = router(query1)
    print(f"Query: '{query1}'")
    print(f" -> Route Name: {classification1.name}")

    print("-" * 20)

    # Test 2
    query2 = "Pink puma shoes in price range 5000 to 10000"
    classification2 = router(query2)
    print(f"Query: '{query2}'")
    print(f" -> Route Name: {classification2.name}")