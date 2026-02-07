"""
Literature-Based Construct Definitions for YouTube Comment Analysis
===================================================================

Seven constructs derived from peer-reviewed literature on social commerce,
purchase intention, and consumer behavior in short-form video platforms.

Each construct includes:
- Theoretical definition with literature source
- Inclusion keywords for classification
- Exclusion keywords to reduce false positives
"""

CONSTRUCTS = {
    "Trust": {
        "literature": (
            "Luo, C., Hasan, N.A.M., Zamri bin Ahmad, A.M. et al. "
            "Influence of short video content on consumers purchase intentions; "
            "Ghenie D., Avorniculei M., Sitar-Taut D. (2026), Influence of "
            "short-form video advertising on purchase intention in social commerce."
        ),
        "definition": (
            "Numerous studies have shown that rich and diverse video content "
            "and its presentation can significantly enhance consumers' trust "
            "in brands or products, thus further influencing their purchasing behavior."
        ),
        "inclusion_keywords": [
            "trust", "trusted", "trustworthy", "reliable", "scam",
            "confidence", "not as trusted", "not as reliable", "no trust",
            "undermines trust", "damaged trust", "don't trust",
            "how can i trust", "never trust",
        ],
        "exclusion_keywords": [],
    },
    "Perceived Quality": {
        "literature": (
            "Social media marketing and purchase intention: the mediation of "
            "perceived quality."
        ),
        "definition": (
            "Perceived quality is the customer's perception of the product's "
            "overall quality."
        ),
        "inclusion_keywords": [
            "quality", "durable", "premium", "performance",
            "low quality", "long-lasting", "weak", "innovation",
        ],
        "exclusion_keywords": [],
    },
    "Entertainment": {
        "literature": (
            "Luo, C., Hasan, N.A.M., Zamri bin Ahmad, A.M. et al. "
            "Influence of short video content on consumers purchase intentions; "
            "Social media marketing and purchase intention: the mediation of."
        ),
        "definition": (
            "Perceived enjoyment, fun, or amusement derived from brand content "
            "or interaction."
        ),
        "inclusion_keywords": [
            "fun", "entertaining", "funny", "incredible", "amazing",
            "awesome", "impressive", "enjoy", "catchy",
        ],
        "exclusion_keywords": [
            "function", "functionality", "functioning",
        ],
    },
    "Usefulness": {
        "literature": (
            "Luo, C., Hasan, N.A.M., Zamri bin Ahmad, A.M. et al. "
            "Influence of short video content on consumers purchase intentions."
        ),
        "definition": (
            "Usefulness refers to the extent to which the detailed product "
            "information presented in videos enables consumers to develop a "
            "clear understanding of product attributes."
        ),
        "inclusion_keywords": [
            "useful", "helps", "practical", "worth it", "works well",
            "convenient", "upgrade", "advantage", "don't use",
            "impractical", "over practical",
        ],
        "exclusion_keywords": [
            "used", "removing useful", "to use", "user",
        ],
    },
    "Price Value Perception": {
        "literature": (
            "Tian Hewei, Lee Youngsook, Factors affecting continuous purchase "
            "intention of fashion."
        ),
        "definition": (
            "Perceived value process is the process in which consumers make "
            "psychological judgments about costs or benefits based on price "
            "comparison in the process of purchasing products or services."
        ),
        "inclusion_keywords": [
            "value", "good for the price", "worth the price", "best value",
            "great values", "good price", "expensive", "affordable",
            "budget friendly", "money", "cheap", "cost", "price",
        ],
        "exclusion_keywords": [],
    },
    "eWOM/Recommendation": {
        "literature": (
            "Yaniv Gvili, Shalom Levy (2023), I Share, Therefore I Trust: "
            "A moderated mediation model."
        ),
        "definition": (
            "eWOM is defined as any positive or negative statement made by "
            "potential, actual, or former customers about a product or company, "
            "which is made available to a multitude of people and institutions."
        ),
        "inclusion_keywords": [
            "best brand", "recommend", "advice", "should buy",
            "worth buying", "resist buying", "worth to buy",
            "canceled order", "don't recommend", "avoid", "worst",
            "don't buy", "stop buying", "not buying",
            "no point in buying it", "regret buying",
            "won't be buying", "forces you to buy", "suggest",
        ],
        "exclusion_keywords": [
            "will buy", "to buy", "going to buy", "buying", "buy",
            "be buying",
        ],
    },
    "Product Information": {
        "literature": (
            "How does time pressure shape impulsive buying behavior? "
            "Hedonic vs. utilitarian perspectives."
        ),
        "definition": (
            "E-commerce platforms provide detailed product information, such as "
            "descriptions, specifications, images, videos, product reviews, and "
            "ratings. High-quality product information can serve as a key driver "
            "of informed purchase decisions."
        ),
        "inclusion_keywords": [
            "efficient", "specification", "feature", "features",
            "description", "details", "design", "battery", "processor",
            "storage", "camera", "display", "screen", "software",
            "dimension", "fast charging", "updates", "innovation",
        ],
        "exclusion_keywords": [],
    },
}

CONSTRUCT_NAMES = list(CONSTRUCTS.keys())

BRANDS = ["Xiaomi", "Samsung", "Huawei"]

EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

RANDOM_STATE = 42
