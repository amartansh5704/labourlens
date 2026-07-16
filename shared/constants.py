# shared/constants.py
# Central place for all constant values used across the project

# ─────────────────────────────────────────────────────────
# JURISDICTIONS
# ─────────────────────────────────────────────────────────
JURISDICTIONS = {
    "Central": "Ministry of Labour and Employment, India",
    "Delhi": "Delhi NCR (Delhi + Haryana + UP portions)",
    "Maharashtra": "Maharashtra (Mumbai)",
    "Karnataka": "Karnataka (Bangalore)",
    "Tamil Nadu": "Tamil Nadu (Chennai)",
    "Telangana": "Telangana (Hyderabad)"
}

# just the names as a list
JURISDICTION_NAMES = list(JURISDICTIONS.keys())

# ─────────────────────────────────────────────────────────
# TOPICS
# ─────────────────────────────────────────────────────────
TOPICS = {
    "minimum_wage": "Minimum Wage",
    "working_hours": "Working Hours and Overtime",
    "leave_policy": "Leave Policies",
    "epf_esi": "EPF and ESI",
    "worker_classification": "Worker Classification"
}

# just the keys as a list
TOPIC_KEYS = list(TOPICS.keys())

# just the display names as a list
TOPIC_NAMES = list(TOPICS.values())

# ─────────────────────────────────────────────────────────
# DOCUMENT TYPES
# ─────────────────────────────────────────────────────────
DOCUMENT_TYPES = [
    "statute",        # actual law passed by parliament
    "notification",   # government notification/circular
    "amendment",      # changes to existing law
    "guidance",       # guidance notes from department
    "FAQ",            # frequently asked questions
    "circular",       # department circulars
    "order"           # government orders
]

# ─────────────────────────────────────────────────────────
# SOURCE URLS TO SCRAPE
# ─────────────────────────────────────────────────────────
SOURCE_URLS = {
    "Central": [
        "https://labour.gov.in",
        "https://epfindia.gov.in",
        "https://esic.gov.in",
        "https://clc.gov.in",
    ],
    "Delhi": [
        "https://labour.delhi.gov.in",
    ],
    "Maharashtra": [
        "https://mahakamgar.maharashtra.gov.in",
    ],
    "Karnataka": [
        "https://labour.kar.nic.in",
    ],
    "Tamil Nadu": [
        "https://labour.tn.gov.in",
    ],
    "Telangana": [
        "https://labour.telangana.gov.in",
    ]
}

# ─────────────────────────────────────────────────────────
# KEYWORD MAP
# helps detect topic from URL or page title
# ─────────────────────────────────────────────────────────
TOPIC_KEYWORDS = {
    "minimum_wage": [
        "minimum wage", "minimum wages", "minimum-wage",
        "wage rate", "wage notification", "wage revision",
        "minimum pay", "wage floor"
    ],
    "working_hours": [
        "working hours", "work hours", "overtime",
        "hours of work", "weekly hours", "daily hours",
        "spread over", "shift", "rest interval"
    ],
    "leave_policy": [
        "leave", "casual leave", "earned leave",
        "sick leave", "annual leave", "maternity leave",
        "paternity leave", "holiday", "PTO"
    ],
    "epf_esi": [
        "provident fund", "EPF", "ESI", "ESIC",
        "employee state insurance", "PF contribution",
        "gratuity", "social security"
    ],
    "worker_classification": [
        "contract labour", "contractor", "worker classification",
        "employee vs contractor", "gig worker", "fixed term",
        "temporary worker", "apprentice", "trainee"
    ]
}

# ─────────────────────────────────────────────────────────
# SCRAPER SETTINGS
# ─────────────────────────────────────────────────────────
SCRAPER_SETTINGS = {
    "DOWNLOAD_DELAY": 2,          # seconds between requests
    "RANDOMIZE_DOWNLOAD_DELAY": True,
    "CONCURRENT_REQUESTS": 1,     # one at a time for govt sites
    "ROBOTSTXT_OBEY": True,
    "TIMEOUT": 30,
    "MAX_RETRY": 3,
    "USER_AGENT": "LaborLens Research Bot 1.0 (Educational Project)"
}

# ─────────────────────────────────────────────────────────
# RAG SETTINGS
# ─────────────────────────────────────────────────────────
RAG_SETTINGS = {
    "CHUNK_SIZE": 512,
    "CHUNK_OVERLAP": 100,
    "TOP_K_RESULTS": 5,
    "MIN_SCORE_THRESHOLD": 0.4,   # below this score = not relevant
    "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
    "EMBEDDING_DIMENSION": 384,
    "EMBEDDING_PREFIX": "Represent this Indian legal document: "
}

# ─────────────────────────────────────────────────────────
# API SETTINGS
# ─────────────────────────────────────────────────────────
DISCLAIMER = (
    "⚠️ This information is for educational purposes only. "
    "It is not legal advice. Laws change frequently. "
    "Always consult a qualified lawyer for legal matters."
)

MAX_QUESTION_LENGTH = 500